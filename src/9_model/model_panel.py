"""Borrower-quarter panel for the early-warning scorecard. One row per borrower and quarter t,
every BDC's loans to the borrower pooled. The warning signs use only what is filed for quarter t;
the one outcome is entry into non-accrual at t + 1. A borrower that leaves the data is classified
(repaid, refinanced, impaired or unclassified), not dropped. Built from the tranche-quarter panel
(monitor_panel.build_monitor_panel) and the FINAL_DATASET rows. Definitions and sample rules:
docs/14_early_warning_model.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

MAX_PRICE = 1.5             # FV / PAR above this is par left in local currency (cleaning issue): no price
MIN_PRIOR_FV = 1_000_000    # sample: the borrower's debt fair value at t is at least $1m
MIN_PRICED_SHARE = 0.5      # sample: priceable par is at least half of the borrower's funded par at t
MAX_CLO_SHARE = 0.5         # sample: CLO / structured tranches under half of par (CLO vehicles are not operating borrowers)
DD_WINDOW = 4               # rolling max drawdown over the preceding four quarters
LOW_MARK = 0.90             # PIK-switch hypotheses: mark below 90% of par ...
MARKDOWN = 0.05             # ... or marked down at least five points in the quarter
JUNIOR_LIENS = {"2L", "SUB", "UNSEC"}
# Debt reported without par is marked at FV / cost (the tranche panel holds only rows with par).
# OTHER_UNKNOWN rows without par are mostly equity and warrants, and CLO rows are not operating debt.
NO_PAR_DEBT = {"SENIOR_SECURED_1L", "UNITRANCHE", "SENIOR_SECURED_2L", "SUBORDINATED_MEZZ", "SENIOR_UNSECURED",
               "SENIOR_SECURED_UNSPEC"}
# Exits: a borrower held at t that no BDC holds at t + 1, classified by its last mark and maturity
IMPAIRED_MARK = 0.80        # left marked below 80: impaired
REPAID_MARK = 0.95          # left marked at 95 or above: paid off, at maturity (repaid) or before it (refinanced)
MATURITY_WINDOW = 2         # "at maturity": its last loan matures by the end of quarter t + 2
EXIT_TYPES = ("repaid", "refinanced", "impaired", "unclassified")


def q_label(qsort) -> list[str]:
    """qsort (year * 4 + quarter) back to 'YYYYQn'."""
    q = np.asarray(qsort, dtype=int)
    return [f"{y}Q{n}" for y, n in zip((q - 1) // 4, (q - 1) % 4 + 1)]


def q_sort(label: str) -> int:
    return int(label[:4]) * 4 + int(label[-1])


def q_next(label: str) -> str:
    return q_label([q_sort(label) + 1])[0]


def q_end(qsort) -> pd.DatetimeIndex:
    """Last day of each quarter."""
    q = np.asarray(qsort, dtype=int)
    return pd.PeriodIndex(year=(q - 1) // 4, quarter=(q - 1) % 4 + 1, freq="Q").end_time.normalize()


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def _wmean(df: pd.DataFrame, col: str, w: str, keys: list[str]) -> pd.Series:
    d = df[df[col].notna() & (df[w] > 0)]
    return (d[col] * d[w]).groupby([d[k] for k in keys]).sum() / d[w].groupby([d[k] for k in keys]).sum()


# ------------------------------------------------------------------ inputs from FINAL_DATASET rows

def debt_rows(rows: pl.DataFrame) -> pd.DataFrame:
    """Debt rows (PAR > 0) with a borrower name: borrower, qsort, cik, FV, non_accrual (1 / 0 / NaN).
    All of them, funded or not and including rows marked at zero, so a loan written off still
    counts for non-accrual and for the borrower's fair value."""
    r = rows.select(["cik", "cal_q", "borrower_canonical", "PAR", "FV",
                     pl.col("non_accrual").cast(pl.Float64, strict=False)]).to_pandas()
    r = r[(r["PAR"] > 0) & r["borrower_canonical"].notna()]
    r = r.assign(borrower=_norm(r["borrower_canonical"]),
                 qsort=r["cal_q"].str[:4].astype(int) * 4 + r["cal_q"].str[-1].astype(int))
    return r[r["borrower"] != ""][["borrower", "qsort", "cik", "FV", "non_accrual"]]


def lender_status(d: pd.DataFrame) -> pd.DataFrame:
    """One row per borrower x quarter x BDC: ls = 1 if the BDC flags any of its loans to the
    borrower non-accrual, 0 if its status is known and none is flagged, NaN if unknown (the
    filing marks non-accrual only in its HTML)."""
    g = d.groupby(["borrower", "qsort", "cik"])["non_accrual"].agg(["count", "max"]).reset_index()
    g["ls"] = g["max"].where(g["count"] > 0)
    return g[["borrower", "qsort", "cik", "ls"]]


def borrower_status(d: pd.DataFrame, ls: pd.DataFrame) -> pd.DataFrame:
    """Per borrower x quarter: na_status (1 if any lender flags it, 0 if some lender's status is
    known and none flags it, NaN if no lender's status is known), na_ever (flagged at or before
    t), fv_debt (fair value of all its debt rows) and the lenders with known status."""
    s = ls.groupby(["borrower", "qsort"])["ls"].agg(na_status="max", n_lenders_known="count")
    s["fv_debt"] = d.groupby(["borrower", "qsort"])["FV"].sum()
    s = s.reset_index().sort_values(["borrower", "qsort"])
    s["na_ever"] = s["na_status"].fillna(0).groupby(s["borrower"]).cummax()
    return s


def same_lender_entries(ls: pd.DataFrame) -> pd.DataFrame:
    """(borrower, qsort = t) where a BDC that reported the borrower accruing at t flags it
    non-accrual at t + 1: the strict version of a new entry."""
    known0 = ls.loc[ls["ls"] == 0, ["borrower", "qsort", "cik"]]
    flag1 = ls.loc[ls["ls"] == 1, ["borrower", "qsort", "cik"]].assign(qsort=lambda x: x["qsort"] - 1)
    return known0.merge(flag1, on=["borrower", "qsort", "cik"])[["borrower", "qsort"]].drop_duplicates()


def no_par_debt(rows: pl.DataFrame) -> pd.DataFrame:
    """Per borrower x quarter: fair value (fv_nopar) and cost (cost_nopar) of funded debt reported
    without par, which the mark takes at FV / cost."""
    r = rows.select(["cal_q", "borrower_canonical", "instrument_seniority", "PAR", "FV", "COST"]).to_pandas()
    r = r[~(r["PAR"] > 0) & (r["FV"] > 0) & (r["COST"] > 0) & r["instrument_seniority"].isin(NO_PAR_DEBT)
          & r["borrower_canonical"].notna()]
    r = r.assign(borrower=_norm(r["borrower_canonical"]),
                 qsort=r["cal_q"].str[:4].astype(int) * 4 + r["cal_q"].str[-1].astype(int))
    return r.groupby(["borrower", "qsort"]).agg(fv_nopar=("FV", "sum"), cost_nopar=("COST", "sum")).reset_index()


# ------------------------------------------------------------------ predictors from the tranche panel

def _named(tr: pd.DataFrame) -> pd.DataFrame:
    t = tr[~tr["borrower_key"].str.startswith("ID: ")].copy()
    t["borrower"] = _norm(t["borrower_key"])
    return t


def borrower_marks(t: pd.DataFrame) -> pd.DataFrame:
    """Per borrower x quarter over funded loans: fv, par, price (= sum FV / sum PAR over priceable
    loans, missing above MAX_PRICE), priced_share, junior_share (2L, subordinated and unsecured
    share of priced par), CLO share, PIK switch, dispersion, crowding (n_families: manager families
    lending to it), size and the latest maturity of its loans."""
    keys = ["borrower", "qsort"]
    t = t.assign(_junior=t["PAR_px"].where(t["lien_group"].isin(JUNIOR_LIENS), 0.0),
                 _clo=t["PAR"].where(t["lien_group"] == "CLO", 0.0),
                 _seasoned=t["link_rule"] > 0,
                 _switch=t["is_distress_amendment"].fillna(False).astype(bool))
    b = t.groupby(keys).agg(fv=("FV", "sum"), par=("PAR", "sum"),
                            fv_px=("FV_px", "sum"), par_px=("PAR_px", "sum"), junior_par=("_junior", "sum"),
                            clo_par=("_clo", "sum"), n_bdcs=("cik", "nunique"), n_families=("family", "nunique"),
                            seasoned=("_seasoned", "max"), pik_switch=("_switch", "max"),
                            last_maturity=("maturity", "max"))
    price = (b["fv_px"] / b["par_px"]).where(b["par_px"] > 0)
    b["price"] = price.where(price <= MAX_PRICE)
    b["priced_share"] = b["par_px"] / b["par"]
    b["junior_share"] = (b.pop("junior_par") / b["par_px"]).where(b["par_px"] > 0)
    b["clo_share"] = b.pop("clo_par") / b["par"]
    b["log_par"] = np.log(b["par"])
    b["pik_switch"] = b["pik_switch"].astype(float).where(b["seasoned"])   # undefined on a borrower's first quarter
    b["dispersion"] = _wmean(t, "V1_std", "PAR", keys)          # std of family marks, comparable tranches, 2+ families
    b["dispersion_range"] = _wmean(t, "V1_range", "PAR", keys)
    b["has_dispersion"] = b["dispersion"].notna().astype(int)
    b["sector"] = t.sort_values("PAR", ascending=False).drop_duplicates(keys).set_index(keys)["sector"]
    return b.reset_index()


def chain_changes(t: pd.DataFrame) -> pd.DataFrame:
    """Per borrower x quarter t, over the same loans (chains) held at t - 1 and t:
    d_price = par-weighted change in FV / PAR (weights: priced par at t - 1), link_par_prev =
    the par those loans carried at t - 1, d_pik_pp = par-weighted change in the PIK share of the
    coupon (percentage points). Draws, repayments and loans entering or leaving do not move them."""
    keys = ["borrower", "qsort"]
    c = t[["chain_id", "qsort", "borrower", "FV_px", "PAR_px", "PAR", "pik_share"]].copy()
    c["cprice"] = (c["FV_px"] / c["PAR_px"]).where(c["PAR_px"] > 0)
    c["cprice"] = c["cprice"].where(c["cprice"] <= MAX_PRICE)
    prev = c[["chain_id", "qsort", "cprice", "PAR_px", "PAR", "pik_share"]].assign(qsort=c["qsort"] + 1)
    m = c.merge(prev, on=["chain_id", "qsort"], suffixes=("", "_prev"))     # consecutive quarters only
    px = m[m["cprice"].notna() & m["cprice_prev"].notna()]
    w = px["PAR_px_prev"]
    out = pd.DataFrame({"d_price": (w * (px["cprice"] - px["cprice_prev"])).groupby([px[k] for k in keys]).sum()
                        / w.groupby([px[k] for k in keys]).sum(),
                        "link_par_prev": w.groupby([px[k] for k in keys]).sum()})
    pk = m[m["pik_share"].notna() & m["pik_share_prev"].notna() & (m["PAR_prev"] > 0)]
    wk = pk["PAR_prev"]
    out = out.join((100 * (wk * (pk["pik_share"] - pk["pik_share_prev"])).groupby([pk[k] for k in keys]).sum()
                    / wk.groupby([pk[k] for k in keys]).sum()).rename("d_pik_pp"), how="outer")
    return out.reset_index()


def rolling_max_drawdown(d: pd.DataFrame, col: str = "d_price", window: int = DD_WINDOW) -> pd.DataFrame:
    """Largest peak-to-trough fall of a borrower's chain-linked mark over quarters t - window .. t,
    in price points (0.05 = five points of par), 0 if it never fell. The index cumulates the
    same-loan changes `col`; a quarter without one adds nothing. dd_n_changes counts the observed
    changes in the window; the drawdown is missing when there are none."""
    w = d.pivot(index="borrower", columns="qsort", values=col)
    w = w.reindex(columns=range(w.columns.min(), w.columns.max() + 1))
    seen = w.notna().to_numpy()
    level = w.fillna(0.0).cumsum(axis=1).to_numpy()
    dd = np.full(level.shape, np.nan)
    n = np.zeros(level.shape)
    for j in range(level.shape[1]):
        lo = max(0, j - window)
        win = level[:, lo:j + 1]
        dd[:, j] = (np.maximum.accumulate(win, axis=1) - win).max(axis=1)
        n[:, j] = seen[:, lo + 1:j + 1].sum(axis=1) if j > lo else 0
    dd[n == 0] = np.nan
    out = pd.DataFrame({"borrower": np.repeat(w.index.to_numpy(), w.shape[1]),
                        "qsort": np.tile(w.columns.to_numpy(), w.shape[0]),
                        "dd": dd.ravel(), "dd_n_changes": n.ravel()})
    return out


# ------------------------------------------------------------------ the panel

def classify_exits(b: pd.DataFrame) -> pd.Series:
    """exit_type for borrower-quarters whose borrower no BDC holds at t + 1 (missing otherwise, and
    in the last quarter of the data). impaired: on non-accrual, or marked below IMPAIRED_MARK.
    Marked at REPAID_MARK or above, it was paid off: repaid if its last loan matures by the end of
    quarter t + MATURITY_WINDOW, refinanced if it left before that (or reports no maturity).
    unclassified: a mark in between, or no mark."""
    near_par = b["price"] >= REPAID_MARK
    matured = b["last_maturity"] <= pd.Series(q_end(b["qsort"] + MATURITY_WINDOW), index=b.index)
    kind = np.select([(b["na_status"] == 1) | (b["price"] < IMPAIRED_MARK), near_par & matured, near_par],
                     ["impaired", "repaid", "refinanced"], "unclassified")
    exited = ~b["present_next"] & (b["qsort"] < b["qsort"].max())
    return pd.Series(kind, index=b.index).where(exited)


def build_model_panel(tr: pd.DataFrame, rows: pl.DataFrame) -> pd.DataFrame:
    """tr: monitor_panel.build_monitor_panel's tranche-quarter panel; rows: its FINAL_DATASET rows.

    Signs at t: price (mark level), d_price (its change over the quarter, same loans),
    max_drawdown, pik_switch (alternative d_pik_pp), dispersion (alternative dispersion_range;
    has_dispersion = 1 when defined), crowding, plus market-adjusted versions (price_rel,
    d_price_adj) and controls.
    Outcome: y_na = 1 if an accruing borrower is on non-accrual at t + 1 or left the data impaired
    (a likely realized loss), 0 if it is still accruing or left paid off (repaid or refinanced),
    missing if it left unclassified or its status at t + 1 is unknown. Robustness:
    y_na_impaired_unlabelled (impaired exits get no label), y_na_first, y_na_same_lender. Sample
    flags: in_sample, na_risk_set."""
    t = _named(tr)
    b = borrower_marks(t)
    # the mark: FV / par on loans with par, FV / cost on debt without par
    b = b.merge(no_par_debt(rows), on=["borrower", "qsort"], how="left").fillna({"fv_nopar": 0.0, "cost_nopar": 0.0})
    price = ((b["fv_px"] + b["fv_nopar"]) / (b["par_px"] + b["cost_nopar"])).where(b["par_px"] + b["cost_nopar"] > 0)
    b["price"] = price.where(price <= MAX_PRICE)
    b["priced_share"] = (b["par_px"] + b["cost_nopar"]) / (b["par"] + b["cost_nopar"])
    ch = chain_changes(t)
    b = b.merge(ch, on=["borrower", "qsort"], how="left")

    # market-adjusted changes and level: the quarter's median borrower subtracted
    b["d_price_adj"] = b["d_price"] - b.groupby("qsort")["d_price"].transform("median")
    b["price_rel"] = b["price"] - b.groupby("qsort")["price"].transform("median")
    dd = rolling_max_drawdown(b, "d_price").rename(columns={"dd": "max_drawdown"})
    b = b.merge(dd, on=["borrower", "qsort"], how="left")
    b["recent_markdown"] = (b["d_price"] <= -MARKDOWN).astype(float).where(b["d_price"].notna())

    # non-accrual status and debt fair value from every debt row
    d = debt_rows(rows)
    ls = lender_status(d)
    s = borrower_status(d, ls)
    b = b.merge(s, on=["borrower", "qsort"], how="left")

    # outcomes at t + 1
    nxt_s = s.assign(qsort=s["qsort"] - 1)[["borrower", "qsort", "na_status"]].rename(
        columns={"na_status": "na_status_next"})
    nxt_s["present_next"] = True
    b = b.merge(nxt_s, on=["borrower", "qsort"], how="left")
    b["present_next"] = b["present_next"].fillna(False).astype(bool)
    b["exit_type"] = classify_exits(b)

    paid_off = b["exit_type"].isin(["repaid", "refinanced"])
    impaired = b["exit_type"] == "impaired"
    b["y_na"] = b["na_status_next"].mask(paid_off, 0.0).mask(impaired, 1.0).where(b["na_status"] == 0)
    b["y_na_impaired_unlabelled"] = b["y_na"].mask(impaired)
    b["y_na_first"] = b["y_na"].where(b["na_ever"] == 0)
    same = same_lender_entries(ls).assign(_same=1.0)
    b = b.merge(same, on=["borrower", "qsort"], how="left")
    b["y_na_same_lender"] = b["y_na"].where((b["y_na"] == 0) | (b["_same"] == 1))   # other entries dropped
    b = b.drop(columns="_same")

    b["crowding"] = b["n_families"].astype(float)

    # sample: an operating borrower, priced at t, large enough, and with the same loans priced the
    # quarter before (change in the mark and PIK switch defined)
    b["in_sample"] = ((b["clo_share"] < MAX_CLO_SHARE) & b["price"].notna()
                      & (b["priced_share"] >= MIN_PRICED_SHARE) & (b["fv_debt"] >= MIN_PRIOR_FV)
                      & b["d_price"].notna() & b["pik_switch"].notna())
    b["na_risk_set"] = b["in_sample"] & (b["na_status"] == 0)

    b["cal_q"] = q_label(b["qsort"])
    b["outcome_q"] = q_label(b["qsort"] + 1)
    first = ["borrower", "cal_q", "outcome_q", "qsort", "sector", "in_sample", "na_risk_set",
             "y_na", "exit_type", "y_na_impaired_unlabelled", "y_na_first", "y_na_same_lender",
             "price", "d_price", "pik_switch", "dispersion", "has_dispersion", "crowding", "max_drawdown",
             "d_pik_pp", "dispersion_range", "price_rel", "d_price_adj", "junior_share", "recent_markdown"]
    return b[first + [c for c in b.columns if c not in first]].sort_values(["qsort", "borrower"]).reset_index(drop=True)


def sample_waterfall(m: pd.DataFrame) -> pd.DataFrame:
    """Borrower-quarters (with a next quarter in the data) left after each sample rule, and the
    new non-accrual entries among them. The last step keeps the labelled rows: status known at
    t + 1, or the borrower left paid off or impaired."""
    m = m[m["qsort"] < m["qsort"].max()]
    steps = [
        ("named borrower-quarters, funded debt", pd.Series(True, index=m.index)),
        ("not a CLO / structured vehicle", m["clo_share"] < MAX_CLO_SHARE),
        ("price at t (priceable par, FV/PAR <= 1.5)", m["price"].notna()),
        (f"priced share >= {MIN_PRICED_SHARE:.0%} of funded par", m["priced_share"] >= MIN_PRICED_SHARE),
        (f"debt fair value >= ${MIN_PRIOR_FV / 1e6:.0f}m", m["fv_debt"] >= MIN_PRIOR_FV),
        ("same loans priced the quarter before (change in mark, PIK switch defined)",
         m["d_price"].notna() & m["pik_switch"].notna()),
        ("non-accrual status known at t and accruing (risk set)", m["na_status"] == 0),
        ("labelled: status known at t + 1, or left paid off or impaired", m["y_na"].notna()),
    ]
    keep = pd.Series(True, index=m.index)
    out = []
    for name, cond in steps:
        keep &= cond.fillna(False).astype(bool)
        k = m[keep]
        out.append({"step": name, "borrower_quarters": int(keep.sum()), "borrowers": k["borrower"].nunique(),
                    "new_nonaccruals": int((k["na_status"].eq(0) & k["na_status_next"].eq(1)).sum())})
    return pd.DataFrame(out)


def event_counts(m: pd.DataFrame) -> pd.DataFrame:
    """Per predictor quarter, what happened by t + 1 to every accruing borrower in the sample:
    entered non-accrual, still accruing, each kind of exit, and status unknown at t + 1 (these
    seven add up to at_risk). events = entered non-accrual + impaired exits; labelled = events +
    still accruing + repaid + refinanced. Unclassified exits and unknown status have no label."""
    r = m[m["na_risk_set"] & (m["qsort"] < m["qsort"].max())]
    held = r["exit_type"].isna()
    cols = {"at_risk": pd.Series(True, index=r.index), "entered_non_accrual": held & (r["na_status_next"] == 1),
            "still_accruing": held & (r["na_status_next"] == 0),
            **{f"exit_{k}": r["exit_type"] == k for k in EXIT_TYPES},
            "status_unknown": held & r["na_status_next"].isna(), "events": r["y_na"] == 1,
            "labelled": r["y_na"].notna()}
    t = pd.DataFrame(cols).groupby([r["cal_q"], r["outcome_q"]]).sum().reset_index()
    t["event_rate"] = t["events"] / t["labelled"]
    total = t.drop(columns=["cal_q", "outcome_q", "event_rate"]).sum()
    total["event_rate"] = total["events"] / total["labelled"]
    return pd.concat([t, pd.DataFrame([{"cal_q": "all", "outcome_q": "", **total}])], ignore_index=True)
