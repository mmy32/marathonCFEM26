"""Borrower-quarter panel for the early-warning scorecard. One row per borrower and quarter t,
every BDC's loans to the borrower pooled. The warning signs use only what is filed for quarter t;
the one outcome is entry into non-accrual within the next two quarters (t + 1 or t + 2), read from
the filings' non-accrual flags and never from a mark. A borrower that leaves every BDC's books
before the window closes has no label (the filings do not say whether it was repaid, refinanced or
sold at a loss); its exit is typed from its scheduled maturity only. Built from the tranche-quarter panel
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
HORIZON = 2                 # the outcome window: non-accrual at t + 1 or t + 2
WINDOW_TEXT = "within the next two quarters"
DD_WINDOW = 4               # rolling max drawdown over the preceding four quarters
LOW_MARK = 0.90             # PIK-switch hypotheses: mark below 90% of par ...
MARKDOWN = 0.05             # ... or marked down at least five points in the quarter
JUNIOR_LIENS = {"2L", "SUB", "UNSEC"}
# Debt reported without par is marked at FV / cost (the tranche panel holds only rows with par).
# OTHER_UNKNOWN rows without par are mostly equity and warrants, and CLO rows are not operating debt.
NO_PAR_DEBT = {"SENIOR_SECURED_1L", "UNITRANCHE", "SENIOR_SECURED_2L", "SUBORDINATED_MEZZ", "SENIOR_UNSECURED",
               "SENIOR_SECURED_UNSPEC"}


def q_label(qsort) -> list[str]:
    """qsort (year * 4 + quarter) back to 'YYYYQn'."""
    q = np.asarray(qsort, dtype=int)
    return [f"{y}Q{n}" for y, n in zip((q - 1) // 4, (q - 1) % 4 + 1)]


def q_sort(label: str) -> int:
    return int(label[:4]) * 4 + int(label[-1])


def q_next(label: str, k: int = 1) -> str:
    return q_label([q_sort(label) + k])[0]


def window_label(qsort) -> list[str]:
    """The outcome window of signal quarter t: 't+1-t+HORIZON', e.g. '2026Q1-2026Q2'."""
    q = np.asarray(qsort, dtype=int)
    return [f"{a}-{b}" for a, b in zip(q_label(q + 1), q_label(q + HORIZON))]


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


def same_lender_entries(ls: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """(borrower, qsort = t) where a BDC that reported the borrower accruing at t flags it
    non-accrual at t + 1 .. t + horizon: the strict version of a new entry."""
    keys = ["borrower", "qsort", "cik"]
    known0 = ls.loc[ls["ls"] == 0, keys]
    flags = pd.concat([ls.loc[ls["ls"] == 1, keys].assign(qsort=lambda x, k=k: x["qsort"] - k)
                       for k in range(1, horizon + 1)]).drop_duplicates()
    return known0.merge(flags, on=keys)[["borrower", "qsort"]].drop_duplicates()


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
    lending to it), size and maturity_q (quarter of its latest scheduled maturity)."""
    keys = ["borrower", "qsort"]
    mat = pd.to_datetime(t["maturity"], errors="coerce")
    t = t.assign(_matq=mat.dt.year * 4 + mat.dt.quarter,
                 _junior=t["PAR_px"].where(t["lien_group"].isin(JUNIOR_LIENS), 0.0),
                 _clo=t["PAR"].where(t["lien_group"] == "CLO", 0.0),
                 _seasoned=t["link_rule"] > 0,
                 _switch=t["is_distress_amendment"].fillna(False).astype(bool))
    b = t.groupby(keys).agg(fv=("FV", "sum"), par=("PAR", "sum"),
                            fv_px=("FV_px", "sum"), par_px=("PAR_px", "sum"), junior_par=("_junior", "sum"),
                            clo_par=("_clo", "sum"), n_bdcs=("cik", "nunique"), n_families=("family", "nunique"),
                            seasoned=("_seasoned", "max"), pik_switch=("_switch", "max"),
                            maturity_q=("_matq", "max"))
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

def build_model_panel(tr: pd.DataFrame, rows: pl.DataFrame) -> pd.DataFrame:
    """tr: monitor_panel.build_monitor_panel's tranche-quarter panel; rows: its FINAL_DATASET rows.

    Signs at t: price (mark level), d_price (its change over the quarter, same loans),
    max_drawdown, pik_switch (alternative d_pik_pp), dispersion (alternative dispersion_range;
    has_dispersion = 1 when defined), crowding, plus market-adjusted versions (price_rel,
    d_price_adj) and controls.
    Outcome: y_na = 1 if an accruing borrower is flagged non-accrual by any BDC at t + 1 or
    t + 2, 0 if it is held and known accruing in both quarters, missing otherwise: it left every
    BDC's books before the window closed without a flag (exited; exit_type from its scheduled
    maturity), its status is unknown, or the window runs past the data. A mark never enters the
    outcome. Robustness: y_na_first, y_na_same_lender. Sample flags: in_sample, na_risk_set."""
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

    # outcomes over t + 1 .. t + HORIZON
    for k in range(1, HORIZON + 1):
        nk = s.assign(qsort=s["qsort"] - k)[["borrower", "qsort", "na_status"]].rename(
            columns={"na_status": f"na_status_{k}"})
        nk[f"held_{k}"] = True
        b = b.merge(nk, on=["borrower", "qsort"], how="left")
        b[f"held_{k}"] = b[f"held_{k}"].fillna(False).astype(bool)
    st = b[[f"na_status_{k}" for k in range(1, HORIZON + 1)]]
    held = b[[f"held_{k}" for k in range(1, HORIZON + 1)]].to_numpy()
    b["window_complete"] = b["qsort"] + HORIZON <= max(b["qsort"].max(), s["qsort"].max())
    b["flag_in_window"] = (st == 1).any(axis=1)
    b["entry_q"] = (b["qsort"] + (st == 1).to_numpy().argmax(axis=1) + 1).where(b["flag_in_window"])
    clean = (st == 0).all(axis=1)
    y = pd.Series(np.select([b["flag_in_window"], clean], [1.0, 0.0], np.nan), index=b.index)
    b["y_na"] = y.where((b["na_status"] == 0) & b["window_complete"])
    # exits: left every BDC's books before the window closed, with no flag in it
    b["exited"] = b["window_complete"] & ~b["flag_in_window"] & ~held.all(axis=1)
    b["exit_q"] = (b["qsort"] + (~held).argmax(axis=1) + 1).where(b["exited"])
    b["exit_type"] = np.select(
        [~b["exited"], b["maturity_q"].isna(), b["maturity_q"] <= b["exit_q"]],
        ["", "maturity not reported", "at scheduled maturity"], "before scheduled maturity")
    b["na_status_next"] = b["na_status_1"]
    b["has_next"] = b["held_1"]

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
    b["outcome_q"] = window_label(b["qsort"])
    first = ["borrower", "cal_q", "outcome_q", "qsort", "sector", "in_sample", "na_risk_set",
             "y_na", "exited", "exit_type", "y_na_first", "y_na_same_lender",
             "price", "d_price", "pik_switch", "dispersion", "has_dispersion", "crowding", "max_drawdown",
             "d_pik_pp", "dispersion_range", "price_rel", "d_price_adj", "junior_share", "recent_markdown"]
    return b[first + [c for c in b.columns if c not in first]].sort_values(["qsort", "borrower"]).reset_index(drop=True)


def sample_waterfall(m: pd.DataFrame) -> pd.DataFrame:
    """Borrower-quarters (with the whole outcome window in the data) left after each sample rule,
    and the new non-accrual entries within the window among them. The last step keeps the labelled
    rows: flagged in the window, or held and known accruing through it."""
    m = m[m["window_complete"].astype(bool)]
    steps = [
        ("named borrower-quarters, funded debt", pd.Series(True, index=m.index)),
        ("not a CLO / structured vehicle", m["clo_share"] < MAX_CLO_SHARE),
        ("price at t (priceable par, FV/PAR <= 1.5)", m["price"].notna()),
        (f"priced share >= {MIN_PRICED_SHARE:.0%} of funded par", m["priced_share"] >= MIN_PRICED_SHARE),
        (f"debt fair value >= ${MIN_PRIOR_FV / 1e6:.0f}m", m["fv_debt"] >= MIN_PRIOR_FV),
        ("same loans priced the quarter before (change in mark, PIK switch defined)",
         m["d_price"].notna() & m["pik_switch"].notna()),
        ("non-accrual status known at t and accruing (risk set)", m["na_status"] == 0),
        ("labelled: flagged within t + 1 .. t + 2, or held and accruing through it", m["y_na"].notna()),
    ]
    keep = pd.Series(True, index=m.index)
    out = []
    for name, cond in steps:
        keep &= cond.fillna(False).astype(bool)
        k = m[keep]
        out.append({"step": name, "borrower_quarters": int(keep.sum()), "borrowers": k["borrower"].nunique(),
                    "new_nonaccruals": int((k["na_status"].eq(0) & k["flag_in_window"].astype(bool)).sum())})
    return pd.DataFrame(out)


EXIT_TYPES = ["at scheduled maturity", "before scheduled maturity", "maturity not reported"]


def event_counts(m: pd.DataFrame) -> pd.DataFrame:
    """Per predictor quarter, what happened within t + 1 .. t + 2 to every accruing borrower in the
    sample: entered non-accrual, still accruing through the window, left every BDC's books, and
    status unknown (these four add up to at_risk). Exits are split by type (the three add up to
    exited). events = entered non-accrual; labelled = events + still accruing. Exits and unknown
    status have no label."""
    r = m[m["na_risk_set"].astype(bool) & m["window_complete"].astype(bool)]
    ex = r["exited"].astype(bool)
    cols = {"at_risk": pd.Series(True, index=r.index), "entered_non_accrual": r["y_na"] == 1,
            "still_accruing": r["y_na"] == 0, "exited": ex,
            "status_unknown": r["y_na"].isna() & ~ex,
            **{"exit_" + k.replace(" ", "_"): ex & (r["exit_type"] == k) for k in EXIT_TYPES},
            "events": r["y_na"] == 1, "labelled": r["y_na"].notna()}
    t = pd.DataFrame(cols).groupby([r["cal_q"], r["outcome_q"]]).sum().reset_index()
    t["event_rate"] = t["events"] / t["labelled"]
    total = t.drop(columns=["cal_q", "outcome_q", "event_rate"]).sum()
    total["event_rate"] = total["events"] / total["labelled"]
    return pd.concat([t, pd.DataFrame([{"cal_q": "all", "outcome_q": "", **total}])], ignore_index=True)


def exit_table(m: pd.DataFrame) -> pd.DataFrame:
    """Every exit in event_counts by type, with the borrower's mark at t beside it as context
    only: the mark never types an exit and never labels it. A borrower refinanced by another BDC
    is still held, so it is never an exit; the filings do not separate early repayment,
    refinancing outside the BDCs, a sale and a write-off."""
    r = m[m["na_risk_set"].astype(bool) & m["window_complete"].astype(bool) & m["exited"].astype(bool)]
    band = pd.cut(r["price"], [-np.inf, 0.80, 0.95, np.inf], labels=["below 80", "80-95", "95 and up"], right=False)
    t = pd.crosstab(r["exit_type"], band).reindex(EXIT_TYPES).fillna(0).astype(int)
    t.columns = [f"mark at t {c}" for c in t.columns]
    t.insert(0, "exits", t.sum(axis=1))
    t.insert(1, "share", t["exits"] / t["exits"].sum())
    t.loc["all"] = t.sum()
    return t.rename_axis("exit_type").reset_index()
