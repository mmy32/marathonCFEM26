"""Borrower-quarter panel for the early-warning models (BDC Credit Risk Capstone outline,
2026-11-12). One row per borrower and quarter t, every BDC's loans to the borrower pooled.
The warning signs use only what is filed for quarter t; the outcomes are non-accrual entry
within 1, 2 and 4 quarters and the change in value to t + 1. Built from the tranche-quarter
panel (monitor_panel.build_monitor_panel) and the FINAL_DATASET rows. Definitions and sample
rules: docs/14_early_warning_model.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

MAX_PRICE = 1.5             # FV / PAR above this is par left in local currency (cleaning issue): no price
MIN_PRIOR_FV = 1_000_000    # sample: the borrower's debt fair value at t is at least $1m
MIN_PRICED_SHARE = 0.5      # sample: priceable par is at least half of the borrower's funded par at t
MIN_LINK_SHARE = 0.5        # price_change: loans priced again at t + 1 carry at least half of priced par at t
MAX_CLO_SHARE = 0.5         # sample: CLO / structured tranches under half of par (CLO vehicles are not operating borrowers)
DD_WINDOW = 4               # rolling max drawdown over the preceding four quarters
LOW_MARK = 0.90             # PIK-switch hypotheses (outline 4.3): mark below 90% of par ...
MARKDOWN = 0.05             # ... or marked down at least five points in the quarter
JUNIOR_LIENS = {"2L", "SUB", "UNSEC"}
HORIZONS = (1, 2, 4)        # non-accrual entry within h quarters: y_na (h = 1), y_na_h2, y_na_h4


def q_label(qsort) -> list[str]:
    """qsort (year * 4 + quarter) back to 'YYYYQn'."""
    q = np.asarray(qsort, dtype=int)
    return [f"{y}Q{n}" for y, n in zip((q - 1) // 4, (q - 1) % 4 + 1)]


def q_sort(label: str) -> int:
    return int(label[:4]) * 4 + int(label[-1])


def q_next(label: str) -> str:
    return q_label([q_sort(label) + 1])[0]


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


# ------------------------------------------------------------------ predictors from the tranche panel

def _named(tr: pd.DataFrame) -> pd.DataFrame:
    t = tr[~tr["borrower_key"].str.startswith("ID: ")].copy()
    t["borrower"] = _norm(t["borrower_key"])
    return t


def borrower_marks(t: pd.DataFrame) -> pd.DataFrame:
    """Per borrower x quarter over funded loans: fv, par, price (= sum FV / sum PAR over priceable
    loans, missing above MAX_PRICE), priced_share, junior_share (2L, subordinated and unsecured
    share of priced par), CLO share, PIK switch, dispersion and size."""
    keys = ["borrower", "qsort"]
    t = t.assign(_junior=t["PAR_px"].where(t["lien_group"].isin(JUNIOR_LIENS), 0.0),
                 _clo=t["PAR"].where(t["lien_group"] == "CLO", 0.0),
                 _seasoned=t["link_rule"] > 0,
                 _switch=t["is_distress_amendment"].fillna(False).astype(bool))
    b = t.groupby(keys).agg(fv=("FV", "sum"), par=("PAR", "sum"),
                            fv_px=("FV_px", "sum"), par_px=("PAR_px", "sum"), junior_par=("_junior", "sum"),
                            clo_par=("_clo", "sum"), n_bdcs=("cik", "nunique"), n_families=("family", "nunique"),
                            seasoned=("_seasoned", "max"), pik_switch=("_switch", "max"))
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

def add_horizon_outcomes(b: pd.DataFrame, s: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Non-accrual entry within h quarters, for borrowers accruing at t: y_na_h{h} = 1 if the
    borrower is on non-accrual in any of t + 1 .. t + h; 0 if not and its status is known (accruing)
    at t + h; missing otherwise (no longer held, or status unknown, at t + h). Missing too when
    t + h is past the last quarter in the data, so that the last quarters are not labelled only
    when an entry came early. first_entry_k: quarters from t to the first non-accrual quarter
    (1 .. max h). y_na_h1 equals y_na."""
    keys = ["borrower", "qsort"]
    last = s["qsort"].max()
    st = s[keys + ["na_status"]]
    for k in range(1, max(horizons) + 1):
        b = b.merge(st.assign(qsort=st["qsort"] - k).rename(columns={"na_status": f"_st{k}"}), on=keys, how="left")
    first = np.full(len(b), np.nan)
    for k in range(max(horizons), 0, -1):
        first = np.where(b[f"_st{k}"].to_numpy() == 1, k, first)
    b["first_entry_k"] = first
    for h in horizons:
        if h == 1:
            continue
        y = np.where(b["first_entry_k"] <= h, 1.0, np.where(b[f"_st{h}"] == 0, 0.0, np.nan))
        b[f"y_na_h{h}"] = pd.Series(y, index=b.index).where((b["na_status"] == 0) & (b["qsort"] + h <= last))
    return b.drop(columns=[f"_st{k}" for k in range(1, max(horizons) + 1)])


def build_model_panel(tr: pd.DataFrame, rows: pl.DataFrame) -> pd.DataFrame:
    """tr: monitor_panel.build_monitor_panel's tranche-quarter panel; rows: its FINAL_DATASET rows.

    Predictors at t: price (mark level), max_drawdown, pik_switch (alternative d_pik_pp),
    dispersion (alternative dispersion_range; has_dispersion = 1 when defined), plus
    market-adjusted versions (price_rel, max_drawdown_adj) and controls.
    Outcomes at t + 1: y_na (new non-accrual; robustness y_na_first, y_na_same_lender),
    val_ret, price_change (same loans; the outline's borrower-level difference is
    price_change_raw). Sample flags: in_sample, na_risk_set, val_sample."""
    t = _named(tr)
    b = borrower_marks(t)
    ch = chain_changes(t)
    b = b.merge(ch, on=["borrower", "qsort"], how="left")

    # market-adjusted changes and level: the quarter's median borrower subtracted
    b["d_price_adj"] = b["d_price"] - b.groupby("qsort")["d_price"].transform("median")
    b["price_rel"] = b["price"] - b.groupby("qsort")["price"].transform("median")
    dd = rolling_max_drawdown(b, "d_price").rename(columns={"dd": "max_drawdown"})
    dd_adj = rolling_max_drawdown(b, "d_price_adj").rename(columns={"dd": "max_drawdown_adj"})
    b = b.merge(dd, on=["borrower", "qsort"], how="left").merge(
        dd_adj[["borrower", "qsort", "max_drawdown_adj"]], on=["borrower", "qsort"], how="left")
    b["recent_markdown"] = (b["d_price"] <= -MARKDOWN).astype(float).where(b["d_price"].notna())

    # non-accrual status and debt fair value from every debt row
    d = debt_rows(rows)
    ls = lender_status(d)
    s = borrower_status(d, ls)
    b = b.merge(s, on=["borrower", "qsort"], how="left")

    # outcomes at t + 1
    nxt_s = s.assign(qsort=s["qsort"] - 1)[["borrower", "qsort", "na_status", "fv_debt"]].rename(
        columns={"na_status": "na_status_next", "fv_debt": "fv_debt_next"})
    nxt_s["present_next"] = True
    nxt_b = b[["borrower", "qsort", "price", "d_price", "link_par_prev"]].assign(qsort=b["qsort"] - 1).rename(
        columns={"price": "price_next", "d_price": "d_price_next", "link_par_prev": "link_par_next"})
    b = b.merge(nxt_s, on=["borrower", "qsort"], how="left").merge(nxt_b, on=["borrower", "qsort"], how="left")
    b["present_next"] = b["present_next"].fillna(False).astype(bool)

    b["y_na"] = b["na_status_next"].where(b["na_status"] == 0)
    b = add_horizon_outcomes(b, s)
    b["y_na_first"] = b["y_na"].where(b["na_ever"] == 0)
    same = same_lender_entries(ls).assign(_same=1.0)
    b = b.merge(same, on=["borrower", "qsort"], how="left")
    b["y_na_same_lender"] = b["y_na"].where((b["y_na"] == 0) | (b["_same"] == 1))   # other entries dropped
    b = b.drop(columns="_same")

    b["val_ret"] = (b["fv_debt_next"] / b["fv_debt"] - 1).where(b["present_next"] & (b["fv_debt"] > 0))
    link_ok = b["link_par_next"] >= MIN_LINK_SHARE * b["par_px"]
    b["price_change"] = b["d_price_next"].where(link_ok)
    b["price_change_raw"] = b["price_next"] - b["price"]

    # sample: an operating borrower, priced at t, large enough, and seen the quarter before
    # (drawdown and PIK switch defined)
    b["in_sample"] = ((b["clo_share"] < MAX_CLO_SHARE) & b["price"].notna()
                      & (b["priced_share"] >= MIN_PRICED_SHARE) & (b["fv_debt"] >= MIN_PRIOR_FV)
                      & b["max_drawdown"].notna() & b["pik_switch"].notna())
    b["na_risk_set"] = b["in_sample"] & (b["na_status"] == 0)
    b["val_sample"] = b["in_sample"] & (b["na_status"] != 1)

    b["cal_q"] = q_label(b["qsort"])
    b["outcome_q"] = q_label(b["qsort"] + 1)
    first = ["borrower", "cal_q", "outcome_q", "qsort", "sector", "in_sample", "na_risk_set", "val_sample",
             "y_na", "y_na_h2", "y_na_h4", "first_entry_k", "y_na_first", "y_na_same_lender", "val_ret",
             "price_change", "price_change_raw",
             "price", "max_drawdown", "pik_switch", "dispersion", "has_dispersion",
             "d_pik_pp", "dispersion_range", "price_rel", "max_drawdown_adj", "junior_share", "recent_markdown"]
    return b[first + [c for c in b.columns if c not in first]].sort_values(["qsort", "borrower"]).reset_index(drop=True)


def sample_waterfall(m: pd.DataFrame) -> pd.DataFrame:
    """Borrower-quarters (with a next quarter in the data) left after each sample rule, and the
    new non-accrual entries among them."""
    m = m[m["qsort"] < m["qsort"].max()]
    steps = [
        ("named borrower-quarters, funded debt", pd.Series(True, index=m.index)),
        ("not a CLO / structured vehicle", m["clo_share"] < MAX_CLO_SHARE),
        ("price at t (priceable par, FV/PAR <= 1.5)", m["price"].notna()),
        (f"priced share >= {MIN_PRICED_SHARE:.0%} of funded par", m["priced_share"] >= MIN_PRICED_SHARE),
        (f"debt fair value >= ${MIN_PRIOR_FV / 1e6:.0f}m", m["fv_debt"] >= MIN_PRIOR_FV),
        ("held the quarter before (drawdown, PIK switch defined)", m["max_drawdown"].notna() & m["pik_switch"].notna()),
        ("non-accrual status known at t and accruing (risk set)", m["na_status"] == 0),
        ("status known at t + 1 (labelled)", m["y_na"].notna()),
    ]
    keep = pd.Series(True, index=m.index)
    out = []
    for name, cond in steps:
        keep &= cond.fillna(False).astype(bool)
        k = m[keep]
        out.append({"step": name, "borrower_quarters": int(keep.sum()), "borrowers": k["borrower"].nunique(),
                    "new_nonaccruals": int((k["na_status"].eq(0) & k["na_status_next"].eq(1)).sum())})
    return pd.DataFrame(out)
