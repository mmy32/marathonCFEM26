"""Borrower-quarter panel (all BDCs combined: price, crowding, lender concentration, dispersion,
PIK, non-accrual) and the case-study tables. The stress events built on the valuation return
(val_ret) were removed on 2026-11-12; see docs/13_distress_event_study.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

MAX_PRICE = 1.5


def _wmean(df: pd.DataFrame, col: str, w: str, keys) -> pd.Series:
    d = df[df[col].notna() & (df[w] > 0)]
    return (d[col] * d[w]).groupby([d[k] for k in keys]).sum() / d[w].groupby([d[k] for k in keys]).sum()


def borrower_quarter_panel(panel: pd.DataFrame, rows: pl.DataFrame) -> pd.DataFrame:
    """panel: monitor_panel.build_monitor_panel output (tranche-quarters with signals).
    price = FV / PAR over the priceable funded rows (the mark level, cents on the dollar / 100)."""
    p = panel[~panel["borrower_key"].str.startswith("ID: ")].copy()
    p["borrower"] = p["borrower_key"].str.strip().str.upper()
    p = p.sort_values(["chain_id", "qsort"])
    keys = ["borrower", "cal_q"]
    b = p.groupby(keys).agg(qsort=("qsort", "first"), sector=("sector", "first"), par=("PAR", "sum"),
                            fv=("FV", "sum"), cost=("COST", "sum"),
                            fv_px=("FV_px", "sum"), par_px=("PAR_px", "sum"),
                            C1_families=("C1_families", "first"), C2_lender_hhi=("C2_lender_hhi", "first"),
                            C3_top_share=("C3_top_share", "first"), C4_d_families=("C4_d_families", "first"),
                            log_borrower_par=("log_borrower_par", "first"),
                            multi_family=("V1_std", lambda s: s.notna().any()))
    price = (b["fv_px"] / b["par_px"]).where(b["par_px"] > 0)
    b["price"] = price.where(price <= MAX_PRICE)     # above: par left in local currency (cleaning issue)
    for col in ["V1_std", "V1_range", "V2_low_mark", "mark_distance", "pik_share"]:
        b[col] = _wmean(p, col, "PAR", keys)
    par = p.groupby(keys)["PAR"].sum()
    share = lambda mask: p.loc[mask].groupby(keys)["PAR"].sum().reindex(par.index).fillna(0) / par
    b["pik_par_share"] = share(p["pik_share"].fillna(0) > 0)
    b["toggle_par_share"] = share(p["is_distress_amendment"].fillna(False).astype(bool))
    b["acquired_par_share"] = share(p["pik_acquired"].fillna(False).astype(bool))
    known = p[p["non_accrual"].notna()]
    na = known.assign(na_cost=known["COST"] * known["non_accrual"]).groupby(keys)[["na_cost", "COST"]].sum()
    b["na_cost_share"] = na["na_cost"] / na["COST"]
    b = b.reset_index()
    return b.merge(lender_weights(rows), on=keys, how="left")


def lender_weights(rows: pl.DataFrame) -> pd.DataFrame:
    """Per borrower-quarter: its cost weight in each lending BDC's book (all investments), the
    largest across lenders and the cost-weighted mean."""
    r = rows.select(["cik", "cal_q", "borrower_canonical", "COST"]).to_pandas()
    r = r[r["COST"] > 0]
    total = r.groupby(["cik", "cal_q"])["COST"].transform("sum")
    r = r.assign(borrower=r["borrower_canonical"].astype(str).str.strip().str.upper(), _total=total)
    r = r[r["borrower_canonical"].notna()]
    w = r.groupby(["borrower", "cal_q", "cik"]).agg(cost=("COST", "sum"), total=("_total", "first"))
    w["weight"] = w["cost"] / w["total"]
    w = w.reset_index()                       # sorted by borrower, cal_q, cik
    # One slice per borrower-quarter. Each slice is summed with ndarray.sum, the same summation
    # as the per-group Series.sum this replaced (a groupby apply, ~100x slower), so the result
    # is bit-identical; groupby().sum() adds in a different order and differs in the last bit.
    first = np.flatnonzero(~w.duplicated(["borrower", "cal_q"]).to_numpy())
    wc, cost = (w["weight"] * w["cost"]).to_numpy(), w["cost"].to_numpy()
    sums = [(wc[a:b].sum(), cost[a:b].sum()) for a, b in zip(first, np.append(first[1:], len(w)))]
    out = w.loc[first, ["borrower", "cal_q"]].reset_index(drop=True)
    out["max_lender_weight"] = w.groupby(["borrower", "cal_q"], sort=False)["weight"].max().to_numpy()
    out["mean_lender_weight"] = np.array([s[0] for s in sums]) / np.array([s[1] for s in sums])
    return out


CASE_QUARTERLY_COLS = ["borrower", "cal_q", "sector", "price", "C1_families", "C2_lender_hhi",
                       "C3_top_share", "C4_d_families", "max_lender_weight", "mean_lender_weight", "V1_std",
                       "V1_range", "V2_low_mark", "mark_distance", "pik_share", "pik_par_share",
                       "toggle_par_share", "acquired_par_share", "na_cost_share", "fv", "par", "cost"]


def case_study_tables(b: pd.DataFrame, rows: pl.DataFrame, borrowers, bdc_names: dict | None = None):
    """For each case borrower: (quarterly signal path from the borrower-quarter panel,
    lender-level detail from FINAL_DATASET). Lender detail is one row per quarter x BDC: manager
    family, BDC type, par, cost, fair value, that BDC's mark (FV / par), the borrower's weight in
    the BDC's book (cost / the BDC's total cost), par-weighted PIK rate, any non-accrual flag."""
    names = [n.strip().upper() for n in borrowers]
    quarterly = b[b["borrower"].isin(names)].sort_values(["borrower", "qsort"])
    quarterly = quarterly[[c for c in CASE_QUARTERLY_COLS if c in quarterly.columns]]
    r = rows.select(["cik", "cal_q", "borrower_canonical", "family", "bdc_type", "PAR", "COST", "FV", "PIK_Final",
                     pl.col("non_accrual").cast(pl.Float64)]).to_pandas()
    totals = r[r["COST"] > 0].groupby(["cik", "cal_q"])["COST"].sum()
    r["borrower"] = r["borrower_canonical"].astype(str).str.strip().str.upper()
    r = r[r["borrower"].isin(names)].copy()
    r["_pik_par"] = r["PIK_Final"].fillna(0) * r["PAR"].fillna(0)
    lenders = (r.groupby(["borrower", "cal_q", "cik", "family", "bdc_type"], dropna=False)
                .agg(par=("PAR", "sum"), cost=("COST", "sum"), fv=("FV", "sum"), _pik_par=("_pik_par", "sum"),
                     non_accrual=("non_accrual", "max"), positions=("FV", "size"))
                .reset_index())
    lenders["mark"] = (lenders["fv"] / lenders["par"]).where(lenders["par"] > 0)
    lenders["pik_rate"] = (lenders.pop("_pik_par") / lenders["par"]).where(lenders["par"] > 0)
    lenders["weight_in_bdc_book"] = [c / totals.get((k, q), np.nan) for c, k, q in
                                     zip(lenders["cost"], lenders["cik"], lenders["cal_q"])]
    if bdc_names is not None:
        lenders.insert(3, "bdc", lenders["cik"].map(bdc_names))
    return quarterly, lenders.sort_values(["borrower", "cal_q", "weight_in_bdc_book"], ascending=[True, True, False])
