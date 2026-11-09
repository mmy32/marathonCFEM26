"""Borrower-quarter panel (all BDCs combined: valuation return, price, crowding, lender concentration,
dispersion, PIK, non-accrual), stress events (a return at or below the pooled bottom 1%, about -19%,
classified persistent / recovered / exited / too_recent), the distressed-borrower register and the
case-study tables. Definitions in docs/13_distress_event_study.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

DROP_PCT = 0.01
MIN_FV_PREV = 1e6
MAX_PRICE = 1.5
RECOVERY_SHARE = 0.5


def _wmean(df: pd.DataFrame, col: str, w: str, keys) -> pd.Series:
    d = df[df[col].notna() & (df[w] > 0)]
    return (d[col] * d[w]).groupby([d[k] for k in keys]).sum() / d[w].groupby([d[k] for k in keys]).sum()


def borrower_quarter_panel(panel: pd.DataFrame, rows: pl.DataFrame) -> pd.DataFrame:
    """panel: monitor_panel.build_monitor_panel output (tranche-quarters with signals)."""
    p = panel[~panel["borrower_key"].str.startswith("ID: ")].copy()
    p["borrower"] = p["borrower_key"].str.strip().str.upper()
    p = p.sort_values(["chain_id", "qsort"])
    g = p.groupby("chain_id")
    linked = (p["qsort"] - g["qsort"].shift(1) == 1) & (g["FV"].shift(1) > 0)
    p["_num"] = (p["FV"] - g["FV"].shift(1) - (p["COST"] - g["COST"].shift(1))).where(linked)
    p["_fv_prev"] = g["FV"].shift(1).where(linked)
    p["_den"] = (p["_fv_prev"] + (p["COST"] - g["COST"].shift(1)).clip(lower=0)).where(linked)
    keys = ["borrower", "cal_q"]
    b = p.groupby(keys).agg(qsort=("qsort", "first"), sector=("sector", "first"), par=("PAR", "sum"),
                            fv=("FV", "sum"), cost=("COST", "sum"), _num=("_num", "sum"), _den=("_den", "sum"),
                            _fv_prev=("_fv_prev", "sum"),
                            fv_px=("FV_px", "sum"), par_px=("PAR_px", "sum"),
                            C1_families=("C1_families", "first"), C2_lender_hhi=("C2_lender_hhi", "first"),
                            C3_top_share=("C3_top_share", "first"), C4_d_families=("C4_d_families", "first"),
                            log_borrower_par=("log_borrower_par", "first"),
                            multi_family=("V1_std", lambda s: s.notna().any()))
    b["val_ret"] = (b["_num"] / b["_den"]).where(b["_den"] > 0)
    b["val_ret_w"] = b["val_ret"].clip(-1, 1)        # for averages: caps large gains on small loans
    b["fv_prev"] = b["_fv_prev"]
    b["capital_at_risk"] = b["_den"]
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
    b = b.drop(columns=["_num", "_den", "_fv_prev"]).reset_index()
    return flag_drops(b.merge(lender_weights(rows), on=keys, how="left"))


def pooled_cutoff(b: pd.DataFrame, pct: float = DROP_PCT) -> float:
    """The stress cutoff: the pct quantile of val_ret over all eligible borrower-quarters pooled
    (a return and at least MIN_FV_PREV of prior fair value)."""
    ok = b["val_ret"].notna() & (b["fv_prev"] >= MIN_FV_PREV)
    return float(b.loc[ok, "val_ret"].quantile(pct))


def flag_drops(b: pd.DataFrame, pct: float = DROP_PCT) -> pd.DataFrame:
    """drop_pct: the borrower-quarter's percentile rank of val_ret in the pooled distribution
    (all quarters), among borrowers with a return and at least MIN_FV_PREV of prior fair value.
    is_drop: val_ret at or below the pooled pct cutoff and negative."""
    b = b.copy()
    ok = b["val_ret"].notna() & (b["fv_prev"] >= MIN_FV_PREV)
    b["drop_pct"] = b["val_ret"].where(ok).rank(pct=True, method="average")
    b["is_drop"] = ok & (b["val_ret"] <= pooled_cutoff(b, pct)) & (b["val_ret"] < 0)
    return b


def drop_cutoffs(b: pd.DataFrame) -> pd.DataFrame:
    """Per quarter: eligible borrowers, events (drops at or below the pooled cutoff), the pooled
    cutoff, that quarter's own 1% return (reference only), the share of borrowers marked down,
    and how many a fixed -20% rule would have flagged."""
    ok = b["val_ret"].notna() & (b["fv_prev"] >= MIN_FV_PREV)
    e = b[ok]
    out = e.groupby("cal_q").agg(
        eligible_borrowers=("val_ret", "size"), drops=("is_drop", "sum"),
        quarter_1pct_return=("val_ret", lambda s: s.quantile(DROP_PCT)),
        median_return=("val_ret", "median"), share_marked_down=("val_ret", lambda s: (s < 0).mean()),
        fixed_20pct_drops=("val_ret", lambda s: int((s <= -0.20).sum()))).reset_index()
    out.insert(3, "pooled_cutoff", pooled_cutoff(b))
    return out


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
    w = w.reset_index()
    return w.groupby(["borrower", "cal_q"]).apply(lambda s: pd.Series({
        "max_lender_weight": s["weight"].max(),
        "mean_lender_weight": (s["weight"] * s["cost"]).sum() / s["cost"].sum()})).reset_index()


def find_events(b: pd.DataFrame) -> pd.DataFrame:
    """First stress drop (at or below the pooled cutoff) per borrower, classified by what follows."""
    b = b.sort_values(["borrower", "qsort"])
    last_q = b["qsort"].max()
    hit = b[b["is_drop"]]
    ev = hit.groupby("borrower").first().reset_index()[["borrower", "cal_q", "qsort", "val_ret", "drop_pct", "fv_prev", "price",
                                                         "sector", "C1_families", "multi_family", "log_borrower_par"]]
    by_b = {k: g.set_index("qsort") for k, g in b.groupby("borrower")}
    status, recov, post_ret = [], [], []
    for _, e in ev.iterrows():
        g, t = by_b[e["borrower"]], e["qsort"]
        if t == last_q:
            status.append("too_recent"); recov.append(np.nan); post_ret.append(np.nan); continue
        if t + 1 not in g.index:
            status.append("exited"); recov.append(np.nan); post_ret.append(np.nan); continue
        level, best = 1.0, -np.inf
        for k in (1, 2):
            if t + k in g.index and pd.notna(g.loc[t + k, "val_ret"]):
                level *= 1 + g.loc[t + k, "val_ret"]
                best = max(best, level)
            else:
                break
        if best == -np.inf:
            status.append("exited"); recov.append(np.nan); post_ret.append(np.nan); continue
        # share of the drop regained: price index after the event was (1 + r_t); the drop is -r_t
        regained = (1 + e["val_ret"]) * (best - 1) / (-e["val_ret"])
        recov.append(regained)
        post_ret.append(level - 1)
        status.append("persistent" if regained < RECOVERY_SHARE else "recovered")
    return ev.assign(status=status, share_regained=recov, post_return_2q=post_ret)


def distressed_register(b: pd.DataFrame, ev: pd.DataFrame, panel: pd.DataFrame, validation_csv=None,
                        data_center_csv=None) -> pd.DataFrame:
    """One row per borrower with a stress markdown (pooled bottom 1%): the event, what followed, who lent,
    and any outside evidence. Joins the hand-checked verdicts (data/reference/
    distress_event_validation.csv) and the data center tags (data/data_center/
    data_center_borrowers.csv) when given."""
    p = panel[~panel["borrower_key"].str.startswith("ID: ")].assign(
        borrower=lambda d: d["borrower_key"].str.strip().str.upper())
    holders = (p.groupby(["borrower", "qsort"])
               .agg(lender_families=("family", lambda s: "; ".join(sorted(set(s.dropna())))),
                    n_bdcs=("cik", "nunique"), bdc_fv=("FV", "sum"), bdc_par=("PAR", "sum"))
               .reset_index())
    reg = ev.rename(columns={"cal_q": "event_quarter", "val_ret": "event_val_ret", "price": "price_at_event"})
    reg["prev_q"] = reg["qsort"] - 1
    reg = reg.merge(holders.rename(columns={"qsort": "prev_q"}), on=["borrower", "prev_q"], how="left")
    bq = b.set_index(["borrower", "qsort"])
    for col, k, name in [("price", -1, "price_before"), ("price", 2, "price_2q_after"),
                         ("na_cost_share", -1, "nonaccrual_before"), ("na_cost_share", 0, "nonaccrual_at_event"),
                         ("na_cost_share", 2, "nonaccrual_2q_after"), ("pik_share", -1, "pik_share_before"),
                         ("pik_share", 2, "pik_share_2q_after")]:
        reg[name] = [bq[col].get((r.borrower, r.qsort + k), np.nan) for r in reg.itertuples()]
    reg["multi_family"] = reg["C1_families"] >= 2
    if data_center_csv is not None:
        dc = pd.read_csv(data_center_csv)
        dc["borrower"] = dc["borrower_canonical"].astype(str).str.strip().str.upper()
        dc = dc.drop_duplicates("borrower")[["borrower", "entity", "scope"]].rename(
            columns={"entity": "data_center_entity", "scope": "data_center_scope"})
        reg = reg.merge(dc, on="borrower", how="left")
    if validation_csv is not None:
        reg = reg.merge(pd.read_csv(validation_csv), on="borrower", how="left")
        reg["external"] = reg["external"].fillna("not checked")
    cols = ["borrower", "sector", "event_quarter", "event_val_ret", "drop_pct", "status", "share_regained", "post_return_2q",
            "price_before", "price_at_event", "price_2q_after", "nonaccrual_before", "nonaccrual_at_event",
            "nonaccrual_2q_after", "pik_share_before", "pik_share_2q_after", "C1_families", "multi_family",
            "n_bdcs", "lender_families", "bdc_fv", "bdc_par", "fv_prev"]
    cols += [c for c in ["data_center_entity", "data_center_scope", "external", "external_note", "source"] if c in reg]
    return reg[cols].rename(columns={"fv_prev": "bdc_fv_before", "bdc_fv": "bdc_fv_before_all",
                                     "bdc_par": "bdc_par_before"}).sort_values(["event_quarter", "event_val_ret"])


CASE_QUARTERLY_COLS = ["borrower", "cal_q", "sector", "price", "val_ret", "C1_families", "C2_lender_hhi",
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
