"""Event study of financially distressed borrowers: crowding, concentration, valuation
dispersion and PIK around a persistent 20%+ markdown.

Borrower-quarter panel (one row per validated borrower and quarter, all BDCs combined):

    val_ret       borrower valuation return: sum over the borrower's loans held by the same BDC
                  in both quarters of (FV - FV_prev - (COST - COST_prev)), over the sum of FV_prev.
                  Draws, paydowns, PIK capitalisation, new loans and exits do not count.
    price         par-weighted fair value / par of its priced loans
    crowding      C1 lender families, C2 lender HHI, C3 largest family's share, C4 change in
                  families over four quarters (borrower level, funded debt)
    concentration the borrower's weight in its lenders' books: cost of the borrower in a BDC over
                  that BDC's total cost; largest across lenders and cost-weighted mean. Cost, not
                  fair value, so a markdown does not shrink the weight mechanically.
    dispersion    par-weighted mean, over the borrower's loans held by 2+ manager families, of V1
                  (std of family marks), V1 range, V2 (lowest mark minus consensus) and each
                  lender's distance from consensus; plus whether any loan is multi-family
    PIK           par-weighted PIK share of the coupon, share of par paying PIK, share of par with
                  a PIK toggle that quarter, share of par on acquired PIK
    non-accrual   share of the borrower's debt cost flagged non-accrual (known filings only)

Event (first per borrower): val_ret <= -20% with at least MIN_FV_PREV of fair value the quarter
before. Classified by what follows (cumulative valuation return over the next two quarters):

    persistent  recovers less than half of the drop (the First Brands pattern)
    recovered   recovers more than half: a blip (re-mark, one lender's view, technical)
    exited      no BDC holds the borrower's loans the next quarter
    too_recent  the event is in the panel's last quarter

Controls: borrowers that never have a 20% quarterly drop, matched to each event on quarter,
sector and size (quintile of log BDC-held par within the quarter), and weighted so every cell
counts as many times as it has events. Signals are followed in event time k = -4..+2; the panel
is unbalanced, so each k averages the borrowers observed then (n reported).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

DROP = -0.20
MIN_FV_PREV = 1e6
MAX_PRICE = 1.5
RECOVERY_SHARE = 0.5
WINDOW = range(-4, 3)

SIGNALS = {
    "price":               "Price (FV / par)",
    "val_ret_w":           "Valuation return (capped at +-100%)",
    "C1_families":         "C1 lender families",
    "C2_lender_hhi":       "C2 lender HHI",
    "C3_top_share":        "C3 largest family's share",
    "C4_d_families":       "C4 change in families (4q)",
    "max_lender_weight":   "Largest weight in a lender's book",
    "mean_lender_weight":  "Mean weight in lenders' books",
    "V1_std":              "V1 dispersion (std of marks)",
    "V1_range":            "V1 range of marks",
    "V2_low_mark":         "V2 lowest mark - consensus",
    "mark_distance":       "Lender distance from consensus",
    "pik_share":           "PIK share of coupon",
    "pik_par_share":       "Share of par paying PIK",
    "toggle_par_share":    "Share of par with PIK toggle",
    "acquired_par_share":  "Share of par on acquired PIK",
    "na_cost_share":       "Share of cost on non-accrual",
}


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
    p["_den"] = g["FV"].shift(1).where(linked)
    keys = ["borrower", "cal_q"]
    b = p.groupby(keys).agg(qsort=("qsort", "first"), sector=("sector", "first"), par=("PAR", "sum"),
                            fv=("FV", "sum"), cost=("COST", "sum"), _num=("_num", "sum"), _den=("_den", "sum"),
                            fv_px=("FV_px", "sum"), par_px=("PAR_px", "sum"),
                            C1_families=("C1_families", "first"), C2_lender_hhi=("C2_lender_hhi", "first"),
                            C3_top_share=("C3_top_share", "first"), C4_d_families=("C4_d_families", "first"),
                            log_borrower_par=("log_borrower_par", "first"),
                            multi_family=("V1_std", lambda s: s.notna().any()))
    b["val_ret"] = (b["_num"] / b["_den"]).where(b["_den"] > 0)
    b["val_ret_w"] = b["val_ret"].clip(-1, 1)        # for averages: a tiny FV_prev can explode the ratio
    b["fv_prev"] = b["_den"]
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
    b = b.drop(columns=["_num", "_den"]).reset_index()
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
    w = w.reset_index()
    return w.groupby(["borrower", "cal_q"]).apply(lambda s: pd.Series({
        "max_lender_weight": s["weight"].max(),
        "mean_lender_weight": (s["weight"] * s["cost"]).sum() / s["cost"].sum()})).reset_index()


def find_events(b: pd.DataFrame) -> pd.DataFrame:
    """First 20%+ drop per borrower, classified by what follows."""
    b = b.sort_values(["borrower", "qsort"])
    last_q = b["qsort"].max()
    hit = b[(b["val_ret"] <= DROP) & (b["fv_prev"] >= MIN_FV_PREV)]
    ev = hit.groupby("borrower").first().reset_index()[["borrower", "cal_q", "qsort", "val_ret", "fv_prev", "price",
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
    """One row per borrower with a 20%+ quarterly markdown: the event, what followed, who lent,
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
    cols = ["borrower", "sector", "event_quarter", "event_val_ret", "status", "share_regained", "post_return_2q",
            "price_before", "price_at_event", "price_2q_after", "nonaccrual_before", "nonaccrual_at_event",
            "nonaccrual_2q_after", "pik_share_before", "pik_share_2q_after", "C1_families", "multi_family",
            "n_bdcs", "lender_families", "bdc_fv", "bdc_par", "fv_prev"]
    cols += [c for c in ["data_center_entity", "data_center_scope", "external", "external_note", "source"] if c in reg]
    return reg[cols].rename(columns={"fv_prev": "bdc_fv_before", "bdc_fv": "bdc_fv_before_all",
                                     "bdc_par": "bdc_par_before"}).sort_values(["event_quarter", "event_val_ret"])


WATCH_PCT = 0.70


def signal_watchlist(b: pd.DataFrame, reg: pd.DataFrame | None = None, min_pct: float = WATCH_PCT):
    """Borrowers that show crowding, lender-side concentration, valuation dispersion and a rising
    PIK trend at the same time. Among borrowers with 2+ lender families and a dispersion reading,
    each signal is ranked within the quarter:

        crowding       C1 lender families
        concentration  largest weight in a lender's book (cost)
        dispersion     V1, std of family marks
        PIK trend      change in PIK share of the coupon over four quarters (two if four are
                       not available); must be > 0

    A borrower-quarter is flagged when all four are at or above min_pct. Returns
    (quarters: every flagged borrower-quarter with its signals, watchlist: one row per borrower
    with the flagged quarters, the latest quarter's signals and any distress event)."""
    b = b.sort_values(["borrower", "qsort"]).copy()
    g = b.groupby("borrower")
    b["pik_trend"] = (b["pik_share"] - g["pik_share"].shift(4)).fillna(b["pik_share"] - g["pik_share"].shift(2))
    m = b[(b["C1_families"] >= 2) & b["V1_std"].notna()].copy()
    for col, name in [("C1_families", "pct_crowding"), ("max_lender_weight", "pct_concentration"),
                      ("V1_std", "pct_dispersion"), ("pik_trend", "pct_pik_trend")]:
        m[name] = m.groupby("cal_q")[col].rank(pct=True)
    pcts = ["pct_crowding", "pct_concentration", "pct_dispersion", "pct_pik_trend"]
    m["min_pct"], m["mean_pct"] = m[pcts].min(axis=1), m[pcts].mean(axis=1)
    keep = ["borrower", "cal_q", "sector", "price", "C1_families", "C2_lender_hhi", "max_lender_weight",
            "V1_std", "V2_low_mark", "pik_share", "pik_trend", "acquired_par_share", "na_cost_share", "fv"] + pcts + ["min_pct", "mean_pct"]
    flagged = m[(m["min_pct"] >= min_pct) & (m["pik_trend"] > 0)][keep]
    last_q = b["qsort"].max()
    latest = b[b["qsort"] == last_q].set_index("borrower")
    w = flagged.groupby("borrower").agg(flagged_quarters=("cal_q", lambda s: ", ".join(sorted(s))),
                                        n_flagged=("cal_q", "size"), first_flagged=("cal_q", "min"),
                                        last_flagged=("cal_q", "max"), best_mean_pct=("mean_pct", "max"),
                                        sector=("sector", "first")).reset_index()
    for col in ["price", "C1_families", "max_lender_weight", "V1_std", "pik_share", "na_cost_share", "fv"]:
        w[f"{col}_latest"] = w["borrower"].map(latest[col])
    if reg is not None:
        w = w.merge(reg[["borrower", "event_quarter", "event_val_ret", "status", "external"]], on="borrower", how="left")
    return flagged.sort_values(["cal_q", "mean_pct"], ascending=[True, False]), \
        w.sort_values(["n_flagged", "best_mean_pct"], ascending=False)


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


def profiles_by_group(b: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """Event-time profiles for the comparison groups reported in the doc."""
    groups = {
        "persistent": ev[ev["status"] == "persistent"],
        "recovered": ev[ev["status"] == "recovered"],
        "persistent_excl_2026Q1": ev[(ev["status"] == "persistent") & (ev["cal_q"] != "2026Q1")],
        "persistent_multi_family": ev[(ev["status"] == "persistent") & (ev["C1_families"] >= 2)],
    }
    return pd.concat([event_time_profiles(b, e).assign(group=name, n_events=len(e)) for name, e in groups.items()],
                     ignore_index=True)


def pre_event_contrast(b: pd.DataFrame, ev: pd.DataFrame, k_range=(-2, -1)) -> pd.DataFrame:
    """Before the event (mean over k in k_range): persistent vs recovered events, per signal,
    with a rank-sum test. Asks whether the signals tell a lasting decline from a blip early."""
    from scipy.stats import mannwhitneyu
    rows = []
    for status in ("persistent", "recovered"):
        e = ev.loc[ev["status"] == status, ["borrower", "qsort"]].rename(columns={"qsort": "t"})
        x = e.merge(b, on="borrower")
        x = x[(x["qsort"] - x["t"]).between(*k_range)]
        rows.append(x.groupby("borrower")[list(SIGNALS)].mean().assign(status=status))
    d = pd.concat(rows)
    out = []
    for col, label in SIGNALS.items():
        a, c = d.loc[d["status"] == "persistent", col].dropna(), d.loc[d["status"] == "recovered", col].dropna()
        p = mannwhitneyu(a, c).pvalue if len(a) > 4 and len(c) > 4 else np.nan
        out.append(dict(signal=label, persistent_mean=a.mean(), recovered_mean=c.mean(),
                        persistent_n=len(a), recovered_n=len(c), p_value=p))
    return pd.DataFrame(out)


def event_time_profiles(b: pd.DataFrame, ev: pd.DataFrame, signals=SIGNALS) -> pd.DataFrame:
    """Mean of each signal at event time k for events and their matched controls."""
    b = b.copy()
    b["size_q"] = b.groupby("cal_q")["log_borrower_par"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    ever = set(b.loc[(b["val_ret"] <= DROP) & (b["fv_prev"] >= MIN_FV_PREV), "borrower"])
    ctrl_pool = b[~b["borrower"].isin(ever)]
    cell = b.set_index(["borrower", "qsort"])[["sector", "size_q"]]
    ev = ev.join(cell, on=["borrower", "qsort"], rsuffix="_c")
    ev["sector"] = ev["sector"].fillna("Other / Unclassified")
    cells = ev.groupby(["qsort", "sector", "size_q"]).size().rename("w").reset_index()

    treated = ev[["borrower", "qsort"]].rename(columns={"qsort": "t"}).merge(b, on="borrower")
    treated["k"] = treated["qsort"] - treated["t"]
    treated = treated[treated["k"].isin(WINDOW)]

    anchors = ctrl_pool.rename(columns={"qsort": "t"})[["borrower", "t", "sector", "size_q"]]
    anchors["sector"] = anchors["sector"].fillna("Other / Unclassified")
    anchors = anchors.merge(cells.rename(columns={"qsort": "t"}), on=["t", "sector", "size_q"])
    anchors["w"] = anchors["w"] / anchors.groupby(["t", "sector", "size_q"])["borrower"].transform("size")
    ctrl = anchors[["borrower", "t", "w"]].merge(b, on="borrower")
    ctrl["k"] = ctrl["qsort"] - ctrl["t"]
    ctrl = ctrl[ctrl["k"].isin(WINDOW)]

    out = []
    for col, label in signals.items():
        for k in WINDOW:
            tk = treated.loc[(treated["k"] == k) & treated[col].notna(), col]
            ck = ctrl[(ctrl["k"] == k) & ctrl[col].notna()]
            out.append(dict(signal=label, column=col, k=k,
                            event_mean=tk.mean(), event_se=tk.std() / np.sqrt(len(tk)) if len(tk) > 1 else np.nan,
                            event_median=tk.median(), event_n=len(tk),
                            control_mean=np.average(ck[col], weights=ck["w"]) if len(ck) else np.nan,
                            control_n=ck["borrower"].nunique()))
    return pd.DataFrame(out)
