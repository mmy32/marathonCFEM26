"""The inputs (signs) of the early-warning model (src/9_prediction), built at the borrower-quarter level
from the tranche-quarter panel: the mark (FV / PAR), its change over the same loans, the rolling max
drawdown, the PIK switch, valuation dispersion and crowding. FEATURES is the one registry of what is
known about each input (unit, cap, direction of risk, event-rate buckets); the other modules read
it. A sign at t uses only what is filed for quarter t (or earlier, for the drawdown)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

import credit_events as CE
from common import norm_name
from config import DD_WINDOW, JUNIOR_LIENS, MARKDOWN, PIK_LOOKBACK

# per input: unit = the step coefficients are reported per (in the input's own units, 0.10 = ten
# points of par); cap = fixed (low, high) bounds applied before fitting (a mark below 70 counts as
# 70, a move of more than 20 points in the quarter or a fall of more than 20 over the year as 20,
# six or more lender families as five: beyond them risk keeps rising far more slowly than the
# logistic's straight line); risk = +1 if a higher value means more risk (equal-weighted ranking);
# zero_if_missing = structurally missing (0 where has_dispersion == 0); buckets = (edges, labels)
# for event rates, in points of par (0.01 = one point), right-closed unless buckets_closed = "left".
# The model reads the mark's move over the quarter as a fall (fall_1q = -d_price, a fall is
# positive), like the fall over the past year; d_price, the signed change, stays in the panel for
# the drawdown and the markdown outcomes.
FEATURES = {
    "price": {"unit": 0.10, "cap": (0.70, None), "risk": -1,
              "buckets": ([-np.inf, 0.80, 0.90, 0.95, 0.98, np.inf],
                          ["below 80", "80-90", "90-95", "95-98", "98 and up"])},
    "price_rel": {"unit": 0.10},
    "fall_1q": {"unit": 0.05, "cap": (-0.20, 0.20), "risk": 1, "buckets_closed": "left",
                "buckets": ([-np.inf, -0.01, 0.01, 0.05, np.inf], ["rose 1+", "within 1", "fell 1-5", "fell 5+"])},
    "fall_1q_adj": {"unit": 0.05},
    "max_drawdown": {"unit": 0.05, "cap": (None, 0.20), "risk": 1,
                     "buckets": ([-np.inf, 0.01, 0.05, 0.10, np.inf], ["under 1", "1-5", "5-10", "over 10"])},
    "pik_switch": {"risk": 1},
    "pik_past_year": {"risk": 1},
    "dispersion": {"unit": 0.01, "risk": 1, "zero_if_missing": True,
                   "buckets": ([-np.inf, 0.005, 0.01, 0.02, 0.05, np.inf],
                               ["under 0.5", "0.5-1", "1-2", "2-5", "over 5"])},
    "dispersion_std": {"unit": 0.01, "zero_if_missing": True},
    "crowding": {"cap": (None, 5), "risk": 1, "buckets": ([-np.inf, 1, 2, 4, np.inf], ["1", "2", "3-4", "5+"])},
    "d_pik_pp": {"unit": 10.0},
    "junior_share": {"unit": 0.5},
}
PREDICTOR_CAPS = {k: v["cap"] for k, v in FEATURES.items() if "cap" in v}
UNITS = {k: v["unit"] for k, v in FEATURES.items() if "unit" in v}
RISK_SIGN = {k: v["risk"] for k, v in FEATURES.items() if "risk" in v}
STRUCTURAL_ZERO = {k for k, v in FEATURES.items() if v.get("zero_if_missing")}
BUCKETS = {k: v["buckets"] for k, v in FEATURES.items() if "buckets" in v}
BUCKETS_LEFT_CLOSED = {k for k, v in FEATURES.items() if v.get("buckets_closed") == "left"}


def _wmean(df: pd.DataFrame, col: str, w: str, keys: list[str]) -> pd.Series:
    d = df[df[col].notna() & (df[w] > 0)]
    return (d[col] * d[w]).groupby([d[k] for k in keys]).sum() / d[w].groupby([d[k] for k in keys]).sum()


def named_tranches(tr: pd.DataFrame) -> pd.DataFrame:
    t = tr[~tr["borrower_key"].str.startswith("ID: ")].copy()
    t["borrower"] = norm_name(t["borrower_key"])
    return t


def no_par_debt(rows: pl.DataFrame) -> pd.DataFrame:
    """Per borrower x quarter: fair value (fv_nopar) and cost (cost_nopar) of funded debt reported
    without par, which the mark takes at FV / cost (the tranche panel holds only rows with par).
    Rows of unknown seniority without par are mostly equity and warrants, so they are left out."""
    r = rows.select(["cal_q", "borrower_canonical", "instrument_seniority", "PAR", "FV", "COST"]).to_pandas()
    r = CE.named_rows(r[~(r["PAR"] > 0) & (r["FV"] > 0) & (r["COST"] > 0) & r["instrument_seniority"].isin(CE.DEBT)])
    return r.groupby(["borrower", "qsort"]).agg(fv_nopar=("FV", "sum"), cost_nopar=("COST", "sum")).reset_index()


def borrower_marks(t: pd.DataFrame) -> pd.DataFrame:
    """Per borrower x quarter over funded loans: par, fv_px and par_px (FV and par of the priceable
    loans; build_features turns them into the mark), junior_share (2L, subordinated and
    unsecured share of priced par), CLO share, PIK switch, dispersion and crowding (n_families:
    manager families lending to it)."""
    keys = ["borrower", "qsort"]
    t = t.assign(_junior=t["PAR_px"].where(t["lien_group"].isin(JUNIOR_LIENS), 0.0),
                 _clo=t["PAR"].where(t["lien_group"] == "CLO", 0.0),
                 _seasoned=t["link_rule"] > 0,
                 _switch=t["pik_switch_clean"].fillna(False).astype(bool))
    b = t.groupby(keys).agg(par=("PAR", "sum"), fv_px=("FV_px", "sum"), par_px=("PAR_px", "sum"),
                            junior_par=("_junior", "sum"), clo_par=("_clo", "sum"),
                            n_families=("family", "nunique"), seasoned=("_seasoned", "max"),
                            pik_switch=("_switch", "max"))
    b["junior_share"] = (b.pop("junior_par") / b["par_px"]).where(b["par_px"] > 0)
    b["clo_share"] = b.pop("clo_par") / b["par"]
    b["log_par"] = np.log(b["par"])
    b["pik_switch"] = b["pik_switch"].astype(float).where(b["seasoned"])   # undefined on a borrower's first quarter
    # valuation gap: highest minus lowest family mark on the same loan (comparable tranches, 2+ families)
    b["dispersion"] = _wmean(t, "V1_range", "PAR", keys)
    b["dispersion_std"] = _wmean(t, "V1_std", "PAR", keys)
    b["has_dispersion"] = b["dispersion"].notna().astype(int)
    b["sector"] = t.sort_values("PAR", ascending=False).drop_duplicates(keys).set_index(keys)["sector"]
    return b.reset_index()


def chain_changes(t: pd.DataFrame) -> pd.DataFrame:
    """Per borrower x quarter t, over the same loans (chains) held at t - 1 and t:
    d_price = par-weighted change in FV / PAR (weights: priced par at t - 1), val_ret = change in
    fair value net of the change in cost over prior fair value plus new cost, d_pik_pp =
    par-weighted change in the PIK share of the coupon (percentage points). Draws, repayments and
    loans entering or leaving do not move them."""
    keys = ["borrower", "qsort"]
    c = t[["chain_id", "qsort", "borrower", "FV", "COST", "FV_px", "PAR_px", "PAR", "pik_share"]].copy()
    c["cprice"] = (c["FV_px"] / c["PAR_px"]).where(c["PAR_px"] > 0)
    c["cprice"] = c["cprice"].where(c["cprice"] <= CE.MAX_MARK)
    prev = c[["chain_id", "qsort", "cprice", "PAR_px", "PAR", "pik_share", "FV", "COST"]].assign(qsort=c["qsort"] + 1)
    m = c.merge(prev, on=["chain_id", "qsort"], suffixes=("", "_prev"))     # consecutive quarters only
    px = m[m["cprice"].notna() & m["cprice_prev"].notna()]
    w = px["PAR_px_prev"]
    out = (w * (px["cprice"] - px["cprice_prev"])).groupby([px[k] for k in keys]).sum() \
        / w.groupby([px[k] for k in keys]).sum()
    out = out.rename("d_price").to_frame()
    # val_ret: change in fair value net of the change in cost (draws, paydowns, capitalised PIK), over
    # prior fair value plus new cost, summed over the same loans and capped at +-100%
    vr = m[m[["FV", "COST", "FV_prev", "COST_prev"]].notna().all(axis=1)]
    dcost = vr["COST"] - vr["COST_prev"]
    num = (vr["FV"] - vr["FV_prev"] - dcost).groupby([vr[k] for k in keys]).sum()
    den = (vr["FV_prev"] + dcost.clip(lower=0)).groupby([vr[k] for k in keys]).sum()
    out = out.join((num / den).where(den > 0).clip(-1, 1).rename("val_ret"), how="outer")
    pk = m[m["pik_share"].notna() & m["pik_share_prev"].notna() & (m["PAR_prev"] > 0)]
    wk = pk["PAR_prev"]
    out = out.join((100 * (wk * (pk["pik_share"] - pk["pik_share_prev"])).groupby([pk[k] for k in keys]).sum()
                    / wk.groupby([pk[k] for k in keys]).sum()).rename("d_pik_pp"), how="outer")
    return out.reset_index()


def rolling_max_drawdown(d: pd.DataFrame, col: str = "d_price", window: int = DD_WINDOW) -> pd.DataFrame:
    """Largest peak-to-trough fall of a borrower's chain-linked mark over quarters t - window .. t,
    in price points (0.05 = five points of par), 0 if it never fell. The index cumulates the
    same-loan changes `col`; a quarter without one adds nothing. The drawdown is missing when the
    window holds no observed change."""
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
    return pd.DataFrame({"borrower": np.repeat(w.index.to_numpy(), w.shape[1]),
                         "qsort": np.tile(w.columns.to_numpy(), w.shape[0]), "dd": dd.ravel()})


def past_pik_switch(b: pd.DataFrame, quarters: int = PIK_LOOKBACK) -> pd.Series:
    """1 if the borrower had a PIK switch in any of the `quarters` quarters before t (not t itself),
    else 0: a quarter it was not held, or its first quarter (switch undefined), counts as no switch."""
    s = b.loc[b["pik_switch"] == 1, ["borrower", "qsort"]]
    hits = pd.concat([s.assign(qsort=s["qsort"] + k) for k in range(1, quarters + 1)]).drop_duplicates()
    flag = b[["borrower", "qsort"]].merge(hits.assign(_hit=1.0), on=["borrower", "qsort"], how="left")["_hit"]
    return pd.Series(flag.fillna(0.0).to_numpy(), index=b.index)


def build_features(tr: pd.DataFrame, rows: pl.DataFrame) -> pd.DataFrame:
    """tr: monitor_panel.build_monitor_panel's tranche-quarter panel; rows: its FINAL_DATASET rows.
    One row per borrower x quarter with the signs at t: price (mark level), fall_1q (its fall over
    the quarter, same loans, a fall is positive; d_price is the signed change), max_drawdown,
    pik_switch (alternative d_pik_pp), pik_past_year (a PIK switch in the three quarters before t),
    dispersion (the valuation gap: highest minus lowest lender
    family's mark; alternative dispersion_std; has_dispersion = 1 when defined), crowding, plus
    market-adjusted versions (price_rel, fall_1q_adj), val_ret and controls."""
    t = named_tranches(tr)
    b = borrower_marks(t)
    # the mark: FV / par on loans with par, FV / cost on debt without par
    b = b.merge(no_par_debt(rows), on=["borrower", "qsort"], how="left").fillna({"fv_nopar": 0.0, "cost_nopar": 0.0})
    price = ((b["fv_px"] + b["fv_nopar"]) / (b["par_px"] + b["cost_nopar"])).where(b["par_px"] + b["cost_nopar"] > 0)
    b["price"] = price.where(price <= CE.MAX_MARK)
    b["priced_share"] = (b["par_px"] + b["cost_nopar"]) / (b["par"] + b["cost_nopar"])
    ch = chain_changes(t)
    b = b.merge(ch, on=["borrower", "qsort"], how="left")

    b["fall_1q"] = -b["d_price"]
    # market-adjusted fall and level: the quarter's median borrower subtracted
    b["fall_1q_adj"] = b["fall_1q"] - b.groupby("qsort")["fall_1q"].transform("median")
    b["price_rel"] = b["price"] - b.groupby("qsort")["price"].transform("median")
    dd = rolling_max_drawdown(b, "d_price").rename(columns={"dd": "max_drawdown"})
    b = b.merge(dd, on=["borrower", "qsort"], how="left")
    b["recent_markdown"] = (b["d_price"] <= -MARKDOWN).astype(float).where(b["d_price"].notna())

    b["crowding"] = b["n_families"].astype(float)
    b["pik_past_year"] = past_pik_switch(b)

    return b
