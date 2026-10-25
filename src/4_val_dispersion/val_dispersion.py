"""Valuation dispersion across BDCs (Risk Dimensions doc, section 3; docs/8_valuation_dispersion.md).

Works on the tranche panel from tranches.build_tranche_panel: loans matched across BDCs on lien,
spread and maturity (never identifier text), one manager family = one holder, price = fair value /
par, unfunded commitments left out.

    family_tranche_prices(tr)      one price per quarter x cross-BDC tranche x family
    tranche_dispersion(tr)         V1 / V2 / V4 per tranche and quarter: how much lenders disagree
    family_marks_vs_consensus(tr)  each family's mark minus the tranche's consensus: which lender
                                   is the outlier (the midterm outline's dispersion signal)
    bdc_mark_distance(marks)       the same per BDC-quarter, par-weighted: how far a BDC's book is
                                   marked from its co-lenders
    match_coverage(tr, disp)       how much par sits in tranches held by 2+ families, by tier

family_tranche_prices and tranche_dispersion moved here from tranches.py on 2026-10-24
(unchanged); tranches.py still exposes them, so `T.tranche_dispersion` keeps working.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_PRICE = 1.2              # family prices above 120% of par are data errors (often par in GBP/EUR/AUD, FV in USD)
MIN_SHARED = 5               # bdc_mark_distance: BDC-quarters with fewer shared loans are left blank


def family_tranche_prices(tr: pd.DataFrame, max_price: float = MAX_PRICE, min_families: int = 1) -> pd.DataFrame:
    """One row per quarter x cross-BDC tranche x manager family: the family's FV / par over its
    priceable rows (price_ok), summed over its BDCs (one family = one holder). Prices above
    max_price (data errors) are dropped. tier is the family's worst match tier in that tranche
    (tier_missing = some member could not be placed). lien_group is the tranche's known lien
    group (ANY if none known). min_families > 1 keeps only tranches held by that many
    families (much faster). FV and PAR in the output are the priced amounts."""
    t = tr[(tr["xb_id"] >= 0) & (tr["PAR_px"] > 0)]
    t = t.assign(FV=t["FV_px"], PAR=t["PAR_px"])
    if min_families > 1:
        nfam = t.drop_duplicates(["xb_id", "family"]).groupby("xb_id").size()
        t = t[t["xb_id"].isin(nfam.index[nfam >= min_families])]
    t = t.assign(_miss=t["xb_tier"].isna(), _wild=t["lien_group"] == "ANY",
                 sector=t["sector"] if "sector" in t else None)
    known = t.sort_values("_wild").drop_duplicates("xb_id").set_index("xb_id")["lien_group"]
    fam = (t.groupby(["cal_q", "xb_id", "family"])
             .agg(FV=("FV", "sum"), PAR=("PAR", "sum"), tier=("xb_tier", "max"), tier_missing=("_miss", "max"),
                  n_bdcs=("cik", "nunique"), borrower_key=("borrower_key", "first"), sector=("sector", "first"))
             .reset_index())
    fam["lien_group"] = fam["xb_id"].map(known)
    fam["price"] = fam["FV"] / fam["PAR"]
    return fam[fam["price"] <= max_price]


def tranche_dispersion(tr: pd.DataFrame, min_families: int = 2, max_tier: float = 3,
                       max_price: float = MAX_PRICE) -> pd.DataFrame:
    """Valuation dispersion per cross-BDC tranche and quarter (Risk Dimensions V1, V2, V4).

    Prices come from family_tranche_prices (FV / par, one family = one holder). Keeps
    tranches with at least min_families families whose worst member tier is <= max_tier.

      V4  consensus price: median family price
      V1  dispersion: std (ddof=1) and IQR of family prices; range for reference
      V2  low mark: lowest family price minus the median (<= 0)
    plus within-quarter percentiles V1_pct (std), V2_pct (low = most pessimistic), V4_pct."""
    fam = family_tranche_prices(tr, max_price, min_families=min_families)
    if fam.empty:
        return pd.DataFrame(columns=["cal_q", "xb_id", "borrower_key", "lien_group", "sector", "n_families",
                                     "n_bdcs", "total_par", "tier", "V4_consensus", "V1_std", "p_min", "p_max",
                                     "V1_iqr", "V1_range", "V2_low_mark", "V1_pct", "V2_pct", "V4_pct"])
    g = fam.groupby(["cal_q", "xb_id"])
    out = g.agg(borrower_key=("borrower_key", "first"), lien_group=("lien_group", "first"),
                sector=("sector", "first"), n_families=("family", "nunique"), n_bdcs=("n_bdcs", "sum"),
                total_par=("PAR", "sum"), tier=("tier", "max"), V4_consensus=("price", "median"),
                V1_std=("price", "std"), p_min=("price", "min"), p_max=("price", "max"),
                tier_missing=("tier_missing", "max"))
    out = out.join(g["price"].quantile([0.25, 0.75]).unstack().rename(columns={0.25: "p25", 0.75: "p75"})).reset_index()
    out = out[(out["n_families"] >= min_families) & ~out["tier_missing"] & (out["tier"] <= max_tier)].copy()
    out["V1_iqr"] = out["p75"] - out["p25"]
    out["V1_range"] = out["p_max"] - out["p_min"]
    out["V2_low_mark"] = out["p_min"] - out["V4_consensus"]
    for c, name in [("V1_std", "V1_pct"), ("V2_low_mark", "V2_pct"), ("V4_consensus", "V4_pct")]:
        out[name] = out.groupby("cal_q")[c].rank(pct=True) * 100
    return out.drop(columns=["p25", "p75", "tier_missing"]).sort_values(["cal_q", "V1_std"], ascending=[True, False])


def family_marks_vs_consensus(tr: pd.DataFrame, min_families: int = 2, max_tier: float = 2,
                              disp: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each family's mark against its co-lenders, per quarter x cross-BDC tranche x family.

      mark_vs_consensus  family price minus the tranche's consensus (V4, median family price);
                         < 0: marks below its co-lenders, > 0: above
      mark_distance      |mark_vs_consensus|
      distance_pct       within-quarter percentile of mark_distance (100 = furthest out)

    This identifies which lender is the outlier, not just that lenders disagree, and is reported
    as a within-quarter percentile because marks compress toward par in calm quarters (Trout,
    p.48). Only tranches kept by tranche_dispersion (2+ families, worst tier <= max_tier).
    Pass `disp` to reuse an existing tranche_dispersion result."""
    if disp is None:
        disp = tranche_dispersion(tr, min_families=min_families, max_tier=max_tier)
    fam = family_tranche_prices(tr, min_families=min_families)
    m = fam.merge(disp[["cal_q", "xb_id", "V4_consensus", "n_families", "tier"]].rename(columns={"tier": "xb_tier"}),
                  on=["cal_q", "xb_id"])
    m["mark_vs_consensus"] = m["price"] - m["V4_consensus"]
    m["mark_distance"] = m["mark_vs_consensus"].abs()
    m["distance_pct"] = m.groupby("cal_q")["mark_distance"].rank(pct=True) * 100
    return m


def bdc_mark_distance(rows: pd.DataFrame, min_shared: int = MIN_SHARED) -> pd.DataFrame:
    """Per BDC-quarter, par-weighted over its loans held by 2+ families: mean distance from the
    co-lender consensus (`dispersion`, the signal-layer measure) and the signed mean
    (`mark_vs_consensus`; < 0 = the BDC marks below its co-lenders), with the number of shared
    loans. `rows` needs cik, cal_q, PAR, mark_distance and mark_vs_consensus, one row per BDC
    tranche-quarter (the risk monitor panel). Fewer than min_shared shared loans: blank."""
    d = rows[rows["mark_distance"].notna() & (rows["PAR"] > 0)]
    keys = [d["cik"], d["cal_q"]]
    par = d["PAR"].groupby(keys).sum()
    out = pd.DataFrame({
        "dispersion": (d["mark_distance"] * d["PAR"]).groupby(keys).sum() / par,
        "mark_vs_consensus": (d["mark_vs_consensus"] * d["PAR"]).groupby(keys).sum() / par,
        "n_shared": d.groupby(keys).size(),
    })
    out.loc[out["n_shared"] < min_shared, ["dispersion", "mark_vs_consensus"]] = np.nan
    return out


def match_coverage(tr: pd.DataFrame, disp: pd.DataFrame) -> pd.DataFrame:
    """Funded par per quarter and how much of it sits in tranches with 2+ families, by tier."""
    out = tr.groupby("cal_q").agg(funded_par=("PAR", "sum"), priced_par=("PAR_px", "sum"),
                                  tranches=("xb_id", "nunique")).reset_index()
    for t in (1, 2, 3):
        d = disp[disp["tier"] <= t].groupby("cal_q").agg(**{f"tranches_tier_le{t}": ("xb_id", "size"),
                                                            f"par_tier_le{t}": ("total_par", "sum")})
        out = out.merge(d.reset_index(), on="cal_q", how="left")
        out[f"par_share_tier_le{t}"] = out[f"par_tier_le{t}"] / out["priced_par"]
    return out
