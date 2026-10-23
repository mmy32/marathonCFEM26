"""Market layer: the panel's aggregates next to the figures the industry publishes.

Reference values are the ones quoted in the CFEM guideline note on the Alternative Credit
Council / Houlihan Lokey report "Financing the Economy 2026" (22 October 2026); page and
figure numbers refer to that report. BDC type and non-accrual status come from FINAL_DATASET's
bdc_type and non_accrual columns. Fund flows (Fig 25) are not in the panel: "pending".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from index_construction import (compute_position_level_flows, compute_market_weights,
                                compute_flow_based_market_index, qsort_expr, safe_div)

MARKET_SIZE_USD = 4.1e12     # p.6, Fig 1
REF_BDC_USD = 500e9          # p.39


def total_fair_value(rows: pl.DataFrame) -> pd.DataFrame:
    return (rows.group_by("cal_q").agg(pl.col("FV").sum().alias("fv")).sort("cal_q").to_pandas()
            .assign(fv_bn=lambda d: d["fv"] / 1e9, share_of_market=lambda d: d["fv"] / MARKET_SIZE_USD))


def quarterly_returns(rows: pl.DataFrame) -> pd.DataFrame:
    """Fair-value-weighted total return by quarter (the project's market index, borrower +
    seniority positions; see docs/4_index_construction.md)."""
    flows = compute_market_weights(compute_position_level_flows(rows, qsort_expr, safe_div), safe_div)
    return compute_flow_based_market_index(flows).to_pandas()[["cal_q", "IndexReturn"]]


def sector_weights(rows: pl.DataFrame, quarter: str) -> pd.Series:
    q = rows.filter(pl.col("cal_q") == quarter).to_pandas()
    return (q.groupby(q["sector"].fillna("Other / Unclassified"))["FV"].sum() / q["FV"].sum()).sort_values(ascending=False)


def top_quintile_origination_share(tr: pd.DataFrame, year: str = "2025") -> tuple[float, int]:
    """Share of new lending (cost of newly started tranche chains) done by the top 20% of
    manager families. Chains starting in a BDC's first panel quarter are its existing book,
    not new lending, and are left out."""
    first_q = tr.groupby("cik")["qsort"].transform("min")
    new = tr[(tr["link_rule"] == 0) & (tr["qsort"] > first_q) & tr["cal_q"].str.startswith(year)]
    by_fam = new.groupby("family")["COST"].sum().sort_values(ascending=False)
    k = max(1, int(np.ceil(0.2 * len(by_fam))))
    return float(by_fam.iloc[:k].sum() / by_fam.sum()), len(by_fam)


def large_borrower_share(tr: pd.DataFrame, quarter: str, threshold: float = 1e9) -> tuple[float, float]:
    """Share of borrowers (and of par) with total BDC-held funded debt above threshold."""
    q = tr[(tr["cal_q"] == quarter) & ~tr["borrower_key"].str.startswith("ID: ")]
    b = q.groupby("borrower_key")["PAR"].sum()
    return float((b > threshold).mean()), float(b[b > threshold].sum() / b.sum())


def amendment_share_by_bdc(tr: pd.DataFrame, year: str = "2025", min_loans: int = 20) -> pd.DataFrame:
    """Per BDC: share of its loans with a term change during the year, on links between
    consecutive quarters: PIK toggle (P2), spread change >= 25 bp, or maturity moved > 31 days
    (each where both quarters report the term)."""
    t = tr.sort_values(["chain_id", "qsort"]).copy()
    g = t.groupby("chain_id")
    t["spread_prev"], t["maturity_prev"] = g["spread"].shift(1), g["maturity"].shift(1)
    linked = (t["link_rule"] > 0) & (t["q_gap"] == 1) & t["cal_q"].str.startswith(year)
    t = t[linked]
    t["pik_toggle"] = t["is_distress_amendment"].astype(bool)
    t["spread_chg"] = (t["spread"] - t["spread_prev"]).abs() >= 0.0025
    t["spread_up"] = (t["spread"] - t["spread_prev"]) >= 0.0025
    t["maturity_chg"] = (t["maturity"] - t["maturity_prev"]).dt.days.abs() > 31
    t["any_change"] = t["pik_toggle"] | t["spread_chg"] | t["maturity_chg"]
    # stress-type: drops spread cuts, which are mostly performing-loan repricings
    t["stress_change"] = t["pik_toggle"] | t["spread_up"] | t["maturity_chg"]
    cols = ["pik_toggle", "spread_chg", "spread_up", "maturity_chg", "any_change", "stress_change"]
    per = t.groupby(["cik", "chain_id"])[cols].max()
    out = per.groupby("cik").agg(loans=("any_change", "size"), **{c: (c, "mean") for c in cols})
    return out[out["loans"] >= min_loans]


def _debt_rows(rows: pl.DataFrame, quarter: str) -> pd.DataFrame:
    """Debt rows (par > 0) of one quarter with a known non-accrual status, as pandas."""
    q = rows.filter((pl.col("cal_q") == quarter) & (pl.col("PAR") > 0)).select(
        ["cik", "sector", "bdc_type", "FV", "COST", pl.col("non_accrual").cast(pl.Float64)]).to_pandas()
    return q[q["non_accrual"].notna()]


def nonaccrual_by_bdc(rows: pl.DataFrame, quarter: str) -> pd.DataFrame:
    """Per BDC: non-accrual share of its debt at cost and at fair value (known filings only)."""
    q = _debt_rows(rows, quarter)
    q["na_cost"], q["na_fv"] = q["COST"] * q["non_accrual"], q["FV"] * q["non_accrual"]
    b = q.groupby(["cik", "bdc_type"])[["COST", "FV", "na_cost", "na_fv"]].sum().reset_index()
    b["rate_cost"], b["rate_fv"] = b["na_cost"] / b["COST"], b["na_fv"] / b["FV"]
    return b


def fv_by_type(rows: pl.DataFrame, quarter: str) -> pd.DataFrame:
    q = rows.filter(pl.col("cal_q") == quarter).group_by("bdc_type").agg(
        pl.col("FV").sum().alias("fv"), pl.col("cik").n_unique().alias("bdcs")).to_pandas()
    return q.set_index("bdc_type")


def nonaccrual_by_sector(rows: pl.DataFrame, quarter: str) -> pd.Series:
    q = _debt_rows(rows, quarter)
    s = q.assign(na_cost=q["COST"] * q["non_accrual"]).groupby(q["sector"].fillna("Other / Unclassified"))[["na_cost", "COST"]].sum()
    return (s["na_cost"] / s["COST"]).sort_values(ascending=False)


def benchmark_table(rows: pl.DataFrame, tr: pd.DataFrame, latest: str = "2026Q2") -> pd.DataFrame:
    fv = total_fair_value(rows).set_index("cal_q")
    ret = quarterly_returns(rows).set_index("cal_q")["IndexReturn"]
    r2025 = float(np.prod(1 + ret.loc[[f"2025Q{i}" for i in range(1, 5)]]) - 1)
    sw = sector_weights(rows, latest)
    topq, nfam = top_quintile_origination_share(tr)
    big_b, big_par = large_borrower_share(tr, latest)
    am = amendment_share_by_bdc(tr)
    fbt = fv_by_type(rows, latest)
    nab = nonaccrual_by_bdc(rows, latest)
    nas = nonaccrual_by_sector(rows, latest)

    def row(fig, page, ref, panel, quantity, status="done", note=""):
        return dict(figure=fig, page=page, reported=ref, panel=panel, panel_quantity=quantity, status=status, note=note)

    sector_txt = "; ".join(f"{k} {v:.1%}" for k, v in sw.head(6).items())
    rows_out = [
        row("Fig 1 / p.39 text", "6, 39", "Market $4.1tn; BDCs ~$500bn (~12%)",
            f"${fv.loc[latest, 'fv_bn']:.0f}bn at {latest} ({fv.loc[latest, 'share_of_market']:.1%} of $4.1tn); "
            f"${fv.loc['2025Q4', 'fv_bn']:.0f}bn at 2025Q4",
            "Total BDC fair value by quarter",
            note="Panel covers 2023Q1-2026Q2; FV of all investments incl. equity"),
        row("Fig 18", "27", "BDC sector weights Jul 2026: business services 25%, software 21%, industrials 17%",
            sector_txt, f"Panel sector weights by FV, {latest}",
            note="Taxonomies differ: panel 'Technology' ~ software; business services sits inside panel 'Industrials'"),
        row("Fig 10", "17", "Top 20% of managers deployed 88% of 2025 capital",
            f"{topq:.0%} (top 20% of {nfam} BDC manager families, 2025)",
            "Share of 2025 new lending (cost of new tranche chains) by top-quintile families",
            note="Report covers all private credit managers; panel covers BDC families only"),
        row("Fig 11", "17", "Deals above $1bn ~12% of transactions",
            f"{big_b:.1%} of borrowers ({big_par:.0%} of BDC-held par), {latest}",
            "Share of borrowers with total BDC-held debt above $1bn",
            note="Different unit: report counts transactions, panel counts BDC holdings per borrower"),
        row("Fig 15", "22", "MSCI private credit closed-end index: 9.5% in 2025, 0.9% in Q1 2026; CDLI 9.3% in 2025 (p.23)",
            f"{r2025:.1%} in 2025; {ret.loc['2026Q1']:.1%} in Q1 2026; {ret.loc['2026Q2']:.1%} in Q2 2026",
            "Fair-value-weighted total return (income + valuation), project market index"),
        row("Fig 40", "55", "Share of loans with a significant amendment: mean 10.6% (2025)",
            f"mean across {len(am)} BDCs: {am['stress_change'].mean():.1%} stress-type change "
            f"(PIK toggle {am['pik_toggle'].mean():.1%}, spread up {am['spread_up'].mean():.1%}, "
            f"maturity moved {am['maturity_chg'].mean():.1%}); {am['any_change'].mean():.1%} any term change "
            f"incl. spread cuts",
            "Share of each BDC's loans with a PIK toggle, spread increase or maturity change during 2025",
            note="Closest to the ACC definition is the stress-type share; spread cuts are mostly repricings. "
                 "Maturity only where tagged (47% of rows)"),
        row("Fig 27", "39", "BDC gross assets by type (listed / non-traded / private), 2015-Apr 2026",
            "; ".join(f"{t} ${r.fv / 1e9:.0f}bn ({int(r.bdcs)} BDCs)" for t, r in fbt.iterrows()),
            f"Fair value and count by BDC type, {latest}",
            note="Panel holds fair value of investments, not gross assets; type from EDGAR filing histories"),
        row("Fig 37", "52", "Non-accruals: 79% of flagship funds <= 2.5% of NAV, 17% above 5%, 3% in 2.6-5%",
            f"{(nab['rate_fv'] <= 0.025).mean():.0%} of {len(nab)} BDCs <= 2.5% of portfolio FV, "
            f"{(nab['rate_fv'] > 0.05).mean():.0%} above 5%, {((nab['rate_fv'] > 0.025) & (nab['rate_fv'] <= 0.05)).mean():.0%} in between",
            f"Distribution across BDCs of non-accrual FV / debt FV, {latest}",
            note="Denominator is the BDC's debt fair value, not NAV (NAV is smaller for levered BDCs, so the panel "
                 "share understates the NAV share); BDCs with known non-accrual status only"),
        row("Fig 39", "54", "Non-accruals 4.2% listed vs 1.1% private BDCs, June 2026",
            "; ".join(f"{t} {g['rate_cost'].mean():.1%} (median {g['rate_cost'].median():.1%}, {len(g)} BDCs)"
                      for t, g in nab.groupby("bdc_type")),
            f"Mean across BDCs of non-accrual cost / debt cost, by BDC type, {latest}",
            note="Report's 'private' compares best with panel 'non-traded'; panel 'private' = privately offered BDCs, "
                 "70% of whose FV has known status"),
        row("Fig 38", "53", "Default rate by industry, Q2 2026 (healthcare highest, 2.7% size-weighted)",
            "; ".join(f"{k} {v:.1%}" for k, v in nas.items() if not k.startswith("Other")),
            f"Non-accrual cost / debt cost by sector, {latest}",
            note="Non-accrual, not default; panel sector taxonomy"),
        row("Fig 25", "38", "Non-listed BDC fundraising $12.2bn 1Q25 -> $2.0bn 2Q26; redemptions $5.9bn", "",
            "Net capital raised by non-traded BDCs", "pending",
            "Needs fund-level share issuance and repurchases from the financial statements (not in FINAL_DATASET)"),
    ]
    return pd.DataFrame(rows_out)
