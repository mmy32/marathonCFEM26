"""Loan-quarter panel for the risk monitor: outcomes and signals on tranche chains.

One row per BDC tranche-quarter (tranches.build_tranche_panel). Adds:

Outcome, rung 1 of the guideline's stress ladder (peer-relative markdown):
    val_ret        valuation return since the chain's previous quarter,
                   (FV - FV_prev - (COST - COST_prev)) / FV_prev. Draws, paydowns and PIK
                   capitalisation move FV and cost together and cancel; only re-marks remain.
                   Defined when the previous quarter is the one before (q_gap == 1).
    mk_event       val_ret < 0 and in the bottom decile of the quarter's cross-section
    mk_event_sec   the same within quarter x sector (sectors with >= 200 loans that quarter)
    Y_mk_h1..h4    any mk_event on the same chain in quarters t+1..t+h (NaN if the chain has
                   no eligible return in the window, or the window runs past the panel)

Outcome, rung 3 (non-accrual onset, from FINAL_DATASET's non_accrual column):
    na_onset       flagged now, accruing in the chain's previous quarter (both known)
    Y_na_h1..h4    an onset in t+1..t+h; defined only for loans accruing at t

Signals at t (raw values; the evaluation converts them to within-quarter percentiles):
    M1 price, M2 price change over two quarters              (baseline)
    P1 pik_share, P1_acq (pik_share on acquired PIK, else 0), P2_recent (toggle at t or t-1),
    P3 pik_at_origination                                    (PIK)
    V1_std, V2_low_mark, V4_consensus, mark_vs_consensus,
    mark_distance (|mark_vs_consensus|)                      (dispersion, 2+ families only)
    C1 n_families, C2 lender_hhi, C4 d_families_4q           (crowding, borrower level)
Controls: log_borrower_par, lien_group, sector, age_q (quarters since the chain started).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_data_processing"))
from paths import add_code_dirs_to_path  # noqa: E402
add_code_dirs_to_path()

import tranches as T  # noqa: E402
from manager_family import add_family  # noqa: E402
from index_construction import load_and_prepare_investment_data, compute_final_interest_rates  # noqa: E402
from subindex_construction import prepare_subindex_input  # noqa: E402
from pik_metrics import classify_loan_pik_transitions  # noqa: E402
from concentration_crowding import compute_borrower_crowding_metrics  # noqa: E402

DECILE = 0.10
MIN_SECTOR_LOANS = 200
HORIZONS = (1, 2, 3, 4)


def load_rows(final_csv) -> pl.DataFrame:
    """FINAL_DATASET rows with final rates, merged sector labels and manager family."""
    df = load_and_prepare_investment_data(str(final_csv))
    df, _ = prepare_subindex_input(df)
    return add_family(compute_final_interest_rates(df))


def _add_returns_and_events(tr: pd.DataFrame) -> pd.DataFrame:
    tr = tr.sort_values(["chain_id", "qsort"]).reset_index(drop=True)
    g = tr.groupby("chain_id")
    for c in ["FV", "COST", "price", "qsort"]:
        tr[c + "_prev"] = g[c].shift(1)
    tr["price_prev2"] = g["price"].shift(2)
    tr["qsort_prev2"] = g["qsort"].shift(2)
    ok = (tr["qsort"] - tr["qsort_prev"] == 1) & (tr["FV_prev"] > 0)
    tr["val_ret"] = ((tr["FV"] - tr["FV_prev"] - (tr["COST"] - tr["COST_prev"])) / tr["FV_prev"]).where(ok)
    tr["M2_price_chg_2q"] = (tr["price"] - tr["price_prev2"]).where(tr["qsort"] - tr["qsort_prev2"] == 2)
    tr["age_q"] = tr["qsort"] - g["qsort"].transform("min")

    def bottom(groups):
        cut = tr.groupby(groups)["val_ret"].transform(lambda s: s.quantile(DECILE))
        return (tr["val_ret"] <= cut) & (tr["val_ret"] < 0)
    tr["mk_event"] = bottom(["cal_q"]).where(tr["val_ret"].notna())
    big = tr.groupby(["cal_q", "sector"])["val_ret"].transform("count") >= MIN_SECTOR_LOANS
    tr["mk_event_sec"] = bottom(["cal_q", "sector"]).where(tr["val_ret"].notna() & big)
    return tr.drop(columns=["FV_prev", "COST_prev", "price_prev", "qsort_prev", "price_prev2", "qsort_prev2"])


def _add_forward_outcomes(tr: pd.DataFrame, event_col: str = "mk_event", prefix: str = "Y_mk") -> pd.DataFrame:
    """Y_h = any event on the chain in t+1..t+h, among quarters with an eligible return."""
    last_q = tr["qsort"].max()
    ev = tr[["chain_id", "qsort", event_col]].dropna()
    ahead = []
    for k in range(1, max(HORIZONS) + 1):
        a = ev.assign(qsort=ev["qsort"] - k).rename(columns={event_col: f"_e{k}"})
        ahead.append(a.set_index(["chain_id", "qsort"]))
    joined = tr[["chain_id", "qsort"]].join(pd.concat(ahead, axis=1), on=["chain_id", "qsort"])
    for h in HORIZONS:
        cols = [f"_e{k}" for k in range(1, h + 1)]
        block = joined[cols].astype(float)
        y = block.max(axis=1, skipna=True)
        tr[f"{prefix}_h{h}"] = y.where(block.notna().any(axis=1) & (tr["qsort"] + h <= last_q))
    return tr


def _add_nonaccrual_outcome(tr: pd.DataFrame) -> pd.DataFrame:
    """Rung 3: non-accrual onset. na_onset = flagged now, accruing (0) in the chain's previous
    quarter, both known. Y_na_h = an onset in t+1..t+h, defined only for loans accruing at t
    (the loans at risk) and when the window has at least one known transition."""
    tr = tr.sort_values(["chain_id", "qsort"])
    prev = tr.groupby("chain_id")["non_accrual"].shift(1)
    known = tr["non_accrual"].notna() & prev.notna()
    tr["na_onset"] = ((tr["non_accrual"] == 1) & (prev == 0)).astype(float).where(known)
    tr = _add_forward_outcomes(tr, "na_onset", "Y_na")
    at_risk = tr["non_accrual"] == 0
    for h in HORIZONS:
        tr[f"Y_na_h{h}"] = tr[f"Y_na_h{h}"].where(at_risk)
    return tr


def _add_pik(tr: pd.DataFrame) -> pd.DataFrame:
    p = classify_loan_pik_transitions(pl.from_pandas(
        tr[["cik", "chain_id", "cal_q", "PIK_Final", "IR_Final", "PAR", "pik_at_origination"]]
        .assign(tranche_id=tr["chain_id"].astype(str), cik=tr["cik"].astype(str))
    )).to_pandas()
    p = p[["chain_id", "cal_q", "pik_share", "is_distress_amendment", "pik_acquired"]]
    tr = tr.merge(p, on=["chain_id", "cal_q"], how="left")
    tr = tr.sort_values(["chain_id", "qsort"])
    prev_toggle = tr.groupby("chain_id")["is_distress_amendment"].shift(1).fillna(False).astype(bool)
    tr["P2_recent"] = (tr["is_distress_amendment"].astype(bool) | prev_toggle).astype(float)
    tr["P1_acq"] = tr["pik_share"].where(tr["pik_acquired"].astype(bool), 0.0)
    tr["P3_orig_pik"] = tr["pik_at_origination"].astype(float)
    return tr


def _add_dispersion(tr: pd.DataFrame) -> pd.DataFrame:
    disp = T.tranche_dispersion(tr, min_families=2, max_tier=2)
    fam = T.family_tranche_prices(tr, min_families=2)
    fam = fam.merge(disp[["cal_q", "xb_id", "V4_consensus"]], on=["cal_q", "xb_id"])
    fam["mark_vs_consensus"] = fam["price"] - fam["V4_consensus"]
    tr = tr.merge(disp[["cal_q", "xb_id", "V1_std", "V1_range", "V2_low_mark", "V4_consensus", "n_families", "tier"]]
                  .rename(columns={"n_families": "xb_families", "tier": "xb_match_tier"}),
                  on=["cal_q", "xb_id"], how="left")
    tr = tr.merge(fam[["cal_q", "xb_id", "family", "mark_vs_consensus"]], on=["cal_q", "xb_id", "family"], how="left")
    tr["mark_distance"] = tr["mark_vs_consensus"].abs()
    return tr


def _add_crowding(tr: pd.DataFrame, rows: pl.DataFrame) -> pd.DataFrame:
    c = compute_borrower_crowding_metrics(rows).to_pandas()
    c = c[["cal_q", "borrower", "n_families", "lender_hhi", "top_share", "d_families_4q", "log_par"]].rename(columns={
        "n_families": "C1_families", "lender_hhi": "C2_lender_hhi", "top_share": "C3_top_share",
        "d_families_4q": "C4_d_families", "log_par": "log_borrower_par"})
    tr["_b"] = tr["borrower_key"].str.strip().str.upper()
    return tr.merge(c, left_on=["cal_q", "_b"], right_on=["cal_q", "borrower"], how="left").drop(columns=["_b", "borrower"])


def build_monitor_panel(final_csv) -> tuple[pd.DataFrame, pl.DataFrame]:
    """Returns (tranche-quarter panel with outcomes and signals, the row-level frame)."""
    rows = load_rows(final_csv)
    tr = T.build_tranche_panel(rows)
    tr = _add_returns_and_events(tr)
    tr = _add_forward_outcomes(tr)
    tr = _add_forward_outcomes(tr, "mk_event_sec", "Y_mksec")
    tr = _add_nonaccrual_outcome(tr)
    tr = _add_pik(tr)
    tr = _add_dispersion(tr)
    tr = _add_crowding(tr, rows)
    return tr, rows
