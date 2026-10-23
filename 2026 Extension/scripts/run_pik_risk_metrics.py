#!/usr/bin/env python3
"""
run_pik_risk_metrics.py

Orchestration script for PIK risk detection, data provenance classification,
loan transition tracking, non-cash metrics measurement, and multi-period
sector sweeps across the private credit panel.

Usage:
    python scripts/run_pik_risk_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT / "src" / "1_data_processing"))  # for paths.py
from paths import FINAL_DIR, PIK_DIR, PIK_PLOTS_DIR, add_code_dirs_to_path
add_code_dirs_to_path()  # modules live in src/*/

import polars as pl

from index_construction import (
    load_and_prepare_investment_data,
    compute_final_interest_rates,
    compute_position_level_flows,
    compute_market_weights,
    qsort_expr,
    safe_div,
)
from subindex_construction import prepare_subindex_input
from tranches import build_tranche_panel
from pik_metrics import (
    classify_pik_provenance,
    aggregate_pik_provenance,
    classify_loan_pik_transitions,
    aggregate_pik_transitions,
    compute_non_cash_metrics,
    aggregate_non_cash_metrics,
    compute_sector_time_series_pik_metrics,
    plot_non_cash_metrics_time_series,
    plot_distress_amendments_vs_markdowns,
    plot_sector_distress_amendments,
    plot_pik_provenance_positions,
    plot_pik_provenance_dollar_volume,
    plot_bdc_idiosyncratic_tagging,
    plot_pik_distress_vs_contractual_share,
    compute_amendment_event_study,
    plot_amendment_event_study,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_pik_risk_metrics")


def main():
    data_csv = FINAL_DIR / "FINAL_DATASET.csv"
    outputs_dir = PIK_DIR
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not data_csv.exists():
        logger.error(f"Input file not found: {data_csv}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 1. Load and prepare data
    # -------------------------------------------------------------------------
    logger.info(f"Loading input dataset: {data_csv}")
    df = load_and_prepare_investment_data(str(data_csv))
    logger.info(f"Loaded {df.height:,} raw rows")

    # Data-quality and sector unification
    logger.info("Applying sub-index data-quality prep (sector label unification & rate ceiling check)...")
    df, clean_report = prepare_subindex_input(df)
    logger.info(f"Sector label prep report: {clean_report}")

    # Compute final rates (PIC_Final, PIK_Final, IR_Final)
    logger.info("Deriving final interest rates (PIC_Final, PIK_Final, IR_Final)...")
    df = compute_final_interest_rates(df)

    # -------------------------------------------------------------------------
    # 2. Disclosed vs. Inferred PIK Classification (Item 0)
    # -------------------------------------------------------------------------
    logger.info("Running Step 0: Classifying PIK data provenance (explicit vs. inferred)...")
    df = classify_pik_provenance(df)
    prov_summary = aggregate_pik_provenance(df)

    q_prov_csv = outputs_dir / "pik_provenance_summary.csv"
    cik_prov_csv = outputs_dir / "pik_provenance_by_cik.csv"

    prov_summary["quarterly"].write_csv(str(q_prov_csv))
    prov_summary["by_cik"].write_csv(str(cik_prov_csv))
    logger.info(f"Saved provenance summary to {q_prov_csv}")
    logger.info(f"Saved CIK provenance summary to {cik_prov_csv}")

    # Print brief summary
    logger.info("Provenance summary by quarter:\n" + str(prov_summary["quarterly"].filter(pl.col("pik_provenance") != "zero_pik")))

    # -------------------------------------------------------------------------
    # 3. Tranche chains, cash flows and market weights
    # -------------------------------------------------------------------------
    # Loans are matched within a borrower on lien, spread and maturity (tranches.py), so a PIK
    # toggle is on the same loan. Only funded debt is kept: unfunded commitments and equity
    # carry no PIK.
    logger.info("Building tranche chains (lien, spread, maturity within borrower)...")
    tr = build_tranche_panel(df)
    tr["tranche_id"] = tr["chain_id"].astype(str)
    logger.info(f"Tranche panel: {len(tr):,} tranche-quarters, {tr['chain_id'].nunique():,} chains; "
                f"link rules {tr['link_rule'].value_counts().sort_index().to_dict()}")
    tr_pl = pl.from_pandas(tr.drop(columns=["maturity"]).assign(cik=tr["cik"].astype(str)))

    logger.info("Computing tranche-level flows, lags, and market weights...")
    flow_df = compute_position_level_flows(tr_pl, qsort_expr, safe_div, group_cols=["cik", "tranche_id"])
    flow_df = compute_market_weights(flow_df, safe_div)
    logger.info(f"Flow-computed panel: {flow_df.height:,} rows")

    # -------------------------------------------------------------------------
    # 4. Contractual vs. Distress Amendment Classification (Item 1)
    # -------------------------------------------------------------------------
    logger.info("Running Step 1: Classifying loan-level transitions (contractual vs. distress amendment)...")
    flow_df = classify_loan_pik_transitions(flow_df)
    trans_summary = aggregate_pik_transitions(flow_df)

    # Loan-level P1-P3 for the signal tests (Risk Dimensions doc, section 2)
    signals = flow_df.select([
        "cik", "family", "cal_q", "borrower_key", "lien", "tranche_id", "xb_id", "xb_tier", "link_rule",
        "sector", "PAR", "FV", "COST", "price", "PIK_Final", "IR_Final", "pik_share", "P1_pct",
        "is_distress_amendment", "pik_transition_state", "pik_at_origination", "origination_censored",
        "pik_acquired", "check_2",
    ])
    signals_csv = outputs_dir / "pik_tranche_signals.csv"
    signals.write_csv(str(signals_csv))
    logger.info(f"Saved tranche-level PIK signals to {signals_csv}")

    trans_csv = outputs_dir / "pik_transitions_summary.csv"
    trans_summary.write_csv(str(trans_csv))
    logger.info(f"Saved transitions summary to {trans_csv}")

    # Print brief summary of distress amendments
    logger.info("Distress amendment volume by quarter:\n" + str(
        trans_summary.select([
            "cal_q", "total_positions", "seasoned_positions",
            "n_amendments", "pct_amendment_count", "amendment_fv", "pct_amendment_fv"
        ])
    ))

    # -------------------------------------------------------------------------
    # 5. Non-Cash Portion of the Loan (Three Distinct Metrics) (Item 2)
    # -------------------------------------------------------------------------
    logger.info("Running Step 2: Computing three distinct non-cash metrics (Rate, Income, Cap Burden)...")
    flow_df = compute_non_cash_metrics(flow_df)
    non_cash_summary = aggregate_non_cash_metrics(flow_df)

    non_cash_csv = outputs_dir / "pik_non_cash_metrics.csv"
    non_cash_summary.write_csv(str(non_cash_csv))
    logger.info(f"Saved non-cash metrics to {non_cash_csv}")

    logger.info("Quarterly non-cash metrics:\n" + str(
        non_cash_summary.select([
            "cal_q", "n_positions",
            "weighted_rate_share", "mean_rate_share",
            "weighted_income_share", "mean_income_share",
            "weighted_cap_expansion", "mean_pos_expansion"
        ])
    ))

    # -------------------------------------------------------------------------
    # 6. Parametric Multi-Period & Sector Sweeps (Item 3)
    # -------------------------------------------------------------------------
    logger.info("Running Step 3: Executing parametric multi-period and sector sweep across all 11 sectors...")
    sector_sweep = compute_sector_time_series_pik_metrics(flow_df)

    sector_csv = outputs_dir / "pik_sector_time_series.csv"
    sector_sweep.write_csv(str(sector_csv))
    logger.info(f"Saved sector sweep results ({sector_sweep.height:,} rows) to {sector_csv}")

    # -------------------------------------------------------------------------
    # 7. Time-Series Visualization Suite
    # -------------------------------------------------------------------------
    plots_dir = PIK_PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating time-series chart figures in {plots_dir}...")

    plot1_path = plots_dir / "pik_non_cash_trends.png"
    plot_non_cash_metrics_time_series(non_cash_summary, save_path=str(plot1_path))
    logger.info(f"Generated {plot1_path}")

    plot2_path = plots_dir / "distress_amendments_vs_markdowns.png"
    plot_distress_amendments_vs_markdowns(trans_summary, flow_df, save_path=str(plot2_path))
    logger.info(f"Generated {plot2_path}")

    plot3_path = plots_dir / "sector_distress_amendments.png"
    plot_sector_distress_amendments(sector_sweep, save_path=str(plot3_path))
    logger.info(f"Generated {plot3_path}")

    plot4_path = plots_dir / "pik_provenance_positions.png"
    plot_pik_provenance_positions(prov_summary["quarterly"], save_path=str(plot4_path))
    logger.info(f"Generated {plot4_path}")

    plot5_path = plots_dir / "pik_provenance_dollar_volume.png"
    plot_pik_provenance_dollar_volume(prov_summary["quarterly"], save_path=str(plot5_path))
    logger.info(f"Generated {plot5_path}")

    plot6_path = plots_dir / "bdc_idiosyncratic_tagging.png"
    plot_bdc_idiosyncratic_tagging(prov_summary["by_cik"], top_n=12, save_path=str(plot6_path))
    logger.info(f"Generated {plot6_path}")

    plot7_path = plots_dir / "pik_distress_vs_contractual_share.png"
    plot_pik_distress_vs_contractual_share(flow_df, save_path=str(plot7_path))
    logger.info(f"Generated {plot7_path}")

    event_study = compute_amendment_event_study(flow_df)
    event_csv = outputs_dir / "pik_amendment_event_study.csv"
    event_study.write_csv(str(event_csv))
    logger.info(f"Saved amendment event study to {event_csv}")

    plot8_path = plots_dir / "amendment_event_study.png"
    plot_amendment_event_study(event_study, save_path=str(plot8_path))
    logger.info(f"Generated {plot8_path}")

    logger.info("PIK risk metrics execution and chart generation complete.")


if __name__ == "__main__":
    main()
