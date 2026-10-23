#!/usr/bin/env python3
"""
run_risk_monitor.py

Quarterly risk monitor, stage 1 (the CFEM guideline's three layers):

    market layer      benchmark table: panel aggregates next to the figures in
                      "Financing the Economy 2026" (ACC / Houlihan Lokey)
    signal layer      share of each BDC's fair value in signal-tail loans, by quarter
    evaluation layer  which signals predict the rung-1 outcome (peer-relative markdown:
                      bottom decile of the quarter's valuation returns) at 1, 2 and 4 quarters

Usage:
    python scripts/run_risk_monitor.py [--reps 200]

Reads data/final_dataset/FINAL_DATASET.csv only. Writes to outputs/risk_monitor/:
    benchmark_table.csv, market_series.csv, market_layer.png
    outcome_rung1_by_quarter.csv
    signal_tail_by_bdc.csv, signal_layer.png
    evaluation_summary.csv, evaluation_buckets.csv, evaluation_h1.png, evaluation_h2.png,
    evaluation_h4.png, evaluation_summary_sector_outcome.csv (outcome within quarter x sector)
    evaluation_na_summary.csv, evaluation_na_buckets.csv, evaluation_na_h{2,4}.png
                      (rung 3: non-accrual onset, loans accruing at t)
Takes about 15 minutes (the quarter bootstrap is most of it).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "1_data_processing"))
from paths import FINAL_DIR, RISK_MONITOR_DIR, add_code_dirs_to_path  # noqa: E402
add_code_dirs_to_path()

import pandas as pd  # noqa: E402

from monitor_panel import build_monitor_panel  # noqa: E402
from benchmarks import benchmark_table, total_fair_value, quarterly_returns  # noqa: E402
from evaluation import evaluate  # noqa: E402
import charts  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200, help="quarter-bootstrap replications")
    args = ap.parse_args()
    out = RISK_MONITOR_DIR
    out.mkdir(parents=True, exist_ok=True)

    print("Building the loan-quarter panel (tranche chains, outcomes, signals) ...")
    panel, rows = build_monitor_panel(FINAL_DIR / "FINAL_DATASET.csv")
    print(f"  {len(panel):,} tranche-quarters")

    print("Market layer ...")
    bench = benchmark_table(rows, panel)
    bench.to_csv(out / "benchmark_table.csv", index=False)
    fv, ret = total_fair_value(rows), quarterly_returns(rows)
    fv.merge(ret, on="cal_q").to_csv(out / "market_series.csv", index=False)
    charts.market_layer(fv, ret, out / "market_layer.png")
    print(bench[["figure", "reported", "panel", "status"]].to_string(index=False))

    print("Outcome, rung 1 ...")
    rung1 = panel.groupby("cal_q").agg(
        eligible_loans=("val_ret", "count"), events=("mk_event", "sum"),
        decile_cutoff=("val_ret", lambda s: s.quantile(0.10)),
        share_marked_down=("val_ret", lambda s: (s < 0).mean()),
        median_val_ret=("val_ret", "median")).reset_index()
    rung1.to_csv(out / "outcome_rung1_by_quarter.csv", index=False)

    print("Signal layer ...")
    charts.signal_layer(panel, out / "signal_layer.png").to_csv(out / "signal_tail_by_bdc.csv", index=False)

    print(f"Evaluation layer ({args.reps} bootstrap reps) ...")
    summary, buckets = evaluate(panel, "Y_mk", reps=args.reps)
    summary.to_csv(out / "evaluation_summary.csv", index=False)
    buckets.to_csv(out / "evaluation_buckets.csv", index=False)
    for h in (1, 2, 4):
        charts.evaluation_layer(buckets, summary, h, out / f"evaluation_h{h}.png")
    sec, _ = evaluate(panel, "Y_mksec", horizons=(2,), reps=args.reps)
    sec.to_csv(out / "evaluation_summary_sector_outcome.csv", index=False)
    cols = ["signal", "horizon", "n", "events", "event_rate", "auc_signal", "auc_gain", "gain_lo90", "gain_hi90", "oos_gain"]
    print(summary[cols].round(3).to_string(index=False))

    print("Evaluation layer, rung 3 (non-accrual onset) ...")
    na_sum, na_buck = evaluate(panel, "Y_na", reps=args.reps)
    na_sum.to_csv(out / "evaluation_na_summary.csv", index=False)
    na_buck.to_csv(out / "evaluation_na_buckets.csv", index=False)
    for h in (2, 4):
        charts.evaluation_layer(na_buck, na_sum, h, out / f"evaluation_na_h{h}.png", "Non-accrual onset")
    print(na_sum[cols].round(3).to_string(index=False))
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
