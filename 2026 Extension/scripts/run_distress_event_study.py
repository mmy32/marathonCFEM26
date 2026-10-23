#!/usr/bin/env python3
"""
run_distress_event_study.py

Crowding, concentration, valuation dispersion and PIK around a persistent 20%+ quarterly
markdown (src/8_risk_monitor/distress_event_study.py; docs/13_distress_event_study.md).

Usage:
    python scripts/run_distress_event_study.py

Reads data/final_dataset/FINAL_DATASET.csv only. Writes to outputs/risk_monitor/distress/:
    borrower_quarter_panel.csv   one row per borrower x quarter with every signal
    distress_events.csv          first 20%+ drop per borrower: quarter, drop, status
                                 (persistent / recovered / exited / too_recent)
    distressed_borrowers_register.csv  the record of distressed borrowers: event, what followed
                                 (price, non-accrual, PIK), lenders at the event, BDC exposure,
                                 data center tag, and the outside check where one was done
                                 (data/reference/distress_event_validation.csv)
    signal_watchlist.csv         borrowers with crowding, concentration, dispersion and a rising
                                 PIK trend all in the top 30% in the same quarter: one row per
                                 borrower (flagged quarters, latest signals, any distress event)
    signal_watchlist_quarters.csv every flagged borrower-quarter with its four percentiles
    case_studies/                for each borrower in data/reference/case_studies.csv:
                                 <name>_quarterly.csv (signal path by quarter) and
                                 <name>_lenders.csv (each BDC: family, type, par, mark, weight
                                 in its book, PIK rate, non-accrual), plus all cases combined
                                 (case_studies_quarterly.csv, case_studies_lenders.csv)
    event_profiles.csv           event-time means (k = -4..+2) for persistent, recovered,
                                 persistent excl. 2026Q1, persistent with 2+ lender families;
                                 matched-control means and n per k
    pre_event_contrast.csv       persistent vs recovered before the event (k = -2..-1)
    distress_profiles.png        the profiles, one small multiple per signal
About 5 minutes.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "1_data_processing"))
from paths import (FINAL_DIR, RISK_MONITOR_DIR, REFERENCE_DIR, DATA_CENTER_DIR, UNIVERSE_DIR,  # noqa: E402
                   add_code_dirs_to_path)
add_code_dirs_to_path()

from monitor_panel import build_monitor_panel  # noqa: E402
import distress_event_study as D  # noqa: E402
import charts  # noqa: E402

warnings.filterwarnings("ignore")


def write_case_studies(b, rows, case_dir):
    """Quarterly path and lender detail for each borrower in data/reference/case_studies.csv."""
    import pandas as pd
    cases = pd.read_csv(REFERENCE_DIR / "case_studies.csv")
    idx = pd.read_csv(UNIVERSE_DIR / "BDC_filings_2001_present.csv", dtype=str, usecols=["CIK", "Company", "Date"])
    bdc_names = idx.assign(cik=idx["CIK"].astype(int)).sort_values("Date").groupby("cik")["Company"].last().to_dict()
    quarterly, lenders = D.case_study_tables(b, rows, cases["borrower"], bdc_names)
    case_dir.mkdir(parents=True, exist_ok=True)
    cases.to_csv(case_dir / "case_studies.csv", index=False)
    quarterly.to_csv(case_dir / "case_studies_quarterly.csv", index=False)
    lenders.to_csv(case_dir / "case_studies_lenders.csv", index=False)
    for name in cases["borrower"]:
        slug = name.strip().lower().replace(" ", "_")
        quarterly[quarterly["borrower"] == name.upper()].to_csv(case_dir / f"{slug}_quarterly.csv", index=False)
        lenders[lenders["borrower"] == name.upper()].to_csv(case_dir / f"{slug}_lenders.csv", index=False)
    print(f"  case studies: {len(cases)} borrowers -> {case_dir}")


def main():
    out = RISK_MONITOR_DIR / "distress"
    out.mkdir(parents=True, exist_ok=True)
    print("Building the loan panel and the borrower-quarter panel ...")
    panel, rows = build_monitor_panel(FINAL_DIR / "FINAL_DATASET.csv")
    b = D.borrower_quarter_panel(panel, rows)
    b.to_csv(out / "borrower_quarter_panel.csv", index=False)
    ev = D.find_events(b)
    ev.to_csv(out / "distress_events.csv", index=False)
    print(f"  {len(b):,} borrower-quarters; {len(ev)} borrowers with a 20%+ drop: {ev['status'].value_counts().to_dict()}")
    reg = D.distressed_register(b, ev, panel,
                                validation_csv=REFERENCE_DIR / "distress_event_validation.csv",
                                data_center_csv=DATA_CENTER_DIR / "data_center_borrowers.csv")
    reg.to_csv(out / "distressed_borrowers_register.csv", index=False)
    print(f"  register: {len(reg)} borrowers -> {out / 'distressed_borrowers_register.csv'}")
    wq, wl = D.signal_watchlist(b, reg)
    wq.to_csv(out / "signal_watchlist_quarters.csv", index=False)
    wl.to_csv(out / "signal_watchlist.csv", index=False)
    print(f"  watchlist: {len(wl)} borrowers with all four signals high in some quarter -> {out / 'signal_watchlist.csv'}")
    write_case_studies(b, rows, out / "case_studies")
    prof = D.profiles_by_group(b, ev)
    prof.to_csv(out / "event_profiles.csv", index=False)
    D.pre_event_contrast(b, ev).to_csv(out / "pre_event_contrast.csv", index=False)
    charts.distress_profiles(prof, out / "distress_profiles.png")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
