#!/usr/bin/env python3
"""Case studies (slides 10, 13, 24): the quarterly signal path and lender detail for each borrower
in data/reference/case_studies.csv. Reads FINAL_DATASET.csv; writes case_studies/ to
outputs/risk_monitor/distress/. About 3 minutes. (The stress events, drop cutoffs and
distressed-borrower register were removed on 2026-11-12 with val_ret.)"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "1_data_processing"))
from paths import FINAL_DIR, RISK_MONITOR_DIR, REFERENCE_DIR, UNIVERSE_DIR, add_code_dirs_to_path  # noqa: E402
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
        charts.case_marks(lenders, name, case_dir / f"{slug}_marks.png")
    print(f"  case studies: {len(cases)} borrowers -> {case_dir}")


def main():
    out = RISK_MONITOR_DIR / "distress"
    out.mkdir(parents=True, exist_ok=True)
    print("Building the loan panel and the borrower-quarter panel ...")
    panel, rows = build_monitor_panel(FINAL_DIR / "FINAL_DATASET.csv")
    b = D.borrower_quarter_panel(panel, rows)
    write_case_studies(b, rows, out / "case_studies")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
