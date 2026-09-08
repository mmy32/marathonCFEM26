import os
import sys
from pathlib import Path

# Add src/ to sys.path so sibling module imports below resolve
ROOT = Path(__file__).resolve().parents[1]  # 2026 Extension/
sys.path.append(str(ROOT / "src"))

from bdc import build_bdc_filings_from_masteridx, build_bdc_intervals_from_filings
from paths import PROCESSED_DIR


def run_bdc_universe(
    user_agent: str,
    start_year: int = 2001,
    end_year: int = 2026,
    out_dir: Path = PROCESSED_DIR,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    filings = build_bdc_filings_from_masteridx(
        start_year=start_year,
        end_year=end_year,
        user_agent=user_agent,
        out_dir=out_dir,
    )
    print(f"[BDC] filings: {len(filings):,} rows written to {out_dir}")

    intervals = build_bdc_intervals_from_filings(
        filings,
        out_dir=out_dir,
    )
    print(f"[BDC] intervals: {len(intervals):,} rows written to {out_dir}")

    return filings, intervals


def main():
    user_agent = "Cornell ORIE5220 Capstone sl3627@cornell.edu"
    run_bdc_universe(user_agent=user_agent)


if __name__ == "__main__":
    main()
