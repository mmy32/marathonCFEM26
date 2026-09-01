import os
import sys
from pathlib import Path

# Add final/ to sys.path so "import src.*" works when running from scripts/
ROOT = Path(__file__).resolve().parents[1]  # final/
sys.path.append(str(ROOT))

from bdc import build_bdc_filings_from_masteridx, build_bdc_intervals_from_filings
from paths import PROCESSED_DIR


def run_bdc_universe(
    user_agent: str,
    start_year: int = 2001,
    end_year: int = 2025,
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
    user_agent = "Your Name your.email@domain.com"
    run_bdc_universe(user_agent=user_agent)


if __name__ == "__main__":
    main()
