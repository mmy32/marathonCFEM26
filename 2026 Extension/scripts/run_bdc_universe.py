import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT / "src" / "1_data_processing"))  # for paths.py
from paths import add_code_dirs_to_path
add_code_dirs_to_path()  # modules live in src/*/

from bdc import build_bdc_filings_from_masteridx, build_bdc_intervals_from_filings
from paths import UNIVERSE_DIR


def run_bdc_universe(
    user_agent: str,
    start_year: int = 2001,
    end_year: int = date.today().year,
    out_dir: Path = UNIVERSE_DIR,
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
    user_agent = "Your Name your.email@domain.com"  # the SEC requires a real name and contact email
    run_bdc_universe(user_agent=user_agent)


if __name__ == "__main__":
    main()
