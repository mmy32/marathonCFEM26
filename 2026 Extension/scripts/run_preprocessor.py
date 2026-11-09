"""Step 4: data/ixbrl/combined/wide_all.csv -> data/parsed/ixbrl_clean.csv, one row per position.
mode in main(): "full", or "test" (2024Q1-Q2 only, with diagnostics)."""
from __future__ import annotations

import sys
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT / "src" / "1_data_processing"))  # for paths.py
from paths import add_code_dirs_to_path
add_code_dirs_to_path()  # modules live in src/*/

from preprocessor import run_preprocess, PreprocessConfig
from paths import IXBRL_WIDE_ALL_CSV, PARSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    mode = "full"  # "test" | "full"
    if mode not in ("test", "full"):
        raise ValueError("mode must be 'test' or 'full'")
    test = mode == "test"
    cfg = PreprocessConfig(
        quarter_from="2024Q1" if test else "2023Q1",
        quarter_to="2024Q2" if test else None,  # full: no upper bound; a fixed cutoff drops every later quarter
        only_no_shares=True,
        drop_amounts_only=True,
        write_stats_csv=test,
    )
    out_csv = PARSED_DIR / ("ixbrl_clean_TEST.csv" if test else "ixbrl_clean.csv")
    run_preprocess(in_csv=IXBRL_WIDE_ALL_CSV, out_csv=out_csv, cfg=cfg, save_diagnostics=test)
    print(f"[PREPROCESS {mode.upper()}] done.")


if __name__ == "__main__":
    main()
