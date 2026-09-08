from __future__ import annotations

import sys
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parents[0]  # final/
sys.path.append(str(ROOT))

from preprocessor import run_preprocess, PreprocessConfig
from paths import IXBRL_WIDE_ALL_CSV, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    user_agent = "Your Name your.email@domain.com"  # not used here, but keep consistent with other scripts

    mode = "full"  # "test" | "full"

    if mode == "test":
        # test: smaller quarter range + diagnostics
        cfg = PreprocessConfig(
            quarter_from="2024Q1",
            quarter_to="2024Q2",
            only_no_shares=True,
            drop_amounts_only=True,
            write_stats_csv=True,
        )
        run_preprocess(
            in_csv=IXBRL_WIDE_ALL_CSV,
            out_csv=PROCESSED_DIR / "ixbrl_clean_TEST.csv",
            cfg=cfg,
            save_diagnostics=True,
        )
        print("[PREPROCESS TEST] done.")

    elif mode == "full":
        cfg = PreprocessConfig(
            quarter_from="2023Q1",
            quarter_to="2026Q4",  # extended to cover newly downloaded filings through 2026; update as new data arrives
            only_no_shares=True,
            drop_amounts_only=True,
            write_stats_csv=False,
        )
        run_preprocess(
            in_csv=IXBRL_WIDE_ALL_CSV,
            out_csv=PROCESSED_DIR / "ixbrl_clean.csv",
            cfg=cfg,
            save_diagnostics=False,
        )
        print("[PREPROCESS FULL] done.")

    else:
        raise ValueError("mode must be 'test' or 'full'")


if __name__ == "__main__":
    main()
