#!/usr/bin/env python3
"""One simple model per input per outcome: logistic for non-accrual within two quarters, linear for
the next-quarter markdown (price change and val_ret). Uses the panel saved by run_model.py
(data/model/model_panel.csv; rebuild it with `python scripts/run_model.py` if it predates the
markdown outcomes). Same time split as the scorecard. Writes outputs/model/simple_models.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "1_data_processing"))
from paths import MODEL_DATA_DIR, MODEL_DIR, add_code_dirs_to_path  # noqa: E402
add_code_dirs_to_path()

from analysis import simple_models  # noqa: E402
from common import q_sort  # noqa: E402
from config import TEST_QS, TRAIN_END  # noqa: E402

if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    m = pd.read_csv(MODEL_DATA_DIR / "model_panel.csv", float_precision="round_trip")
    t = simple_models(m, q_sort(TRAIN_END), [q_sort(q) for q in TEST_QS])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(MODEL_DIR / "simple_models.csv", index=False)
    print(t.round(4).to_string(index=False))
