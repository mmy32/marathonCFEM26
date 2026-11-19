#!/usr/bin/env python3
"""Look-ahead check for the early-warning scorecard: rebuild the whole model panel from a copy of
FINAL_DATASET.csv cut after quarter T (as if the data ended there), and compare it with the panel
built from all the data. Nothing known at T may change when later quarters are added:

  - every warning sign, sample flag and status at quarters <= T;
  - every outcome whose window had closed by T (signal quarters <= T - 2);
  - the scorecard's scores for quarter T, fitted on the windows closed by T.

Cuts: 2025Q1 (the end of the development quarters) and 2025Q3 (the out-of-sample test quarter).
Writes outputs/model/look_ahead_check.csv (one row per cut and column: rows compared, rows that
differ, largest difference). About 8 minutes. Covers everything from FINAL_DATASET on; the
cleaning before it (borrower names, seniority, sector labels) is labelled once over all quarters
and is not re-run here. See docs/14_early_warning_model.md."""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "1_data_processing"))
from paths import FINAL_DIR, MODEL_DATA_DIR, MODEL_DIR, add_code_dirs_to_path  # noqa: E402
add_code_dirs_to_path()

from common import q_sort  # noqa: E402
from config import HORIZON  # noqa: E402
from evaluation import split_predictions  # noqa: E402
from frames import model_frame  # noqa: E402
from panel import build_model_panel  # noqa: E402
from specs import SCORECARD  # noqa: E402

warnings.filterwarnings("ignore")

CUTS = ["2025Q1", "2025Q3"]
KEYS = ["borrower", "qsort"]
KNOWN_AT_T = ["price", "d_price", "pik_switch", "max_drawdown", "dispersion", "has_dispersion", "crowding",
              "fv_debt", "priced_share", "na_status", "na_ever", "in_sample", "na_risk_set",
              "restr_pik", "restr_spread", "restr_maturity", "restr_equity", "restr_write_down"]
CLOSED_BY_T = ["y_event", "y_na", "y_all_terms", "event_q"]      # signal quarters <= T - HORIZON


def build(final_csv: Path) -> pd.DataFrame:
    from monitor_panel import build_monitor_panel
    tr, rows = build_monitor_panel(final_csv)
    return build_model_panel(tr, rows)[0]


def compare(full: pd.DataFrame, cut: pd.DataFrame, cols: list[str], last_q: int, label: str) -> list[dict]:
    a = full[full["qsort"] <= last_q].set_index(KEYS)
    b = cut[cut["qsort"] <= last_q].set_index(KEYS)
    out = [{"what": label, "column": "(rows)", "rows_full": len(a), "rows_cut": len(b),
            "rows_differ": len(a.index.symmetric_difference(b.index)), "max_abs_diff": np.nan}]
    j = a[cols].join(b[cols], how="inner", rsuffix="_cut")
    for c in cols:
        x = pd.to_numeric(j[c], errors="coerce").astype(float)
        y = pd.to_numeric(j[c + "_cut"], errors="coerce").astype(float)
        both_nan = x.isna() & y.isna()
        diff = (x - y).abs().where(~both_nan, 0.0)
        bad = (diff > 1e-9) | (x.isna() != y.isna())
        out.append({"what": label, "column": c, "rows_full": len(j), "rows_cut": len(j),
                    "rows_differ": int(bad.sum()), "max_abs_diff": float(diff.max()) if len(diff) else 0.0})
    return out


def main():
    full = pd.read_csv(MODEL_DATA_DIR / "model_panel.csv", float_precision="round_trip", low_memory=False)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for T in CUTS:
            t = q_sort(T)
            path = Path(tmp) / f"FINAL_DATASET_to_{T}.csv"
            print(f"Cutting FINAL_DATASET after {T} and rebuilding the panel ...")
            (pl.scan_csv(FINAL_DIR / "FINAL_DATASET.csv", infer_schema_length=0)
             .filter(pl.col("cal_q") <= T).sink_csv(path))
            cut = build(path)
            path.unlink()
            results += [dict(r, cut=T) for r in compare(full, cut, KNOWN_AT_T, t, f"signs and flags, quarters <= {T}")]
            results += [dict(r, cut=T) for r in compare(full, cut, CLOSED_BY_T, t - HORIZON,
                                                        f"outcomes, windows closed by {T}")]
            p_full = split_predictions(model_frame(full, SCORECARD, labelled=False), SCORECARD, t - HORIZON, [t])
            p_cut = split_predictions(model_frame(cut, SCORECARD, labelled=False), SCORECARD, t - HORIZON, [t])
            j = p_full.merge(p_cut, on=KEYS, how="outer", suffixes=("", "_cut"), indicator=True)
            both = j["_merge"] == "both"
            results.append({"cut": T, "what": f"scorecard scores for {T}", "column": "pred", "rows_full": len(p_full),
                            "rows_cut": len(p_cut), "rows_differ": int((~both).sum() + ((j["pred"] - j["pred_cut"]).abs()[both] > 1e-9).sum()),
                            "max_abs_diff": float((j["pred"] - j["pred_cut"]).abs()[both].max())})
    r = pd.DataFrame(results)[["cut", "what", "column", "rows_full", "rows_cut", "rows_differ", "max_abs_diff"]]
    r.to_csv(MODEL_DIR / "look_ahead_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.to_string(index=False))
    bad = r[r["rows_differ"] > 0]
    print("No look-ahead found." if bad.empty else f"{len(bad)} checks differ: see look_ahead_check.csv")


if __name__ == "__main__":
    main()
