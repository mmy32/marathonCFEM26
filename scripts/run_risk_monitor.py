#!/usr/bin/env python3
"""Borrower warning signs (crowding, disagreement on value, PIK) and each BDC's exposure to the
flagged borrowers (slides 11-15). Reads FINAL_DATASET.csv; writes borrower_warning_flags.csv,
warning_tail_summary.csv, the three warning-sign charts and the bdc_exposure files to
outputs/risk_monitor/signals/. A few minutes."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "1_data_processing"))
from paths import FINAL_DIR, RISK_MONITOR_DIR, add_code_dirs_to_path  # noqa: E402
add_code_dirs_to_path()

from monitor_panel import build_monitor_panel  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def write_signal_tails(panel, rows, out):
    """Warning signs per borrower (one chart per sign) and each BDC's exposure to the flagged
    borrowers."""
    import signal_tails as S
    import distress_event_study as D
    out.mkdir(parents=True, exist_ok=True)
    b = D.borrower_quarter_panel(panel, rows)
    flags, thr = S.warning_flags(b)
    flags.to_csv(out / "borrower_warning_flags.csv", index=False)
    tail = S.warning_tail(flags)
    tail.assign(threshold=tail["sign"].map(thr)).to_csv(out / "warning_tail_summary.csv", index=False)
    print(f"  rules: {thr}")
    print(tail.pivot(index="cal_q", columns="sign", values="share").round(3).to_string())
    crowd = S.crowding_distribution(b)
    crowd.to_csv(out / "crowding_distribution.csv", index=False)
    S.plot_crowding_distribution(crowd, out / "warning_crowding.png")
    print(crowd.pivot(index="cal_q", columns="bucket", values="share_par").round(3).to_string())
    disp = S.disagreement_percentiles(b)
    disp.to_csv(out / "disagreement_percentiles.csv", index=False)
    S.plot_disagreement_percentiles(disp, out / "warning_disagreement.png")
    print(disp.round(4).to_string(index=False))
    S.plot_pik_tail(tail, out / "warning_pik.png")
    e = S.bdc_exposure(rows, flags)
    e.to_csv(out / "bdc_exposure.csv", index=False)
    ex = S.plot_exposure(e, out / "bdc_exposure.png")
    S.exposure_by_quarter(e).to_csv(out / "bdc_exposure_by_quarter.csv", index=False)
    print(ex.round(3).to_string(index=False))
    print(f"  signal layer -> {out}")


def main():
    print("Building the loan-quarter panel (tranche chains, outcomes, signals) ...")
    panel, rows = build_monitor_panel(FINAL_DIR / "FINAL_DATASET.csv")
    print(f"  {len(panel):,} tranche-quarters")
    print("Signal layer ...")
    write_signal_tails(panel, rows, RISK_MONITOR_DIR / "signals")
    print("Done.")


if __name__ == "__main__":
    main()
