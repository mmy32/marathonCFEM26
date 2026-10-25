#!/usr/bin/env python3
"""
run_dispersion_signals.py

Valuation dispersion signals (Risk Dimensions doc, V1 / V2 / V4) on tranches matched across
BDCs by lien, spread and maturity (src/1_data_processing/tranches.py), one manager family =
one holder, price = fair value / par.

Usage:
    python scripts/run_dispersion_signals.py

Writes to outputs/risk_signals/:
    dispersion_tranches.csv      one row per cross-BDC tranche x quarter held by 2+ families
                                 (all tiers; filter on `tier` for the strict sample)
    tranche_match_coverage.csv   per quarter: funded par, par in 2+ family tranches by tier
    dispersion_family_marks.csv  one row per tranche x quarter x family (tier <= 2): the
                                 family's mark minus the consensus and its within-quarter
                                 percentile (which lender is the outlier)

Code: src/4_val_dispersion/val_dispersion.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "1_data_processing"))
sys.path.insert(0, str(ROOT / "src" / "4_val_dispersion"))

import pandas as pd

from paths import FINAL_DIR, RISK_SIGNALS_DIR
import tranches as T
import val_dispersion as VD


def main():
    print("Building tranche panel from FINAL_DATASET.csv ...")
    tr = T.build_tranche_panel(pd.read_csv(FINAL_DIR / "FINAL_DATASET.csv", low_memory=False), with_origination=False)
    print(f"  {len(tr):,} tranche-quarters, {tr['chain_id'].nunique():,} chains")
    disp = VD.tranche_dispersion(tr, min_families=2, max_tier=3)
    RISK_SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    disp.to_csv(RISK_SIGNALS_DIR / "dispersion_tranches.csv", index=False)
    cov = VD.match_coverage(tr, disp)
    cov.to_csv(RISK_SIGNALS_DIR / "tranche_match_coverage.csv", index=False)
    print(f"  {len(disp):,} tranche-quarters held by 2+ families -> {RISK_SIGNALS_DIR / 'dispersion_tranches.csv'}")
    marks = VD.family_marks_vs_consensus(tr, disp=disp[disp["tier"] <= 2])
    marks.to_csv(RISK_SIGNALS_DIR / "dispersion_family_marks.csv", index=False)
    print(f"  {len(marks):,} family marks vs consensus (tier <= 2) -> {RISK_SIGNALS_DIR / 'dispersion_family_marks.csv'}")
    print(cov[["cal_q", "priced_par", "par_share_tier_le1", "par_share_tier_le2", "par_share_tier_le3"]]
          .assign(priced_par=lambda d: d["priced_par"] / 1e9).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
