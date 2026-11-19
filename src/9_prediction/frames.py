"""Model frames (src/9_prediction): the rows a model sees, whatever the estimator. model_frame keeps accruing
borrowers in the sample with the predictors (and the outcome) defined; design applies the fixed caps
and fills the structurally missing dispersion."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import OUTCOME
from features import PREDICTOR_CAPS, STRUCTURAL_ZERO


def model_frame(m: pd.DataFrame, cols: list[str], outcome: str = OUTCOME, only_multi: bool = False,
                labelled: bool = True) -> pd.DataFrame:
    """Accruing borrowers in the sample (na_risk_set) with every predictor in `cols` defined
    (dispersion may be structurally missing; design() fills it with 0) and, if `labelled`, the
    outcome observed. only_multi keeps borrowers with dispersion defined. The outcome is renamed y."""
    d = m[m["na_risk_set"].astype(bool)]
    if labelled:
        d = d[d[outcome].notna()]
    if only_multi:
        d = d[d["has_dispersion"] == 1]
    d = d.dropna(subset=[c for c in cols if c not in STRUCTURAL_ZERO])
    keep = ["borrower", "cal_q", "outcome_q", "qsort", outcome] + [c for c in cols if c != outcome]
    if "has_dispersion" not in keep:
        keep.append("has_dispersion")
    return d[keep].rename(columns={outcome: "y"}).reset_index(drop=True)


def design(d: pd.DataFrame, cols: list[str], caps: dict = PREDICTOR_CAPS) -> np.ndarray:
    """Apply the fixed caps (PREDICTOR_CAPS; caps={} for the uncapped robustness check), then fill
    the structurally missing dispersion with 0 (has_dispersion carries the difference). Other
    predictors are used as reported."""
    X = d[cols].astype(float).copy()
    for c in caps.keys() & set(cols):
        X[c] = X[c].clip(*caps[c])
    for c in STRUCTURAL_ZERO & set(cols):
        X[c] = X[c].fillna(0.0)
    return X.to_numpy()
