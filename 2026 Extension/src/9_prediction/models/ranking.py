"""Equal-weighted ranking (src/9_prediction/models): nothing is fitted; the benchmark the fitted models
are compared with."""
from __future__ import annotations

import pandas as pd

from features import RISK_SIGN


def equal_weight_score(d: pd.DataFrame, cols: list[str]) -> pd.Series:
    """The equal-weighted ranking of the same signs, with nothing fitted: each sign is ranked within
    the quarter (as a share, riskiest = 1; ties share their average rank) and the ranks are averaged."""
    signs = [c for c in cols if c in RISK_SIGN]
    X = d[signs].astype(float).fillna(0.0) * [RISK_SIGN[c] for c in signs]
    return X.groupby(d["qsort"]).rank(pct=True).mean(axis=1)
