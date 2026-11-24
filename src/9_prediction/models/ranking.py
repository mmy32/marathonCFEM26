"""Equal-weighted benchmarks (src/9_prediction/models): nothing is fitted. The equal-weighted ranking is
the benchmark the fitted models are compared with; equal weights on standardized signs is the
stricter version, reported beside it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import PREDICTOR_CAPS, RISK_SIGN


def equal_weight_score(d: pd.DataFrame, cols: list[str]) -> pd.Series:
    """The equal-weighted ranking of the same signs, with nothing fitted: each sign is ranked within
    the quarter (as a share, riskiest = 1; ties share their average rank) and the ranks are averaged."""
    signs = [c for c in cols if c in RISK_SIGN]
    X = d[signs].astype(float).fillna(0.0) * [RISK_SIGN[c] for c in signs]
    return X.groupby(d["qsort"]).rank(pct=True).mean(axis=1)


def equal_z_score(d: pd.DataFrame, cols: list[str], caps: dict = PREDICTOR_CAPS) -> pd.Series:
    """Equal weights on standardized signs ("unit weights"), with nothing fitted: each sign capped as
    in the fit and turned so that higher is riskier, standardized within the quarter (mean 0, SD 1),
    and the z-scores averaged. Unlike the ranks it keeps how far a value lies from the rest, so a
    rare PIK switch gets a large z and a mark of 70 counts far more than one of 98."""
    signs = [c for c in cols if c in RISK_SIGN]
    X = d[signs].astype(float).fillna(0.0)
    for c in caps.keys() & set(signs):
        X[c] = X[c].clip(*caps[c])
    X = X * [RISK_SIGN[c] for c in signs]
    g = X.groupby(d["qsort"])
    Z = (X - g.transform("mean")) / g.transform(lambda s: s.std(ddof=0)).replace(0.0, np.nan)
    return Z.fillna(0.0).mean(axis=1)
