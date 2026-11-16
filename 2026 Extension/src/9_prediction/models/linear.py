"""Linear regression (src/9_prediction/models) for the continuous markdown outcomes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from features import UNITS
from frames import design


def fit_linear(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit ordinary least squares on `train` and return the predictions for `test` and for `train`,
    and the coefficient of the first input per its UNITS step."""
    mdl = LinearRegression().fit(design(train, cols), train["y"])
    return (mdl.predict(design(test, cols)), mdl.predict(design(train, cols)),
            mdl.coef_[0] * UNITS.get(cols[0], 1.0))
