"""Logistic regression (src/9_prediction/models): the unpenalized logit with borrower-clustered standard
errors, the coefficient table, and fit_predict (scaler + logit) used by every out-of-sample test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import UNITS
from frames import design


def _loglik(X1: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    eta = X1 @ beta
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def logit_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray, max_iter: int = 200) -> dict:
    """Unpenalized logit by Newton-Raphson with step halving (plain Newton overshoots with rare
    events). Sandwich covariance clustered by `groups`, with the usual G / (G - 1) factor."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X1.shape[1])
    beta[0] = np.log(y.mean() / (1 - y.mean()))
    ll = _loglik(X1, y, beta)
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-(X1 @ beta)))
        step = np.linalg.solve(X1.T @ (X1 * (p * (1 - p))[:, None]), X1.T @ (y - p))
        t = 1.0
        while _loglik(X1, y, beta + t * step) < ll - 1e-12 and t > 1e-10:
            t /= 2
        beta = beta + t * step
        ll_new = _loglik(X1, y, beta)
        done = np.abs(t * step).max() < 1e-10 or abs(ll_new - ll) < 1e-12
        ll = ll_new
        if done:
            break
    p = 1 / (1 + np.exp(-(X1 @ beta)))
    bread = np.linalg.inv(X1.T @ (X1 * (p * (1 - p))[:, None]))
    S = pd.DataFrame(X1 * (y - p)[:, None]).groupby(groups).sum().to_numpy()
    cov = len(S) / (len(S) - 1) * bread @ (S.T @ S) @ bread
    p0 = y.mean()
    ll0 = len(y) * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    return {"beta": beta, "cov": cov, "fit": 1 - ll / ll0, "fitted": p}


def coefficient_table(d: pd.DataFrame, cols: list[str], label: dict) -> pd.DataFrame:
    """In-sample logit on frame `d`, predictors expressed per UNITS step; standard errors clustered
    by borrower. One row per term."""
    X = design(d, cols) / np.array([UNITS.get(c, 1.0) for c in cols])
    y = d["y"].to_numpy(dtype=float)
    r = logit_cluster(X, y, d["borrower"].to_numpy())
    se = np.sqrt(np.diag(r["cov"]))
    z = r["beta"] / se
    out = pd.DataFrame({"term": ["const"] + cols, "per": [np.nan] + [UNITS.get(c, 1.0) for c in cols],
                        "coef": r["beta"], "se_cluster": se, "z": z, "p_value": 2 * norm.sf(np.abs(z)),
                        "ci_low": r["beta"] - 1.96 * se, "ci_high": r["beta"] + 1.96 * se})
    out["odds_ratio"] = np.exp(out["coef"])
    out["or_ci_low"], out["or_ci_high"] = np.exp(out["ci_low"]), np.exp(out["ci_high"])
    for k, v in {**label, "n_obs": len(d), "n_borrowers": d["borrower"].nunique(), "n_events": int(y.sum()),
                 "pseudo_r2": r["fit"], "auc_in_sample": roc_auc_score(y, r["fitted"])}.items():
        out[k] = v
    return out


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Fit the unpenalized logit on `train` and return the predicted probabilities for `test`."""
    model = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=5000))
    model.fit(design(train, cols), train["y"].to_numpy(dtype=float))
    return model.predict_proba(design(test, cols))[:, 1]
