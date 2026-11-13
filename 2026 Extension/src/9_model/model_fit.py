"""Estimation and evaluation for the early-warning models (src/9_model): model frames, the
logistic and linear baselines with borrower-clustered standard errors, the non-accrual
extensions (ridge / lasso logistic, gradient-boosted trees, a small MLP), expanding-window
validation for hyperparameters and the threshold, the rolling evaluation and the metrics."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, mean_absolute_error, mean_squared_error, \
    r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WINSOR = (0.01, 0.99)          # continuous outcomes (val_ret, price_change) clipped at the training 1st / 99th percentiles
# Fixed caps on the two continuous signs: a mark below 70 counts as 70, a drawdown above 20 points as 20.
# Beyond them risk keeps rising, but far more slowly than the logistic's straight line (entry within a
# year: 35% at marks of 70-80, 52% below 50); uncapped, the deepest marks got 95-99%. Round fixed caps
# replace the earlier 1st / 99th percentile caps, which gave the same results but are harder to explain.
PREDICTOR_CAPS = {"price": (0.70, None), "max_drawdown": (None, 0.20)}
TARGET_RECALL = 0.80           # threshold: flag enough borrowers to catch 80% of entries in validation
N_VAL_QUARTERS = 4             # expanding-window validation on the last four training quarters
MLP_HIDDEN = 4                 # one hidden layer of four units
MLP_MIN_EVENTS_PER_PARAM = 10  # run the MLP for non-accrual only with at least 10 events per weight
N_BOOT = 1000
SEED = 0

OUTCOMES = {
    "y_na": {"kind": "binary", "sample": "na_risk_set", "label": "new non-accrual at t+1"},
    "price_change": {"kind": "continuous", "sample": "val_sample", "label": "change in FV/PAR, same loans, t to t+1"},
    "val_ret": {"kind": "continuous", "sample": "val_sample", "label": "fair-value return, t to t+1"},
}
# Predictors (outline section 4). Dispersion exists only for borrowers with a tranche held by 2+
# manager families; on the full sample it is 0 elsewhere and comes with its indicator.
PREDICTORS = {"price": ["price"], "max_drawdown": ["max_drawdown"], "pik_switch": ["pik_switch"],
              "dispersion": ["dispersion", "has_dispersion"]}
COMBINED = ["price", "max_drawdown", "pik_switch", "dispersion", "has_dispersion"]
# The headline model: the three signs that explain themselves. Dispersion is left out: it adds
# nothing out of sample, and it needs its indicator (and a reason for the indicator's sign).
HEADLINE = ["price", "max_drawdown", "pik_switch"]
STRUCTURAL_ZERO = {"dispersion", "dispersion_range"}       # 0 where has_dispersion == 0
BINARY = {"pik_switch", "has_dispersion", "recent_markdown"}
# Coefficients are reported per these steps, in the predictor's own units (0.10 = ten points of par)
UNITS = {"price": 0.10, "price_rel": 0.10, "max_drawdown": 0.05, "max_drawdown_adj": 0.05,
         "dispersion": 0.01, "dispersion_range": 0.01, "d_pik_pp": 10.0, "junior_share": 0.5}

GRIDS = {   # the valuation outcomes use OLS only: their rolling test showed nothing to gain from more
    "binary": {
        "logit": [{}],
        "ridge": [{"C": c} for c in (0.001, 0.01, 0.1, 1.0)],
        "lasso": [{"C": c} for c in (0.001, 0.01, 0.1, 1.0)],
        "gbt": [{"max_depth": d, "learning_rate": lr} for d in (2, 3) for lr in (0.03, 0.1)],
        "mlp": [{"alpha": a} for a in (0.001, 0.01, 0.1, 1.0)],
    },
    "continuous": {"ols": [{}]},
}
BASELINE = {"binary": "logit", "continuous": "ols"}

# Main evaluation (rolling): non-accrual entry within h quarters, review lists, fixed hyperparameters
NA_HORIZONS = {1: "y_na", 2: "y_na_h2", 4: "y_na_h4"}
ROLL_MIN_TRAIN_QUARTERS = 3        # a test quarter needs at least three labelled training quarters
REVIEW_SHARES = (0.01, 0.02, 0.05, 0.10, 0.20)
ROLL_PARAMS = {"gbt": {"max_depth": 2, "learning_rate": 0.1}}   # chosen in the holdout validation
N_BOOT_ROLL = 200


# ------------------------------------------------------------------ frames and preprocessing

def model_frame(m: pd.DataFrame, outcome: str, cols: list[str], sample: str | None = None,
                only_multi: bool = False, labelled: bool = True) -> pd.DataFrame:
    """Rows of the outcome's sample with every predictor in `cols` defined (dispersion may be
    structurally missing; it is filled with 0 in design()) and, if `labelled`, the outcome
    observed. only_multi keeps borrowers with dispersion defined. labelled=False also keeps the
    rows whose outcome quarter is not in the data yet (for scoring the latest quarter)."""
    sample = sample or OUTCOMES[outcome]["sample"]
    observed = m[outcome].notna()
    if not labelled:
        observed |= m["qsort"] == m["qsort"].max()
    d = m[m[sample].astype(bool) & observed]
    if only_multi:
        d = d[d["has_dispersion"] == 1]
    need = [c for c in cols if c not in STRUCTURAL_ZERO]
    d = d.dropna(subset=need)
    keep = ["borrower", "cal_q", "outcome_q", "qsort", outcome] + [c for c in cols if c != outcome]
    if "has_dispersion" not in keep:
        keep.append("has_dispersion")
    return d[keep].rename(columns={outcome: "y"}).reset_index(drop=True)


def design(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Apply the fixed caps (PREDICTOR_CAPS), then fill the structurally missing dispersion with 0
    (has_dispersion carries the difference). Other predictors are used as reported."""
    X = d[cols].astype(float).copy()
    for c in PREDICTOR_CAPS.keys() & set(cols):
        X[c] = X[c].clip(*PREDICTOR_CAPS[c])
    for c in STRUCTURAL_ZERO & set(cols):
        X[c] = X[c].fillna(0.0)
    return X.to_numpy()


def outcome_bounds(train: pd.DataFrame, kind: str) -> tuple | None:
    """Training 1st / 99th percentiles of a continuous outcome (None for non-accrual)."""
    return tuple(train["y"].quantile(list(WINSOR))) if kind == "continuous" else None


def outcome_vector(d: pd.DataFrame, ybounds: tuple | None) -> np.ndarray:
    y = d["y"].to_numpy(dtype=float)
    return np.clip(y, *ybounds) if ybounds is not None else y


# ------------------------------------------------------------------ baselines with clustered SEs

def _cluster_cov(bread: np.ndarray, scores: np.ndarray, groups: np.ndarray, ols: bool) -> np.ndarray:
    """Sandwich covariance clustered by borrower, with the usual G / (G - 1) small-sample
    factor (and (n - 1) / (n - k) for OLS)."""
    S = pd.DataFrame(scores).groupby(groups).sum().to_numpy()
    G, (n, k) = S.shape[0], scores.shape
    c = G / (G - 1) * ((n - 1) / (n - k) if ols else 1.0)
    return c * bread @ (S.T @ S) @ bread


def _loglik(X1: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    eta = X1 @ beta
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def logit_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray, max_iter: int = 200) -> dict:
    """Unpenalized logit by Newton-Raphson with step halving (plain Newton overshoots with rare
    events); covariance clustered by `groups`."""
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
    H = X1.T @ (X1 * (p * (1 - p))[:, None])
    p0 = y.mean()
    ll0 = len(y) * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    cov = _cluster_cov(np.linalg.inv(H), X1 * (y - p)[:, None], groups, ols=False)
    return {"beta": beta, "cov": cov, "fit": 1 - ll / ll0, "fitted": p}


def ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    e = y - X1 @ beta
    cov = _cluster_cov(np.linalg.inv(X1.T @ X1), X1 * e[:, None], groups, ols=True)
    return {"beta": beta, "cov": cov, "fit": 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean())), "fitted": X1 @ beta}


def coefficient_table(d: pd.DataFrame, cols: list[str], kind: str, label: dict) -> pd.DataFrame:
    """In-sample baseline (logit or OLS) on frame `d`, predictors expressed per
    UNITS step; standard errors clustered by borrower. One row per term."""
    X = design(d, cols) / np.array([UNITS.get(c, 1.0) for c in cols])
    y = outcome_vector(d, outcome_bounds(d, kind))
    groups = d["borrower"].to_numpy()
    r = logit_cluster(X, y, groups) if kind == "binary" else ols_cluster(X, y, groups)
    se = np.sqrt(np.diag(r["cov"]))
    z = r["beta"] / se
    out = pd.DataFrame({"term": ["const"] + cols, "per": [np.nan] + [UNITS.get(c, 1.0) for c in cols],
                        "coef": r["beta"], "se_cluster": se, "z": z, "p_value": 2 * norm.sf(np.abs(z)),
                        "ci_low": r["beta"] - 1.96 * se, "ci_high": r["beta"] + 1.96 * se})
    if kind == "binary":
        out["odds_ratio"] = np.exp(out["coef"])
        out["or_ci_low"], out["or_ci_high"] = np.exp(out["ci_low"]), np.exp(out["ci_high"])
    for k, v in {**label, "n_obs": len(d), "n_borrowers": d["borrower"].nunique(),
                 "n_events": int(y.sum()) if kind == "binary" else np.nan,
                 ("pseudo_r2" if kind == "binary" else "r2"): r["fit"],
                 "auc_in_sample": roc_auc_score(y, r["fitted"]) if kind == "binary" else np.nan}.items():
        out[k] = v
    return out


# ------------------------------------------------------------------ prediction models

def make_model(kind: str, name: str, **p):
    scaled = lambda est: make_pipeline(StandardScaler(), est)
    if kind == "binary":
        return {"logit": lambda: scaled(LogisticRegression(penalty=None, max_iter=5000)),
                "ridge": lambda: scaled(LogisticRegression(penalty="l2", max_iter=5000, **p)),
                "lasso": lambda: scaled(LogisticRegression(penalty="l1", solver="saga", max_iter=5000,
                                                           random_state=SEED, **p)),   # saga shuffles: seed it
                "gbt": lambda: HistGradientBoostingClassifier(max_iter=200, min_samples_leaf=100, early_stopping=False,
                                                              random_state=SEED, **p),
                "mlp": lambda: scaled(MLPClassifier(hidden_layer_sizes=(MLP_HIDDEN,), max_iter=1000,
                                                    random_state=SEED, **p))}[name]()
    return {"ols": lambda: scaled(LinearRegression())}[name]()


def mlp_allowed(n_features: int, n_events: int) -> bool:
    n_params = n_features * MLP_HIDDEN + 2 * MLP_HIDDEN + 1
    return n_events / n_params >= MLP_MIN_EVENTS_PER_PARAM


def fit_predict(kind: str, name: str, params: dict, train: pd.DataFrame, test: pd.DataFrame,
                cols: list[str]) -> tuple[np.ndarray, np.ndarray, object]:
    """Fit on `train` and predict `test`. Returns (test predictions, test y clipped to the
    training outcome bounds, fitted model)."""
    ybounds = outcome_bounds(train, kind)
    model = make_model(kind, name, **params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(design(train, cols), outcome_vector(train, ybounds))
    Xte = design(test, cols)
    pred = model.predict_proba(Xte)[:, 1] if kind == "binary" else model.predict(Xte)
    return pred, outcome_vector(test, ybounds), model


def _score(kind: str, y: np.ndarray, pred: np.ndarray) -> float:
    """Validation score, higher is better: PR-AUC (non-accrual) or minus the MSE."""
    if kind == "binary":
        return average_precision_score(y, pred) if y.sum() > 0 else np.nan
    return -mean_squared_error(y, pred)


def validate(kind: str, name: str, d: pd.DataFrame, cols: list[str], train_end: int) -> dict:
    """Expanding-window validation over the last N_VAL_QUARTERS predictor quarters up to
    train_end: for each, fit on the quarters before it and predict it. Picks the grid point
    with the best pooled score; returns it with the pooled out-of-fold predictions."""
    folds = [q for q in sorted(d["qsort"].unique()) if train_end - N_VAL_QUARTERS < q <= train_end]
    best = None
    for params in GRIDS[kind][name]:
        ys, ps = [], []
        for v in folds:
            tr, va = d[d["qsort"] < v], d[d["qsort"] == v]
            if va.empty or (kind == "binary" and tr["y"].sum() == 0):
                continue
            p, y, _ = fit_predict(kind, name, params, tr, va, cols)
            ys.append(y), ps.append(p)
        y, p = np.concatenate(ys), np.concatenate(ps)
        s = _score(kind, y, p)
        if best is None or s > best["score"]:
            best = {"params": params, "score": s, "y_oof": y, "p_oof": p}
    return best


def recall_threshold(y: np.ndarray, p: np.ndarray, target: float = TARGET_RECALL) -> float:
    """Highest probability cut-off that still flags `target` of the positives in (y, p)."""
    pos = np.sort(p[y == 1])[::-1]
    return float(pos[int(np.ceil(target * len(pos))) - 1]) if len(pos) else np.nan


def _boot_ci(y: np.ndarray, p: np.ndarray, fn, n: int = N_BOOT) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(fn(y[i], p[i]))
    return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float, boot: bool = True) -> dict:
    flag = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, flag, labels=[0, 1]).ravel()
    out = {"n": len(y), "events": int(y.sum()), "base_rate": y.mean(), "threshold": threshold,
           "flagged": int(flag.sum()), "flag_rate": flag.mean(),
           "recall": tp / (tp + fn) if tp + fn else np.nan, "precision": tp / (tp + fp) if tp + fp else np.nan,
           "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
    if 0 < y.sum() < len(y):
        out["roc_auc"], out["pr_auc"] = roc_auc_score(y, p), average_precision_score(y, p)
        if boot:
            out["roc_auc_lo"], out["roc_auc_hi"] = _boot_ci(y, p, roc_auc_score)
            out["pr_auc_lo"], out["pr_auc_hi"] = _boot_ci(y, p, average_precision_score)
    return out


def regression_metrics(y: np.ndarray, pred: np.ndarray, train_mean: float) -> dict:
    sse = np.sum((y - pred) ** 2)
    return {"n": len(y), "r2": r2_score(y, pred), "r2_vs_train_mean": 1 - sse / np.sum((y - train_mean) ** 2),
            "mae": mean_absolute_error(y, pred), "rmse": np.sqrt(mean_squared_error(y, pred))}


def evaluate(kind: str, name: str, d: pd.DataFrame, cols: list[str], train_end: int,
             test_q: int, boot: bool = True) -> tuple[dict, pd.DataFrame]:
    """Validate (hyperparameters and, for non-accrual, the threshold) on the training window,
    refit on the labelled predictor quarters <= train_end and score quarter test_q. `d` is a
    model_frame(..., labelled=False): test rows whose outcome is not filed yet are still
    scored, and the test metrics are left out. Returns (metrics, test predictions)."""
    train = d[(d["qsort"] <= train_end) & d["y"].notna()]
    test = d[d["qsort"] == test_q]
    v = validate(kind, name, train, cols, train_end)
    pred, y, _ = fit_predict(kind, name, v["params"], train, test, cols)
    labelled = len(test) > 0 and test["y"].notna().all()
    res = {"model": name, "params": v["params"], "val_score": v["score"], "n_train": len(train),
           "train_borrowers": train["borrower"].nunique(),
           "train_events": int(train["y"].sum()) if kind == "binary" else np.nan,
           "n_test": len(test), "test_labelled": labelled}
    if kind == "binary":
        thr = recall_threshold(v["y_oof"], v["p_oof"])
        res["val_recall_at_threshold"] = float(np.mean(v["p_oof"][v["y_oof"] == 1] >= thr))
        res["val_flag_rate"] = float(np.mean(v["p_oof"] >= thr))
        res.update(classification_metrics(y, pred, thr, boot) if labelled else {"threshold": thr})
    elif labelled:
        res.update(regression_metrics(y, pred, float(outcome_vector(train, outcome_bounds(train, kind)).mean())))
    preds = test[["borrower", "cal_q", "outcome_q"]].assign(y=test["y"].to_numpy(), pred=pred, model=name)
    if kind == "binary":
        preds["flag"] = preds["pred"] >= res["threshold"]
    return res, preds


# ------------------------------------------------------------------ rolling (expanding-window) evaluation

def rolling_predictions(d: pd.DataFrame, cols: list[str], kind: str, h: int, models=("logit",),
                        min_train_quarters: int = ROLL_MIN_TRAIN_QUARTERS) -> pd.DataFrame:
    """Out-of-sample scores for every quarter: for test predictor quarter q, fit on the labelled
    quarters <= q - h (their outcomes, up to quarter q, are all filed by the time q is scored) and
    score q. Quarters with fewer than min_train_quarters training quarters are skipped.
    Hyperparameters are fixed (ROLL_PARAMS). Returns one row per test borrower-quarter and model."""
    d = d[d["y"].notna()]
    folds = []
    for q in sorted(d["qsort"].unique()):
        train = d[d["qsort"] <= q - h]
        if train["qsort"].nunique() >= min_train_quarters:
            folds.append((q, train, d[d["qsort"] == q]))
    out = []
    for q, train, test in folds:
        ybounds = outcome_bounds(train, kind)
        train_mean = float(outcome_vector(train, ybounds).mean()) if ybounds else np.nan
        for name in models:
            pred, y, _ = fit_predict(kind, name, ROLL_PARAMS.get(name, {}), train, test, cols)
            out.append(test[["borrower", "cal_q", "outcome_q", "qsort"]].assign(
                y=y, pred=pred, model=name, train_last_q=train["qsort"].max(),
                train_quarters=train["qsort"].nunique(), train_mean=train_mean))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _cluster_boot(y: np.ndarray, p: np.ndarray, groups: np.ndarray, fn, n: int = N_BOOT_ROLL) -> tuple:
    """95% interval of fn(y, p) when borrowers (not rows) are resampled: a borrower appears in
    several quarters, so rows are not independent."""
    codes, uniq = pd.factorize(groups)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes]
        keep = w > 0
        if 0 < (w * y)[keep].sum() < w[keep].sum():
            vals.append(fn(y[keep], p[keep], sample_weight=w[keep]))
    return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)


def quarter_rank(p: pd.DataFrame) -> pd.Series:
    """Each row's rank within its quarter as a share (0.01 = top 1% of that quarter's borrowers)."""
    return p.groupby("qsort")["pred"].rank(method="first", ascending=False, pct=True)


def ranking_metrics(p: pd.DataFrame, boot: bool = True) -> dict:
    """Pooled out-of-sample ranking metrics over the test quarters of `p` (rolling_predictions
    rows of one model), with borrower-resampled 95% intervals, and the range of the per-quarter
    ROC-AUC."""
    y, s, g = p["y"].to_numpy(), p["pred"].to_numpy(), p["borrower"].to_numpy()
    per_q = p.groupby("qsort").apply(lambda x: roc_auc_score(x["y"], x["pred"]) if 0 < x["y"].sum() < len(x) else np.nan)
    out = {"test_quarters": p["qsort"].nunique(), "first_test_q": p["cal_q"].min(), "last_test_q": p["cal_q"].max(),
           "n": len(p), "borrowers": p["borrower"].nunique(), "positives": int(y.sum()), "base_rate": y.mean(),
           "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
           "auc_quarter_min": per_q.min(), "auc_quarter_median": per_q.median(), "auc_quarter_max": per_q.max()}
    out["pr_lift"] = out["pr_auc"] / out["base_rate"]
    if boot:
        out["roc_auc_lo"], out["roc_auc_hi"] = _cluster_boot(y, s, g, roc_auc_score)
        out["pr_auc_lo"], out["pr_auc_hi"] = _cluster_boot(y, s, g, average_precision_score)
    return out


def review_capture(p: pd.DataFrame, shares=REVIEW_SHARES) -> pd.DataFrame:
    """Review list: flag the top `share` of borrowers each quarter by score. Share of the
    positives caught (capture = recall), precision and lift over the base rate, pooled."""
    r = quarter_rank(p)
    nq = p["qsort"].nunique()
    rows = []
    for k in shares:
        f = (r <= k).to_numpy()
        y = p["y"].to_numpy()
        rows.append({"share_flagged": k, "flagged": int(f.sum()), "flagged_per_quarter": f.sum() / nq,
                     "caught": int(y[f].sum()), "positives": int(y.sum()), "capture": y[f].sum() / y.sum(),
                     "precision": y[f].mean(), "lift": y[f].mean() / y.mean()})
    return pd.DataFrame(rows)


def calibration_table(p: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability against the actual rate, by decile of the pooled scores."""
    dec = pd.qcut(p["pred"].rank(method="first"), bins, labels=False) + 1
    return (p.groupby(dec).agg(n=("y", "size"), mean_pred=("pred", "mean"), actual=("y", "mean"),
                               positives=("y", "sum")).rename_axis("decile").reset_index())


def rolling_regression_metrics(p: pd.DataFrame) -> dict:
    """Pooled out-of-sample fit of a continuous outcome: R2 against each fold's training mean
    (the forecast a model with no predictors would make), MAE, RMSE, and the per-quarter range."""
    e = p["y"] - p["pred"]
    e0 = p["y"] - p["train_mean"]
    per_q = p.assign(_e=e ** 2, _e0=e0 ** 2).groupby("qsort").apply(lambda x: 1 - x["_e"].sum() / x["_e0"].sum())
    return {"test_quarters": p["qsort"].nunique(), "first_test_q": p["cal_q"].min(), "last_test_q": p["cal_q"].max(),
            "n": len(p), "r2_vs_train_mean": 1 - (e ** 2).sum() / (e0 ** 2).sum(),
            "mae": e.abs().mean(), "rmse": np.sqrt((e ** 2).mean()),
            "r2_quarter_min": per_q.min(), "r2_quarter_median": per_q.median(), "r2_quarter_max": per_q.max()}
