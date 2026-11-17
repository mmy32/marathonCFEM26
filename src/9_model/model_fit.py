"""Estimation and evaluation for the early-warning scorecard (src/9_model): model frames, the
logistic regression with borrower-clustered standard errors, the equal-weighted ranking, the out-of-sample predictions (rolling and single split) and their
metrics with confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model_panel import HORIZON

OUTCOME = "y_event"            # the one outcome: an accruing borrower has a credit event at t + 1 or t + 2
                               # (non-accrual, or a lender's concession: PIK switch, debt-for-equity, write-down)
# Fixed caps: a mark below 70 counts as 70, a move of more than 20 points in the quarter (or a fall of
# more than 20 over the year) as 20, six or more lender families as five. Beyond them risk keeps
# rising, but far more slowly than the logistic's straight line. The caps were fixed on 2026-11-13
# and kept on 2026-11-16 when the outcome was non-accrual alone; caps_comparison.csv shows the
# scorecard with and without them on the current outcome.
PREDICTOR_CAPS = {"price": (0.70, None), "d_price": (-0.20, 0.20), "max_drawdown": (None, 0.20),
                  "crowding": (None, 5)}
# The baseline: the mark and its change over the quarter. Each other sign is added to it alone and
# kept only if it improves the ranking out of sample (model_analysis.sign_selection).
BASELINE = ["price", "d_price"]
# Dispersion (the valuation gap: highest minus lowest lender family's mark) exists only for
# borrowers with a tranche held by 2+ manager families; it is 0 elsewhere and comes with its indicator.
CANDIDATES = {"pik_switch": ["pik_switch"], "dispersion": ["dispersion", "has_dispersion"],
              "crowding": ["crowding"], "max_drawdown": ["max_drawdown"]}
# The baseline and the signs the frozen rule kept on the development quarters (2026-11-16, outcome:
# a credit event within two quarters): PIK switch and fall over the past year; the valuation gap
# (+0.004, under the rule's +0.005) and crowding were dropped. With non-accrual alone as the
# outcome the rule had also kept the valuation gap. run_model.py stops if sign_selection.csv ever
# keeps a different set.
SCORECARD = BASELINE + CANDIDATES["pik_switch"] + CANDIDATES["max_drawdown"]
RISK_SIGN = {"price": -1, "d_price": -1, "pik_switch": 1, "dispersion": 1, "crowding": 1,
             "max_drawdown": 1}             # equal-weighted ranking: +1 if a higher value means more risk
STRUCTURAL_ZERO = {"dispersion", "dispersion_std"}       # 0 where has_dispersion == 0
# Coefficients are reported per these steps, in the predictor's own units (0.10 = ten points of par)
UNITS = {"price": 0.10, "price_rel": 0.10, "d_price": 0.05, "d_price_adj": 0.05, "max_drawdown": 0.05,
         "dispersion": 0.01, "dispersion_std": 0.01, "d_pik_pp": 10.0, "junior_share": 0.5}

MIN_TRAIN_QUARTERS = 3         # rolling: a test quarter needs at least three labelled training quarters
N_BOOT = 500
SEED = 0


# ------------------------------------------------------------------ frames and preprocessing

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


# ------------------------------------------------------------------ logit with clustered SEs

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


def coefficient_table(d: pd.DataFrame, cols: list[str], label: dict, caps: dict = PREDICTOR_CAPS) -> pd.DataFrame:
    """In-sample logit on frame `d`, predictors expressed per UNITS step; standard errors clustered
    by borrower. One row per term."""
    X = design(d, cols, caps) / np.array([UNITS.get(c, 1.0) for c in cols])
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


# ------------------------------------------------------------------ predictions

def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], caps: dict = PREDICTOR_CAPS) -> np.ndarray:
    """Fit the unpenalized logit on `train` and return the predicted probabilities for `test`."""
    model = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=5000))
    model.fit(design(train, cols, caps), train["y"].to_numpy(dtype=float))
    return model.predict_proba(design(test, cols, caps))[:, 1]


def _predictions(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    return test[["borrower", "cal_q", "outcome_q", "qsort", "y"]].assign(pred=pred)


def split_predictions(d: pd.DataFrame, cols: list[str], train_end: int, test_qs: list[int],
                      caps: dict = PREDICTOR_CAPS) -> pd.DataFrame:
    """One fit on the labelled quarters <= train_end, scored on the quarters `test_qs`. A training
    quarter's window closes HORIZON quarters later, so every test quarter is at least train_end +
    HORIZON: the outcomes fitted on are all filed when the test quarter is scored. `d` may hold
    unlabelled rows: they are scored, never fitted on. train_auc is the same model's ROC-AUC on the
    rows it was fitted on: a test ROC-AUC well below it would mean overfitting."""
    if min(test_qs) < train_end + HORIZON:
        raise ValueError("training windows must close by the first test quarter")
    train = d[(d["qsort"] <= train_end) & d["y"].notna()]
    test = d[d["qsort"].isin(test_qs)]
    return _predictions(test, fit_predict(train, test, cols, caps)).assign(
        train_auc=roc_auc_score(train["y"], fit_predict(train, train, cols, caps)))


def rolling_predictions(d: pd.DataFrame, cols: list[str],
                        min_train_quarters: int = MIN_TRAIN_QUARTERS, caps: dict = PREDICTOR_CAPS) -> pd.DataFrame:
    """Out-of-sample scores for every quarter: quarter q is scored by a logit fitted on the labelled
    quarters up to q - HORIZON (their windows close by q, so their outcomes are filed when q is
    scored).
    Quarters with fewer than min_train_quarters training quarters are skipped. train_auc is each
    fit's ROC-AUC on its own training rows."""
    d = d[d["y"].notna()]
    out = []
    for q in sorted(d["qsort"].unique()):
        train = d[d["qsort"] <= q - HORIZON]
        if train["qsort"].nunique() >= min_train_quarters:
            test = d[d["qsort"] == q]
            out.append(_predictions(test, fit_predict(train, test, cols, caps)).assign(
                train_last_q=train["qsort"].max(), train_events=int(train["y"].sum()),
                train_auc=roc_auc_score(train["y"], fit_predict(train, train, cols, caps))))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def equal_weight_score(d: pd.DataFrame, cols: list[str]) -> pd.Series:
    """The equal-weighted ranking of the same signs, with nothing fitted: each sign is ranked within
    the quarter (as a share, riskiest = 1; ties share their average rank) and the ranks are averaged."""
    signs = [c for c in cols if c in RISK_SIGN]
    X = d[signs].astype(float).fillna(0.0) * [RISK_SIGN[c] for c in signs]
    return X.groupby(d["qsort"]).rank(pct=True).mean(axis=1)


# ------------------------------------------------------------------ metrics

def _boot_counts(n_groups: int, n: int = N_BOOT):
    """How many times each of n_groups borrowers is drawn, in each of n resamples."""
    rng = np.random.default_rng(SEED)
    for _ in range(n):
        yield np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)


def _boot_weights(groups: np.ndarray, n: int = N_BOOT):
    """Bootstrap weights when borrowers (not rows) are resampled: a borrower appears in several
    quarters, so rows are not independent."""
    codes, uniq = pd.factorize(groups)
    for c in _boot_counts(len(uniq), n):
        yield c[codes]


def ratio_ci(num, den, groups) -> tuple[float, float]:
    """Borrower-resampled 95% interval for sum(num) / sum(den). Every rate in the model uses it
    (event rates: num = y, den = 1; recall: num = caught events, den = events), so a borrower whose
    one default is counted in two overlapping windows counts as one draw, not two."""
    t = pd.DataFrame({"num": np.asarray(num, float), "den": np.asarray(den, float),
                      "g": np.asarray(groups)}).groupby("g")[["num", "den"]].sum()
    if t["den"].sum() == 0:
        return np.nan, np.nan
    a, b = t["num"].to_numpy(), t["den"].to_numpy()
    vals = [(c @ a) / (c @ b) if c @ b > 0 else np.nan for c in _boot_counts(len(t))]
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def _auc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    k = w > 0
    return roc_auc_score(y[k], s[k], sample_weight=w[k]) if 0 < (w * y)[k].sum() < w[k].sum() else np.nan


def _ap(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    k = w > 0
    return average_precision_score(y[k], s[k], sample_weight=w[k]) if (w * y)[k].sum() > 0 else np.nan


def auc_ci(p: pd.DataFrame) -> tuple[float, float, float]:
    """ROC-AUC of predictions `p` with a borrower-resampled 95% interval."""
    y, s = p["y"].to_numpy(), p["pred"].to_numpy()
    vals = [_auc(y, s, w) for w in _boot_weights(p["borrower"].to_numpy())]
    return (roc_auc_score(y, s), *np.nanpercentile(vals, [2.5, 97.5]))


def pr_auc_ci(p: pd.DataFrame) -> tuple[float, float, float]:
    """PR-AUC (average precision) of predictions `p` with a borrower-resampled 95% interval. It
    scores only how well the events are found, so a model that ranks at random scores the event
    rate (about 0.02 here), not 0.5."""
    y, s = p["y"].to_numpy(), p["pred"].to_numpy()
    vals = [_ap(y, s, w) for w in _boot_weights(p["borrower"].to_numpy())]
    return (average_precision_score(y, s), *np.nanpercentile(vals, [2.5, 97.5]))


def recall_at(p: pd.DataFrame, share: float) -> tuple[float, float, float]:
    """Share of the events caught by reviewing the riskiest `share` of each quarter's borrowers
    (borrower-resampled 95% interval; the review list is held fixed): the miss rate for a fixed
    review budget is one minus this."""
    y = p["y"].to_numpy()
    caught = y * (quarter_rank(p) <= share).to_numpy()
    return (caught.sum() / y.sum(), *ratio_ci(caught, y, p["borrower"]))


def auc_gain(p: pd.DataFrame, base: pd.DataFrame) -> tuple[float, float, float]:
    """ROC-AUC of `p` minus that of `base` on the rows they share, with a borrower-resampled 95%
    interval (both scored on the same resamples, so the interval is for the difference)."""
    j = p.merge(base[["borrower", "qsort", "pred"]], on=["borrower", "qsort"], suffixes=("", "_base"),
                validate="one_to_one")
    y, a, b = j["y"].to_numpy(), j["pred"].to_numpy(), j["pred_base"].to_numpy()
    vals = [_auc(y, a, w) - _auc(y, b, w) for w in _boot_weights(j["borrower"].to_numpy())]
    return (roc_auc_score(y, a) - roc_auc_score(y, b), *np.nanpercentile(vals, [2.5, 97.5]))


def quarter_rank(p: pd.DataFrame) -> pd.Series:
    """Each row's rank within its quarter as a share (0.01 = top 1% of that quarter's borrowers)."""
    return p.groupby("qsort")["pred"].rank(method="first", ascending=False, pct=True)


def score_decile(p: pd.DataFrame, bins: int = 10) -> pd.Series:
    """Score decile within the quarter: 10 = the riskiest tenth of that quarter's borrowers."""
    return (bins + 1 - np.ceil((quarter_rank(p) * bins).round(9))).astype(int)


def decile_table(p: pd.DataFrame, probabilities: bool = True) -> pd.DataFrame:
    """Realized event rate by score decile and the share of all events each decile holds. When the
    scores are probabilities, also the predicted rate and realized / predicted (1 = the model
    predicts the right number of events; 1.25 = 25% more events than predicted). Intervals are
    borrower-resampled 95%, with the deciles and the predictions held fixed."""
    p = p.assign(decile=score_decile(p))
    t = p.groupby("decile").agg(n=("y", "size"), events=("y", "sum"), expected=("pred", "sum")).reset_index()
    t = t.sort_values("decile", ascending=False).reset_index(drop=True)
    parts = [p[p["decile"] == k] for k in t["decile"]]
    t["event_rate"] = t["events"] / t["n"]
    t["rate_ci_low"], t["rate_ci_high"] = zip(*(ratio_ci(s["y"], np.ones(len(s)), s["borrower"]) for s in parts))
    t["predicted_rate"] = t["expected"] / t["n"] if probabilities else np.nan
    if probabilities:
        t["realized_to_predicted"] = t["events"] / t["expected"]
        t["ratio_ci_low"], t["ratio_ci_high"] = zip(*(ratio_ci(s["y"], s["pred"], s["borrower"]) for s in parts))
    t["share_of_events"] = t["events"] / t["events"].sum()
    return t.drop(columns="expected")


def performance(p: pd.DataFrame, probabilities: bool = True) -> dict:
    """Out-of-sample summary of predictions `p`: ROC-AUC, PR-AUC, the share of events caught in the
    riskiest 10% and 20%, the riskiest tenth's event rate, and the realized against the predicted
    event rate (realized / predicted: 1 = right on average), each with a 95% interval."""
    auc, lo, hi = auc_ci(p)
    ap, ap_lo, ap_hi = pr_auc_ci(p)
    r20, r20_lo, r20_hi = recall_at(p, 0.20)
    top = decile_table(p, probabilities).iloc[0]
    n, events = len(p), int(p["y"].sum())
    r_lo, r_hi = ratio_ci(p["y"], np.ones(n), p["borrower"])
    out = {"test_quarters": f"{p['cal_q'].min()} to {p['cal_q'].max()}", "n": n, "events": events,
           "roc_auc": auc, "roc_auc_lo": lo, "roc_auc_hi": hi,
           "pr_auc": ap, "pr_auc_lo": ap_lo, "pr_auc_hi": ap_hi,
           "recall_top_20pct": r20, "recall_top_20pct_lo": r20_lo, "recall_top_20pct_hi": r20_hi,
           "top_decile_event_rate": top["event_rate"], "top_decile_rate_lo": top["rate_ci_low"],
           "top_decile_rate_hi": top["rate_ci_high"], "top_decile_share_of_events": top["share_of_events"],
           "realized_rate": events / n, "realized_rate_lo": r_lo, "realized_rate_hi": r_hi,
           "predicted_rate": np.nan, "realized_to_predicted": np.nan, "ratio_lo": np.nan, "ratio_hi": np.nan}
    if probabilities:
        out["predicted_rate"] = p["pred"].mean()
        out["realized_to_predicted"] = events / p["pred"].sum()
        out["ratio_lo"], out["ratio_hi"] = ratio_ci(p["y"], p["pred"], p["borrower"])
    return out
