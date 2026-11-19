"""Evaluation of the early-warning model (src/9_prediction): the time-ordered splits (one fit, or a rolling
fit that only ever sees closed windows), and the metrics with their intervals: ROC-AUC and PR-AUC
with borrower-resampled bootstrap intervals (also for rates and ratios), score deciles, and R2 /
RMSE for the markdown outcomes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import HORIZON, MIN_TRAIN_QUARTERS, N_BOOT, SEED
from features import PREDICTOR_CAPS
from models.logit import fit_predict


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


def regression_scores(y: pd.Series, pred: np.ndarray, benchmark: float) -> dict:
    """R2 and RMSE of `pred` against `y`, with the RMSE and R2 of always predicting `benchmark` (the
    training mean): R2 = 1 - SSE / SST about the benchmark, so it is out of sample when `benchmark`
    comes from the training rows."""
    sse, sse0 = ((y - pred) ** 2).sum(), ((y - benchmark) ** 2).sum()
    return {"r2": 1 - sse / sse0, "rmse": np.sqrt(sse / len(y)), "rmse_benchmark": np.sqrt(sse0 / len(y))}
