"""Evaluation of the early-warning model (src/9_prediction): the time-ordered splits (one fit, or a rolling
fit that only ever sees closed windows), and the metrics with their intervals: ROC-AUC and PR-AUC
with borrower-resampled bootstrap intervals, Wilson intervals for rates, score deciles, and R2 /
RMSE for the markdown outcomes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import HORIZON, MIN_TRAIN_QUARTERS, N_BOOT, SEED
from models.logit import fit_predict


def _predictions(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    return test[["borrower", "cal_q", "outcome_q", "qsort", "y"]].assign(pred=pred)


def split_predictions(d: pd.DataFrame, cols: list[str], train_end: int, test_qs: list[int]) -> pd.DataFrame:
    """One fit on the labelled quarters <= train_end, scored on the quarters `test_qs`. A training
    quarter's window closes HORIZON quarters later, so every test quarter is at least train_end +
    HORIZON: the outcomes fitted on are all filed when the test quarter is scored. `d` may hold
    unlabelled rows: they are scored, never fitted on. train_auc is the same model's ROC-AUC on the
    rows it was fitted on: a test ROC-AUC well below it would mean overfitting."""
    if min(test_qs) < train_end + HORIZON:
        raise ValueError("training windows must close by the first test quarter")
    train = d[(d["qsort"] <= train_end) & d["y"].notna()]
    test = d[d["qsort"].isin(test_qs)]
    return _predictions(test, fit_predict(train, test, cols)).assign(
        train_auc=roc_auc_score(train["y"], fit_predict(train, train, cols)))


def rolling_predictions(d: pd.DataFrame, cols: list[str],
                        min_train_quarters: int = MIN_TRAIN_QUARTERS) -> pd.DataFrame:
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
            out.append(_predictions(test, fit_predict(train, test, cols)).assign(
                train_last_q=train["qsort"].max(), train_events=int(train["y"].sum()),
                train_auc=roc_auc_score(train["y"], fit_predict(train, train, cols))))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _boot_weights(groups: np.ndarray, n: int = N_BOOT):
    """Bootstrap weights when borrowers (not rows) are resampled: a borrower appears in several
    quarters, so rows are not independent."""
    codes, uniq = pd.factorize(groups)
    rng = np.random.default_rng(SEED)
    for _ in range(n):
        yield np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes]


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
    rate (about 0.008 here), not 0.5."""
    y, s = p["y"].to_numpy(), p["pred"].to_numpy()
    vals = [_ap(y, s, w) for w in _boot_weights(p["borrower"].to_numpy())]
    return (average_precision_score(y, s), *np.nanpercentile(vals, [2.5, 97.5]))


def recall_at(p: pd.DataFrame, share: float) -> tuple[float, float, float]:
    """Share of the events caught by reviewing the riskiest `share` of each quarter's borrowers
    (Wilson 95% interval): the miss rate for a fixed review budget is one minus this."""
    y = p["y"].to_numpy()
    caught = int(y[(quarter_rank(p) <= share).to_numpy()].sum())
    return (caught / y.sum(), *wilson(caught, int(y.sum())))


def auc_gain(p: pd.DataFrame, base: pd.DataFrame) -> tuple[float, float, float]:
    """ROC-AUC of `p` minus that of `base` on the rows they share, with a borrower-resampled 95%
    interval (both scored on the same resamples, so the interval is for the difference)."""
    j = p.merge(base[["borrower", "qsort", "pred"]], on=["borrower", "qsort"], suffixes=("", "_base"),
                validate="one_to_one")
    y, a, b = j["y"].to_numpy(), j["pred"].to_numpy(), j["pred_base"].to_numpy()
    vals = [_auc(y, a, w) - _auc(y, b, w) for w in _boot_weights(j["borrower"].to_numpy())]
    return (roc_auc_score(y, a) - roc_auc_score(y, b), *np.nanpercentile(vals, [2.5, 97.5]))


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a rate of k events in n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def quarter_rank(p: pd.DataFrame) -> pd.Series:
    """Each row's rank within its quarter as a share (0.01 = top 1% of that quarter's borrowers)."""
    return p.groupby("qsort")["pred"].rank(method="first", ascending=False, pct=True)


def score_decile(p: pd.DataFrame, bins: int = 10) -> pd.Series:
    """Score decile within the quarter: 10 = the riskiest tenth of that quarter's borrowers."""
    return (bins + 1 - np.ceil((quarter_rank(p) * bins).round(9))).astype(int)


def decile_table(p: pd.DataFrame, probabilities: bool = True) -> pd.DataFrame:
    """Realized event rate by score decile (Wilson 95% interval), the share of all events each decile
    holds and, when the scores are probabilities, the predicted rate next to the realized one."""
    g = p.assign(decile=score_decile(p)).groupby("decile")
    t = g.agg(n=("y", "size"), events=("y", "sum")).reset_index().sort_values("decile", ascending=False)
    t["event_rate"] = t["events"] / t["n"]
    ci = [wilson(k, n) for k, n in zip(t["events"], t["n"])]
    t["rate_ci_low"], t["rate_ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    t["predicted_rate"] = g["pred"].mean().reindex(t["decile"]).to_numpy() if probabilities else np.nan
    t["share_of_events"] = t["events"] / t["events"].sum()
    return t.reset_index(drop=True)


def performance(p: pd.DataFrame, probabilities: bool = True) -> dict:
    """Out-of-sample summary of predictions `p`: ROC-AUC, PR-AUC, the share of events caught in the
    riskiest 10% and 20%, the riskiest tenth's event rate, and predicted against realized events,
    each with a 95% interval."""
    auc, lo, hi = auc_ci(p)
    ap, ap_lo, ap_hi = pr_auc_ci(p)
    r20, r20_lo, r20_hi = recall_at(p, 0.20)
    top = decile_table(p, probabilities).iloc[0]
    n, events = len(p), int(p["y"].sum())
    r_lo, r_hi = wilson(events, n)
    return {"test_quarters": f"{p['cal_q'].min()} to {p['cal_q'].max()}", "n": n, "events": events,
            "roc_auc": auc, "roc_auc_lo": lo, "roc_auc_hi": hi,
            "pr_auc": ap, "pr_auc_lo": ap_lo, "pr_auc_hi": ap_hi,
            "recall_top_20pct": r20, "recall_top_20pct_lo": r20_lo, "recall_top_20pct_hi": r20_hi,
            "top_decile_event_rate": top["event_rate"], "top_decile_rate_lo": top["rate_ci_low"],
            "top_decile_rate_hi": top["rate_ci_high"], "top_decile_share_of_events": top["share_of_events"],
            "realized_rate": events / n, "realized_rate_lo": r_lo, "realized_rate_hi": r_hi,
            "predicted_rate": p["pred"].mean() if probabilities else np.nan}


def regression_scores(y: pd.Series, pred: np.ndarray, benchmark: float) -> dict:
    """R2 and RMSE of `pred` against `y`, with the RMSE and R2 of always predicting `benchmark` (the
    training mean): R2 = 1 - SSE / SST about the benchmark, so it is out of sample when `benchmark`
    comes from the training rows."""
    sse, sse0 = ((y - pred) ** 2).sum(), ((y - benchmark) ** 2).sum()
    return {"r2": 1 - sse / sse0, "rmse": np.sqrt(sse / len(y)), "rmse_benchmark": np.sqrt(sse0 / len(y))}
