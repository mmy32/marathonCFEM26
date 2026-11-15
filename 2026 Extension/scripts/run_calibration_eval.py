"""Calibration of the early-warning scorecard on the out-of-sample test, done three ways so the
old check and the stricter ones can be compared on the same predictions:

  A. old:    per-decile Wilson interval on the realized rate, predicted rate taken as exact
             (what model_fit.decile_table reports).
  B. fixed:  borrower-cluster bootstrap of the TEST set, predictions held fixed
             (sampling noise of the realized rates, allowing for a borrower appearing in two
             test quarters with overlapping windows).
  C. refit:  borrower-cluster bootstrap of training AND test borrowers together, the logit refitted
             on every resample (adds the estimation noise of the predictions themselves).

Each gives: observed / expected events, calibration intercept (calibration-in-the-large, logit
offset), calibration slope, Brier score, and per-decile realized minus predicted.

Recalibration, without touching test labels: a model fitted on signals <= 2024Q2 (windows closed
by 2024Q4) scores the signal quarters 2024Q4-2025Q1 (windows closed by 2025Q3, the first test
quarter), the same shape as the main split. Its calibration intercept (R1) or intercept and slope (R2) on
those quarters is then applied to the main model's test predictions. Every label used is filed
before the first test quarter is scored. The recalibration is redone inside every bootstrap
resample.

Usage: python scripts/run_calibration_eval.py [--boot 1000]   (needs data/model/model_panel.csv
from run_model.py). Writes outputs/model/calibration_*.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "1_data_processing"))
from paths import MODEL_DATA_DIR, MODEL_DIR, add_code_dirs_to_path  # noqa: E402

add_code_dirs_to_path()
from model_fit import SCORECARD, decile_table, design, model_frame, score_decile  # noqa: E402
from model_panel import HORIZON, q_sort  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_model import TEST_QS, TRAIN_END  # noqa: E402

SEED = 20261115


# ------------------------------------------------------------------ fitting with weights

def fit(train: pd.DataFrame, w: np.ndarray | None = None):
    """The same pipeline as model_fit.fit_predict (scaler + unpenalized logit). A bootstrap weight
    w = k fits a borrower-quarter drawn k times exactly as k copies would."""
    model = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=5000))
    kw = {} if w is None else {"standardscaler__sample_weight": w, "logisticregression__sample_weight": w}
    model.fit(design(train, SCORECARD), train["y"].to_numpy(float), **kw)
    return model


def predict(model, d: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(design(d, SCORECARD))[:, 1]


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def recal_params(y, p, w, slope: bool) -> tuple[float, float]:
    """Weighted logistic recalibration of predictions p on outcomes y by Newton's method:
    logit P(y=1) = a + b * logit(p). Intercept only (b fixed at 1, logit(p) an offset) when
    slope is False. Returns (a, b)."""
    lp = _logit(p)
    X = np.column_stack([np.ones_like(lp), lp]) if slope else np.ones((len(lp), 1))
    off = np.zeros_like(lp) if slope else lp
    beta = np.zeros(X.shape[1]) if not slope else np.array([0.0, 1.0])
    for _ in range(100):
        eta = off + X @ beta
        mu = 1 / (1 + np.exp(-eta))
        H = X.T @ (X * (w * mu * (1 - mu))[:, None])
        step = np.linalg.solve(H, X.T @ (w * (y - mu)))
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            break
    return (beta[0], beta[1]) if slope else (beta[0], 1.0)


def apply_recal(p, a, b):
    return 1 / (1 + np.exp(-(a + b * _logit(p))))


# ------------------------------------------------------------------ metrics on a weighted test set

def weighted_deciles(qsort: np.ndarray, pred: np.ndarray, w: np.ndarray, bins: int = 10) -> np.ndarray:
    """Score decile within the quarter on a weighted sample: rows sorted by score, the cumulative
    weight share sets the decile (with unit weights this is model_fit.score_decile)."""
    dec = np.zeros(len(pred), dtype=int)
    for q in np.unique(qsort):
        i = np.where(qsort == q)[0]
        i = i[np.argsort(-pred[i], kind="stable")]
        share = np.cumsum(w[i]) / w[i].sum()
        dec[i] = bins + 1 - np.ceil(np.round(share * bins, 9)).astype(int)
    return dec


def metrics(y, p, w, qsort) -> dict:
    """Calibration metrics of predictions p on outcomes y with row weights w."""
    k = w > 0
    y, p, w, qsort = y[k], p[k], w[k], qsort[k]
    obs, exp = (w * y).sum(), (w * p).sum()
    a, _ = recal_params(y, p, w, slope=False)
    _, b = recal_params(y, p, w, slope=True)
    out = {"observed": obs, "expected": exp, "o_e": obs / exp, "cal_intercept": a, "cal_slope": b,
           "brier": (w * (p - y) ** 2).sum() / w.sum()}
    dec = weighted_deciles(qsort, p, w)
    for d in range(1, 11):
        j = dec == d
        r, e = (w[j] * y[j]).sum() / w[j].sum(), (w[j] * p[j]).sum() / w[j].sum()
        out[f"d{d}_realized"], out[f"d{d}_predicted"], out[f"d{d}_gap"] = r, e, r - e
    return out


# ------------------------------------------------------------------ the three versions

def frames(m: pd.DataFrame):
    train_end = q_sort(TRAIN_END)
    test_qs = [q_sort(q) for q in TEST_QS]
    d = model_frame(m, SCORECARD, labelled=True)
    train = d[d["qsort"] <= train_end].reset_index(drop=True)
    test = d[d["qsort"].isin(test_qs)].reset_index(drop=True)
    # recalibration: fit on windows closed by train_end, score the next HORIZON signal quarters
    # (same shape as the main split: fit <= 2024Q2, skip 2024Q3, score 2024Q4-2025Q1)
    recal_set = d[(d["qsort"] > train_end - HORIZON) & (d["qsort"] <= train_end)].reset_index(drop=True)
    recal_fit = d[d["qsort"] <= recal_set["qsort"].min() - HORIZON].reset_index(drop=True)
    # leakage guards: every label fitted or recalibrated on is filed by the first test quarter
    assert train["qsort"].max() + HORIZON <= min(test_qs)
    assert recal_fit["qsort"].max() + HORIZON <= recal_set["qsort"].min()
    assert recal_set["qsort"].max() + HORIZON <= min(test_qs)
    assert set(test["qsort"]) == set(test_qs)
    return train, test, recal_fit, recal_set


def one_pass(train, test, recal_fit, recal_set, w_of):
    """Fit, recalibrate and score once with borrower weights w_of(frame) (all ones = point estimate)."""
    wt, ws = w_of(train), w_of(test)
    p = predict(fit(train, wt), test)
    pr = predict(fit(recal_fit, w_of(recal_fit)), recal_set)
    wr = w_of(recal_set)
    a1, _ = recal_params(recal_set["y"].to_numpy(float), pr, wr, slope=False)
    a2, b2 = recal_params(recal_set["y"].to_numpy(float), pr, wr, slope=True)
    y, q = test["y"].to_numpy(float), test["qsort"].to_numpy()
    return {"as fitted": metrics(y, p, ws, q),
            "R1: intercept updated": metrics(y, apply_recal(p, a1, 1.0), ws, q),
            "R2: intercept and slope updated": metrics(y, apply_recal(p, a2, b2), ws, q)}, \
        {"R1_a": a1, "R2_a": a2, "R2_b": b2}, p


def main(n_boot: int):
    m = pd.read_csv(MODEL_DATA_DIR / "model_panel.csv", low_memory=False)
    train, test, recal_fit, recal_set = frames(m)
    ones = lambda f: np.ones(len(f))  # noqa: E731
    point, recal, p_hat = one_pass(train, test, recal_fit, recal_set, ones)

    # check: the point estimate reproduces the committed decile table
    ref = pd.read_csv(MODEL_DIR / "oos_deciles.csv").query("model == 'scorecard'")
    mine = decile_table(test.assign(pred=p_hat))
    assert np.allclose(mine["predicted_rate"].to_numpy(), ref["predicted_rate"].to_numpy(), atol=1e-6)
    assert (mine["events"].to_numpy() == ref["events"].to_numpy()).all()

    borrowers = pd.Index(pd.concat([train, test, recal_fit, recal_set])["borrower"].unique())
    codes = {k: borrowers.get_indexer(f["borrower"]) for k, f in
             [("train", train), ("test", test), ("recal_fit", recal_fit), ("recal_set", recal_set)]}
    name_of = {id(train): "train", id(test): "test", id(recal_fit): "recal_fit", id(recal_set): "recal_set"}
    rng = np.random.default_rng(SEED)
    y, q = test["y"].to_numpy(float), test["qsort"].to_numpy()

    fixed, refit = [], []
    for b in range(n_boot):
        # one draw of borrowers, used for every frame: a borrower drawn k times counts k times in
        # training, recalibration and test alike
        cnt = np.bincount(rng.integers(0, len(borrowers), len(borrowers)), minlength=len(borrowers))
        w_of = lambda f: cnt[codes[name_of[id(f)]]].astype(float)  # noqa: E731
        # B: test borrowers resampled, predictions (and recalibration) fixed at the point estimate
        ws = w_of(test)
        fixed.append({"as fitted": metrics(y, p_hat, ws, q),
                      "R1: intercept updated": metrics(y, apply_recal(p_hat, recal["R1_a"], 1.0), ws, q),
                      "R2: intercept and slope updated": metrics(y, apply_recal(p_hat, recal["R2_a"], recal["R2_b"]), ws, q)})
        # C: everything refitted on the resample
        refit.append(one_pass(train, test, recal_fit, recal_set, w_of)[0])
        if (b + 1) % 100 == 0:
            print(f"bootstrap {b + 1}/{n_boot}", flush=True)

    rows = []
    for version in point:
        for stat, val in point[version].items():
            r = {"version": version, "stat": stat, "estimate": val}
            for tag, draws in [("fixed", fixed), ("refit", refit)]:
                v = np.array([d[version][stat] for d in draws], dtype=float)
                r[f"{tag}_lo"], r[f"{tag}_hi"] = np.nanpercentile(v, [2.5, 97.5])
            rows.append(r)
    out = pd.DataFrame(rows)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(MODEL_DIR / "calibration_bootstrap.csv", index=False)
    pd.DataFrame([{**recal, "n_boot": n_boot, "train_n": len(train), "test_n": len(test),
                   "recal_fit_q_max": recal_fit["qsort"].max(), "recal_set_n": len(recal_set),
                   "recal_set_events": int(recal_set["y"].sum()),
                   "train_event_rate": train["y"].mean(), "recal_set_event_rate": recal_set["y"].mean(),
                   "test_event_rate": test["y"].mean()}]).to_csv(MODEL_DIR / "calibration_recal_params.csv", index=False)
    print(out.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    main(ap.parse_args().boot)
