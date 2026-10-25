"""Evaluation layer: which signals predict which outcomes, at what horizon.

For each signal and horizon (outcome Y_h = a stress event on the same loan in t+1..t+h):

- Event rate by signal bucket (quintiles of the within-quarter percentile; 0/1 for flags),
  with the sample size and events in each bucket and the lift over the overall rate.
- AUC of the signal alone (risk-increasing direction; below 0.5 means the opposite).
- AUC of the baseline B and of B + signal (logistic regressions), the gain, and a 90%
  interval from a bootstrap that resamples whole quarters (predictions fixed, not refit).
- Out of sample: fit on windows ending by 2025Q4, score the windows inside 2026 (the
  1H26 stress episode).

Baseline B (Risk Dimensions doc, plus the guideline's size confounder): price percentile (M1),
two-quarter price change percentile (M2), PIK at origination (P3, with an unknown flag), lien
group, log of the borrower's BDC-held par. Each signal enters as a within-quarter percentile.
Each signal is evaluated on the rows where it is defined (dispersion: tranches held by 2+
families), with the baseline refit on the same rows.

Rows are loan-quarters per BDC, so the same loan held by several BDCs appears several times;
the quarter bootstrap does not account for that, so intervals are somewhat too narrow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

# signal column -> (label, +1 if a higher value means more risk, -1 if lower, kind)
SIGNALS = {
    "price":             ("M1 price (baseline)",                    -1, "pct"),
    "M2_price_chg_2q":   ("M2 two-quarter price change (baseline)", -1, "pct"),
    "pik_share":         ("P1 PIK share of coupon",                 +1, "pct"),
    "P1_acq":            ("P1 acquired PIK share",                  +1, "pct"),
    "P2_recent":         ("P2 PIK toggle at t or t-1",              +1, "flag"),
    "V1_std":            ("V1 dispersion (std of family marks)",    +1, "pct"),
    "V2_low_mark":       ("V2 low mark minus consensus",            -1, "pct"),
    "mark_vs_consensus": ("Lender's mark minus consensus",          +1, "pct"),
    "mark_distance":     ("Lender's distance from consensus",       +1, "pct"),
    "C1_families":       ("C1 lender families",                     +1, "pct"),
    "C2_lender_hhi":     ("C2 lender HHI",                          +1, "pct"),
    "C4_d_families":     ("C4 change in families over 4 quarters",  +1, "pct"),
}
BASE_COLS = ["b_M1", "b_M2", "b_M2_missing", "b_P3", "b_P3_unknown", "log_borrower_par"]
OOS_START = "2026Q1"     # windows inside 2026 are scored out of sample


def auc(score, y) -> float:
    """Mann-Whitney AUC; NaN if one class is missing."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = float(y.sum()), float((~y).sum())     # float: int32 sums overflow n1 * n0 on Windows
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def within_quarter_pct(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("cal_q")[col].rank(pct=True) * 100


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    """Add risk-oriented within-quarter percentiles of each signal and the baseline columns."""
    p = panel.copy()
    for col, (_, sign, kind) in SIGNALS.items():
        p["s_" + col] = sign * (p[col] if kind == "flag" else within_quarter_pct(p, col))
    p["b_M1"] = within_quarter_pct(p, "price").fillna(50)
    p["b_M2_missing"] = p["M2_price_chg_2q"].isna().astype(float)
    p["b_M2"] = within_quarter_pct(p, "M2_price_chg_2q").fillna(50)
    p["b_P3_unknown"] = p["P3_orig_pik"].isna().astype(float)
    p["b_P3"] = p["P3_orig_pik"].fillna(0)
    p["log_borrower_par"] = p["log_borrower_par"].fillna(p["log_borrower_par"].median())
    lien = pd.get_dummies(p["lien_group"], prefix="lien", dtype=float)
    return pd.concat([p, lien.drop(columns=[c for c in lien.columns if c.endswith("_1L")])], axis=1)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols, y):
    m = LogisticRegression(max_iter=500)
    mu, sd = train[cols].mean(), train[cols].std().replace(0, 1)
    m.fit((train[cols] - mu) / sd, train[y])
    return m.predict_proba((test[cols] - mu) / sd)[:, 1]


def _boot_gain(d: pd.DataFrame, y: str, p0: np.ndarray, p1: np.ndarray, reps: int, rng) -> tuple[float, float]:
    qs = d["cal_q"].to_numpy()
    uq = np.unique(qs)
    idx_by_q = {q: np.flatnonzero(qs == q) for q in uq}
    yy = d[y].to_numpy().astype(bool)
    gains = []
    for _ in range(reps):
        idx = np.concatenate([idx_by_q[q] for q in rng.choice(uq, size=len(uq), replace=True)])
        gains.append(auc(p1[idx], yy[idx]) - auc(p0[idx], yy[idx]))
    return float(np.nanpercentile(gains, 5)), float(np.nanpercentile(gains, 95))


def evaluate(panel: pd.DataFrame, outcome: str = "Y_mk", horizons=(1, 2, 4), reps: int = 200,
             seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary: one row per signal x horizon, buckets: event rate per signal bucket)."""
    p = prepare(panel)
    lien_cols = [c for c in p.columns if c.startswith("lien_") and c != "lien_group"]
    base = BASE_COLS + lien_cols
    rng = np.random.default_rng(seed)
    rows, buckets = [], []
    for h in horizons:
        y = f"{outcome}_h{h}"
        for col, (label, _, kind) in SIGNALS.items():
            s = "s_" + col
            d = p[p[y].notna() & p[s].notna()].copy()
            if d[y].sum() < 20:
                continue
            d[y] = d[y].astype(int)
            rate = d[y].mean()
            # buckets
            if kind == "flag":
                d["bucket"] = d[col].astype(int).astype(str)
            else:
                d["bucket"] = pd.qcut(d[s].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
            b = d.groupby("bucket")[y].agg(n="size", events="sum", rate="mean").reset_index()
            b["lift"] = b["rate"] / rate
            buckets.append(b.assign(signal=label, horizon=h))
            # incremental AUC in sample
            is_base = col in ("price", "M2_price_chg_2q")
            p0 = _fit_predict(d, d, base, y)
            p1 = p0 if is_base else _fit_predict(d, d, base + [s], y)
            a0, a1 = auc(p0, d[y]), auc(p1, d[y])
            lo, hi = (np.nan, np.nan) if is_base else _boot_gain(d, y, p0, p1, reps, rng)
            # out of sample: test on windows (t+1..t+h) that start in 2026, train on windows
            # that end by 2025Q4; windows straddling the two are left out
            oos_mask = d["cal_q"].map(lambda q: _shift_q(q, 1) >= OOS_START)
            train_mask = d["cal_q"].map(lambda q: _shift_q(q, h) < OOS_START)
            tr_, te_ = d[train_mask], d[oos_mask]
            o0 = o1 = np.nan
            if len(te_) and te_[y].nunique() == 2 and tr_[y].nunique() == 2:
                o0 = auc(_fit_predict(tr_, te_, base, y), te_[y])
                o1 = o0 if is_base else auc(_fit_predict(tr_, te_, base + [s], y), te_[y])
            rows.append(dict(signal=label, column=col, horizon=h, n=len(d), events=int(d[y].sum()),
                             event_rate=rate, auc_signal=auc(d[s], d[y]), auc_base=a0, auc_base_plus=a1,
                             auc_gain=a1 - a0, gain_lo90=lo, gain_hi90=hi,
                             oos_n=len(te_), oos_auc_base=o0, oos_auc_base_plus=o1,
                             oos_gain=(o1 - o0) if o0 == o0 else np.nan))
    return pd.DataFrame(rows), pd.concat(buckets, ignore_index=True)


def _shift_q(q: str, k: int) -> str:
    n = int(q[:4]) * 4 + int(q[-1]) - 1 + k
    return f"{n // 4}Q{n % 4 + 1}"
