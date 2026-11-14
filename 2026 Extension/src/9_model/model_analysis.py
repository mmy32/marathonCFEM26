"""Tables for the early-warning scorecard (src/9_model): the signs one by one (event rates by
bucket, the PIK-switch hypotheses, the mark path before an entry), the in-sample coefficients,
the sign selection, the out-of-sample test, robustness and the scorecard itself. One outcome
throughout: an accruing borrower is on non-accrual next quarter."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from model_fit import BASELINE, CANDIDATES, OUTCOME, SCORECARD, auc_ci, auc_gain, coefficient_table, \
    decile_table, equal_weight_score, model_frame, performance, rolling_predictions, score_decile, \
    split_predictions, wilson, _predictions
from model_panel import LOW_MARK, MARKDOWN

# A sign is kept if it adds at least half a point of ROC-AUC to the baseline, out of sample on the
# development quarters only. This threshold was set on 2026-11-13 after the development and test
# results had been seen (a rule of "any gain" would also have kept the one-year drawdown); it is
# fixed from here on, and the reserved quarter (2026Q3) is the test that no choice has seen.
MIN_GAIN = 0.005
SIGN_NAMES = {"pik_switch": "PIK switch", "dispersion": "valuation gap", "crowding": "crowding",
              "max_drawdown": "fall over the past year"}

# Event-rate buckets, in points of par (0.01 = one point)
BUCKETS = {
    "price": ([-np.inf, 0.80, 0.90, 0.95, 0.98, np.inf], ["below 80", "80-90", "90-95", "95-98", "98 and up"]),
    "d_price": ([-np.inf, -0.05, -0.01, 0.01, np.inf], ["fell 5+", "fell 1-5", "within 1", "rose 1+"]),
    "dispersion": ([-np.inf, 0.005, 0.01, 0.02, 0.05, np.inf], ["under 0.5", "0.5-1", "1-2", "2-5", "over 5"]),
    "crowding": ([-np.inf, 1, 2, 4, np.inf], ["1", "2", "3-4", "5+"]),
}

ROBUSTNESS = [  # (variant, outcome, predictors, multi-lender borrowers only); the scorecard varied
    ("scorecard: mark, change in mark, PIK switch", OUTCOME, SCORECARD, False),
    ("PIK: change in PIK share of coupon (pp)", OUTCOME, BASELINE + ["d_pik_pp"], False),
    ("market-adjusted mark and change", OUTCOME, ["price_rel", "d_price_adj", "pik_switch"], False),
    ("+ seniority mix (junior share of par)", OUTCOME, SCORECARD + ["junior_share"], False),
    ("+ borrower size (log BDC-held par)", OUTCOME, SCORECARD + ["log_par"], False),
    ("valuation gap as the range of family marks", OUTCOME, SCORECARD + ["dispersion_range", "has_dispersion"], False),
    ("borrowers with 2+ lender families, + valuation gap", OUTCOME, SCORECARD + ["dispersion"], True),
    ("outcome: impaired exits left unlabelled", "y_na_impaired_unlabelled", SCORECARD, False),
    ("outcome: exits left unlabelled", "y_na_no_exits", SCORECARD, False),
    ("outcome: first-ever non-accrual", "y_na_first", SCORECARD, False),
    ("outcome: same-lender entries only", "y_na_same_lender", SCORECARD, False),
]


def labelled_risk_set(m: pd.DataFrame, fit_end: int) -> pd.DataFrame:
    return m[m["na_risk_set"] & m[OUTCOME].notna() & (m["qsort"] <= fit_end)]


# ------------------------------------------------------------------ descriptive tables

def predictor_summary(m: pd.DataFrame, fit_end: int) -> pd.DataFrame:
    """Each sign on the labelled risk set: all borrowers, then by outcome."""
    d = labelled_risk_set(m, fit_end)
    cols = ["price", "d_price", "pik_switch", "dispersion", "crowding", "max_drawdown", "d_pik_pp", "dispersion_range"]
    out = []
    for group, g in [("all", d), ("no entry at t+1", d[d[OUTCOME] == 0]), ("entry at t+1", d[d[OUTCOME] == 1])]:
        for c in cols:
            s = g[c]
            out.append({"predictor": c, "group": group, "n": len(s), "missing_share": s.isna().mean(),
                        "mean": s.mean(), "p10": s.quantile(0.1), "median": s.median(), "p90": s.quantile(0.9)})
    return pd.DataFrame(out)


def event_rates(m: pd.DataFrame, fit_end: int) -> pd.DataFrame:
    """Entry rate (Wilson 95% interval) by bucket of each sign (labelled risk set)."""
    d = labelled_risk_set(m, fit_end)
    parts = []
    for col, (bins, labels) in BUCKETS.items():
        b = pd.cut(d[col], bins=bins, labels=labels, right=True).astype(object)
        if col == "dispersion":
            b = b.where(d["has_dispersion"] == 1, "single lender")
        parts.append(d.assign(predictor=col, bucket=b))
    parts.append(d.assign(predictor="pik_switch", bucket=np.where(d["pik_switch"] == 1, "switched", "no switch")))
    order = {lab: i for col in BUCKETS for i, lab in enumerate(["single lender"] + BUCKETS[col][1])}
    order.update({"no switch": 0, "switched": 1})
    t = (pd.concat(parts).groupby(["predictor", "bucket"])[OUTCOME].agg(n="size", events="sum", rate="mean")
         .reset_index())
    ci = [wilson(k, n) for k, n in zip(t["events"], t["n"])]
    t["rate_ci_low"], t["rate_ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    t["order"] = t["bucket"].map(order)
    t["base_rate"] = d[OUTCOME].mean()
    return t.sort_values(["predictor", "order"]).drop(columns="order").reset_index(drop=True)


def pik_hypotheses(m: pd.DataFrame, fit_end: int) -> pd.DataFrame:
    """Is a PIK switch a stronger warning on a low mark (below LOW_MARK) or after a markdown of
    MARKDOWN or more in the quarter? Entry rates with Wilson 95% intervals."""
    d = labelled_risk_set(m, fit_end)
    sw, low, md = d["pik_switch"] == 1, d["price"] < LOW_MARK, d["recent_markdown"] == 1
    groups = [("all borrowers", sw | ~sw), ("no PIK switch", ~sw), ("PIK switch", sw),
              (f"PIK switch, mark below {LOW_MARK:.0%}", sw & low),
              (f"PIK switch, mark {LOW_MARK:.0%} or above", sw & ~low),
              (f"PIK switch, marked down {MARKDOWN * 100:.0f}+ points this quarter", sw & md),
              (f"PIK switch, not marked down {MARKDOWN * 100:.0f}+ points", sw & ~md),
              ("PIK switch, low mark or markdown", sw & (low | md)),
              ("PIK switch, neither", sw & ~low & ~md),
              (f"no PIK switch, mark below {LOW_MARK:.0%}", ~sw & low),
              (f"no PIK switch, marked down {MARKDOWN * 100:.0f}+ points", ~sw & md)]
    out = []
    for name, mask in groups:
        g = d.loc[mask, OUTCOME]
        lo, hi = wilson(g.sum(), len(g))
        out.append({"group": name, "n": len(g), "events": int(g.sum()), "rate": g.mean() if len(g) else np.nan,
                    "rate_ci_low": lo, "rate_ci_high": hi})
    return pd.DataFrame(out)


def event_profile(m: pd.DataFrame, fit_end: int, horizon: int = 4) -> pd.DataFrame:
    """Mark path around a non-accrual entry: for borrowers entering at e = t + 1, their price and
    PIK switch at quarters e - horizon .. e (k = -horizon .. 0); the same for borrowers that stay
    accruing, at the matching quarters."""
    d = labelled_risk_set(m, fit_end)
    full = m.set_index(["borrower", "qsort"])
    out = []
    for group, g in [("had a credit event", d[d[OUTCOME] == 1]), ("stayed accruing", d[d[OUTCOME] == 0])]:
        for k in range(-horizon, 1):
            idx = pd.MultiIndex.from_arrays([g["borrower"], g["qsort"] + 1 + k])
            px = full["price"].reindex(idx)
            sw = full["pik_switch"].reindex(idx)
            out.append({"group": group, "k": k, "borrowers": len(g), "n_priced": int(px.notna().sum()),
                        "median_price": px.median(), "p25_price": px.quantile(0.25), "p75_price": px.quantile(0.75),
                        "share_pik_switch": sw.mean()})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ the model

def specs() -> dict:
    """The mark alone, the baseline (mark and its change), and the baseline with each other sign."""
    return {"mark alone": ["price"], "baseline": BASELINE,
            **{f"baseline + {SIGN_NAMES[k]}": BASELINE + cols for k, cols in CANDIDATES.items()}}


def coefficients(m: pd.DataFrame, fit_end: int) -> pd.DataFrame:
    """In-sample logits on predictor quarters <= fit_end, standard errors clustered by borrower.
    'baseline + PIK switch' is the scorecard."""
    out = []
    for spec, cols in specs().items():
        d = model_frame(m, cols)
        out.append(coefficient_table(d[d["qsort"] <= fit_end], cols, {"spec": spec, "scorecard": cols == SCORECARD}))
    return pd.concat(out, ignore_index=True)


def rolling(m: pd.DataFrame, last_q: int) -> pd.DataFrame:
    """Rolling out-of-sample scores of every spec, test quarters up to last_q: each quarter scored by
    a logit fitted only on the quarters before it."""
    out = []
    for spec, cols in specs().items():
        d = model_frame(m, cols)
        out.append(rolling_predictions(d[d["qsort"] <= last_q], cols).assign(spec=spec))
    return pd.concat(out, ignore_index=True)


def sign_selection(roll: pd.DataFrame, dev_end: int) -> pd.DataFrame:
    """Which signs to keep, decided on the development quarters only (rolling test quarters up to
    dev_end, so the test quarters play no part): ROC-AUC of the baseline, and of the baseline with
    each sign added alone, with the gain over the baseline (borrower-resampled 95% intervals). A
    sign is kept if it adds at least MIN_GAIN; whether the gain's interval clears zero is reported
    next to it."""
    r = roll[roll["qsort"] <= dev_end]
    base = r[r["spec"] == "baseline"]
    out = []
    for spec, cols in specs().items():
        p = r[r["spec"] == spec]
        auc, lo, hi = auc_ci(p)
        row = {"spec": spec, "test_quarters": f"{p['cal_q'].min()} to {p['cal_q'].max()}", "n": len(p),
               "events": int(p["y"].sum()), "roc_auc": auc, "roc_auc_lo": lo, "roc_auc_hi": hi}
        if spec != "baseline":
            row["gain_vs_baseline"], row["gain_lo"], row["gain_hi"] = auc_gain(p, base)
            if spec != "mark alone":
                row["kept"] = row["gain_vs_baseline"] >= MIN_GAIN
                row["gain_interval_above_zero"] = row["gain_lo"] > 0
        out.append(row)
    return pd.DataFrame(out)


def by_quarter(roll: pd.DataFrame) -> pd.DataFrame:
    """Events and ROC-AUC of each rolling test quarter: mark alone, the baseline and the scorecard,
    with the scorecard's ROC-AUC on the quarters it was fitted on (the overfitting check)."""
    names = {"mark alone": "mark alone", "baseline": "baseline", "baseline + PIK switch": "scorecard"}
    r = roll[roll["spec"].isin(names)].assign(spec=lambda x: x["spec"].map(names))
    t = r.groupby(["cal_q", "outcome_q", "spec"]).apply(lambda x: roc_auc_score(x["y"], x["pred"])).unstack("spec")
    n = r[r["spec"] == "scorecard"].groupby(["cal_q", "outcome_q"]).agg(
        n=("y", "size"), events=("y", "sum"), train_events=("train_events", "first"),
        scorecard_train_auc=("train_auc", "first"))
    return n.join(t[list(names.values())]).reset_index()


def oos_test(m: pd.DataFrame, train_end: int, test_qs: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The out-of-sample test: every model fitted once on quarters <= train_end and scored on
    `test_qs`. The scorecard against the mark alone, the baseline and the equal-weighted ranking
    of the scorecard's signs, plus the scorecard scored on non-accrual alone (impaired exits, which
    are defined by a low mark, left out of the outcome). train_roc_auc is each model's ROC-AUC on the quarters it was fitted
    on. Returns (one row per model, the decile tables of the scorecard and the equal-weighted
    ranking, the predictions)."""
    runs = [("mark alone", ["price"], "logit", OUTCOME), ("baseline: mark and its change", BASELINE, "logit", OUTCOME),
            ("scorecard: + PIK switch", SCORECARD, "logit", OUTCOME),
            ("equal-weighted ranking, same signs", SCORECARD, "equal", OUTCOME),
            ("scorecard, non-accrual only", SCORECARD, "logit", "y_na_impaired_unlabelled")]
    preds = {}
    for label, cols, name, outcome in runs:
        d = model_frame(m, cols, outcome, labelled=False)
        if name == "equal":
            d = d[d["qsort"].isin(test_qs)]
            p = _predictions(d, equal_weight_score(d, cols).to_numpy())
        else:
            p = split_predictions(d, cols, train_end, test_qs)
        preds[label] = p[p["y"].notna()]
    card = preds["scorecard: + PIK switch"]
    rows, deciles = [], []
    for label, cols, name, outcome in runs:
        p, prob = preds[label], name != "equal"
        row = {"model": label, "outcome": outcome, **performance(p, prob),
               "train_roc_auc": p["train_auc"].iloc[0] if prob else np.nan}
        if p is not card and outcome == OUTCOME:         # differences only on the same outcome
            row["auc_minus_scorecard"], row["diff_lo"], row["diff_hi"] = auc_gain(p, card)
        rows.append(row)
        if cols == SCORECARD and outcome == OUTCOME:
            deciles.append(decile_table(p, prob).assign(model=label))
    out = pd.concat([p.assign(model=k) for k, p in preds.items()], ignore_index=True)
    return pd.DataFrame(rows), pd.concat(deciles, ignore_index=True), out


def robustness(m: pd.DataFrame, fit_end: int, train_end: int, test_qs: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alternative sign and outcome definitions: in-sample coefficients on predictor quarters
    <= fit_end, and the ROC-AUC of the out-of-sample test."""
    m = m.assign(y_na_no_exits=m[OUTCOME].where(m["exit_type"].isna()))
    coefs, summary = [], []
    for variant, outcome, cols, multi in ROBUSTNESS:
        d = model_frame(m, cols, outcome, only_multi=multi)
        c = coefficient_table(d[d["qsort"] <= fit_end], cols, {"variant": variant, "outcome": outcome})
        coefs.append(c)
        p = split_predictions(d, cols, train_end, test_qs)
        summary.append({"variant": variant, "outcome": outcome, "predictors": ", ".join(cols),
                        "n_obs": c["n_obs"].iloc[0], "n_events": c["n_events"].iloc[0],
                        "auc_in_sample": c["auc_in_sample"].iloc[0], "test_n": len(p),
                        "test_events": int(p["y"].sum()), "test_roc_auc": roc_auc_score(p["y"], p["pred"])})
    return pd.DataFrame(summary), pd.concat(coefs, ignore_index=True)


def scorecard(m: pd.DataFrame, score_q: int, context: list[str]) -> pd.DataFrame:
    """The scorecard for quarter score_q: every accruing borrower's probability of entering
    non-accrual next quarter (SCORECARD logit fitted on the labelled quarters before score_q), its
    rank and its score decile (10 = riskiest tenth). The model counts a mark below 70 as 70, so
    the deepest borrowers can tie; ties are ordered by the lower mark. `y` is the outcome, empty
    until the next quarter is filed."""
    d = model_frame(m, SCORECARD, labelled=False)
    p = split_predictions(d, SCORECARD, score_q - 1, [score_q]).rename(columns={"pred": "probability"})
    p = p.merge(m[["borrower", "qsort"] + [c for c in context if c not in p]], on=["borrower", "qsort"], how="left")
    p = p.sort_values(["probability", "price"], ascending=[False, True]).reset_index(drop=True)
    p.insert(0, "rank", range(1, len(p) + 1))
    p["decile"] = score_decile(p.rename(columns={"probability": "pred"}))
    first = ["rank", "borrower", "cal_q", "outcome_q", "probability", "decile"]
    return p[first + [c for c in context if c in p] + ["y"]]
