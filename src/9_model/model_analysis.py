"""Tables for the early-warning models (src/9_model): sample sizes, predictor summaries, event
rates by predictor bucket, the PIK-switch hypotheses, the mark path before a non-accrual entry,
and the model runs (in-sample coefficients, out-of-sample metrics, robustness)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from model_fit import BASELINE, COMBINED, HEADLINE, NA_HORIZONS, OUTCOMES, PREDICTORS, calibration_table, \
    coefficient_table, evaluate, mlp_allowed, model_frame, quarter_rank, ranking_metrics, review_capture, \
    rolling_predictions, rolling_regression_metrics
from model_panel import LOW_MARK, MARKDOWN

COMBINED_MULTI = ["price", "max_drawdown", "pik_switch", "dispersion"]   # robustness: 2+ lender families
BOOT_SPECS = {"headline", "price", "combined"}   # rolling: intervals only for the models the doc reports
EXTENSIONS = ["ridge", "lasso", "gbt", "mlp"]

# Event-rate buckets, in points of par (0.01 = one point)
BUCKETS = {
    "price": ([-np.inf, 0.80, 0.90, 0.95, 0.98, np.inf], ["below 80", "80-90", "90-95", "95-98", "98 and up"]),
    "max_drawdown": ([-np.inf, 0.0, 0.01, 0.03, 0.05, 0.10, np.inf], ["0", "0-1", "1-3", "3-5", "5-10", "over 10"]),
    "dispersion": ([-np.inf, 0.005, 0.01, 0.02, 0.05, np.inf], ["under 0.5", "0.5-1", "1-2", "2-5", "over 5"]),
}

ROBUSTNESS = [  # (variant, outcome, predictors, multi-lender borrowers only); the headline model varied
    ("headline: mark, drawdown, PIK switch", "y_na", HEADLINE, False),
    ("+ dispersion (std of family marks; the outline's four)", "y_na", COMBINED, False),
    ("+ dispersion as the range of family marks", "y_na", HEADLINE + ["dispersion_range", "has_dispersion"], False),
    ("PIK: change in PIK share of coupon (pp)", "y_na", ["price", "max_drawdown", "d_pik_pp"], False),
    ("market-adjusted mark and drawdown", "y_na", ["price_rel", "max_drawdown_adj", "pik_switch"], False),
    ("+ seniority mix (junior share of par)", "y_na", HEADLINE + ["junior_share"], False),
    ("+ borrower size (log BDC-held par)", "y_na", HEADLINE + ["log_par"], False),
    ("outcome: first-ever non-accrual", "y_na_first", HEADLINE, False),
    ("outcome: same-lender entries only", "y_na_same_lender", HEADLINE, False),
    ("borrowers with 2+ lender families, + dispersion", "y_na", COMBINED_MULTI, True),
    ("price_change: primary (same loans)", "price_change", HEADLINE, False),
    ("price_change: borrower FV/PAR difference (outline formula)", "price_change_raw", HEADLINE, False),
]


def specs() -> dict:
    """Each predictor alone, the headline model (mark, drawdown, PIK switch) and the outline's
    four combined."""
    return {**PREDICTORS, "headline": HEADLINE, "combined": COMBINED}


def _kind(outcome: str) -> str:
    return "binary" if outcome.startswith("y_na") else "continuous"


def _sample(outcome: str) -> str:
    return "na_risk_set" if outcome.startswith("y_na") else "val_sample"


def labelled_risk_set(m: pd.DataFrame, train_end: int) -> pd.DataFrame:
    return m[m["na_risk_set"] & m["y_na"].notna() & (m["qsort"] <= train_end)]


# ------------------------------------------------------------------ descriptive tables

def distinct_entries(rows: pd.DataFrame) -> int:
    """Distinct non-accrual entries (borrower x entry quarter) among positive rows. With h > 1 one
    entry is positive in up to h consecutive predictor quarters."""
    pos = rows[rows["y"] == 1] if "y" in rows else rows
    return len(set(zip(pos["borrower"], pos["qsort"] + pos["first_entry_k"])))


def sample_summary(m: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Observations, unique borrowers and distinct non-accrual entries per window
    {name: (first predictor qsort, last predictor qsort)}, for each outcome's sample (non-accrual
    within 1, 2 and 4 quarters, and the two valuation outcomes)."""
    out = []
    outcomes = list(NA_HORIZONS.values()) + [o for o in OUTCOMES if o != "y_na"]
    for name, (lo, hi) in windows.items():
        for outcome in outcomes:
            d = m[m[_sample(outcome)] & m[outcome].notna() & m["qsort"].between(lo, hi)]
            row = {"window": name, "outcome": outcome,
                   "predictor_quarters": f"{d['cal_q'].min()} to {d['cal_q'].max()}" if len(d) else "",
                   "observations": len(d), "borrowers": d["borrower"].nunique()}
            if outcome.startswith("y_na"):
                pos = d[d[outcome] == 1]
                row["positive_rows"] = len(pos)
                row["distinct_entries"] = distinct_entries(pos)
                row["borrowers_with_entry"] = pos["borrower"].nunique()
                row["positive_rate"] = d[outcome].mean() if len(d) else np.nan
                row["with_dispersion"] = int(d["has_dispersion"].sum())
            out.append(row)
    return pd.DataFrame(out)


def predictor_summary(m: pd.DataFrame, train_end: int) -> pd.DataFrame:
    """Each predictor on the labelled risk set: all borrowers, then by next-quarter outcome."""
    d = labelled_risk_set(m, train_end)
    cols = ["price", "max_drawdown", "pik_switch", "dispersion", "d_pik_pp", "dispersion_range", "val_ret",
            "price_change"]
    out = []
    for group, g in [("all", d), ("no entry at t+1", d[d["y_na"] == 0]), ("entry at t+1", d[d["y_na"] == 1])]:
        for c in cols:
            s = g[c]
            out.append({"predictor": c, "group": group, "n": len(s), "missing_share": s.isna().mean(),
                        "mean": s.mean(), "p10": s.quantile(0.1), "median": s.median(), "p90": s.quantile(0.9)})
    return pd.DataFrame(out)


def event_rates(m: pd.DataFrame, train_end: int) -> pd.DataFrame:
    """Next-quarter non-accrual entry rate by predictor bucket (labelled risk set)."""
    d = labelled_risk_set(m, train_end)
    parts = []
    for col, (bins, labels) in BUCKETS.items():
        b = pd.cut(d[col], bins=bins, labels=labels, right=True).astype(object)
        if col == "dispersion":
            b = b.where(d["has_dispersion"] == 1, "single lender")
        parts.append(d.assign(predictor=col, bucket=b))
    parts.append(d.assign(predictor="pik_switch", bucket=np.where(d["pik_switch"] == 1, "switched", "no switch")))
    order = {lab: i for col in BUCKETS for i, lab in enumerate(["single lender"] + BUCKETS[col][1])}
    order.update({"no switch": 0, "switched": 1})
    t = (pd.concat(parts).groupby(["predictor", "bucket"])["y_na"].agg(n="size", events="sum", rate="mean")
         .reset_index())
    t["order"] = t["bucket"].map(order)
    t["base_rate"] = d["y_na"].mean()
    return t.sort_values(["predictor", "order"]).drop(columns="order").reset_index(drop=True)


def _wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def pik_hypotheses(m: pd.DataFrame, train_end: int) -> pd.DataFrame:
    """Outline 4.3: is a PIK switch a stronger warning on a low mark (below LOW_MARK) or after a
    markdown of MARKDOWN or more in the quarter? Entry rates with Wilson 95% intervals."""
    d = labelled_risk_set(m, train_end)
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
        g = d.loc[mask, "y_na"]
        lo, hi = _wilson(g.sum(), len(g))
        out.append({"group": name, "n": len(g), "events": int(g.sum()), "rate": g.mean() if len(g) else np.nan,
                    "rate_ci_low": lo, "rate_ci_high": hi})
    return pd.DataFrame(out)


def event_profile(m: pd.DataFrame, train_end: int, horizon: int = 4) -> pd.DataFrame:
    """Mark path around a non-accrual entry: for borrowers entering at e = t + 1, their price and
    PIK switch at quarters e - horizon .. e (k = -horizon .. 0); the same for borrowers that stay
    accruing, at the matching quarters."""
    d = labelled_risk_set(m, train_end)
    full = m.set_index(["borrower", "qsort"])
    out = []
    for group, g in [("entered non-accrual", d[d["y_na"] == 1]), ("stayed accruing", d[d["y_na"] == 0])]:
        for k in range(-horizon, 1):
            idx = pd.MultiIndex.from_arrays([g["borrower"], g["qsort"] + 1 + k])
            px = full["price"].reindex(idx)
            sw = full["pik_switch"].reindex(idx)
            out.append({"group": group, "k": k, "borrowers": len(g), "n_priced": int(px.notna().sum()),
                        "median_price": px.median(), "p25_price": px.quantile(0.25), "p75_price": px.quantile(0.75),
                        "share_pik_switch": sw.mean()})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ model runs

def coefficients(m: pd.DataFrame, train_end: int) -> pd.DataFrame:
    """In-sample baselines (logit for non-accrual, OLS for the valuation outcomes) on predictor
    quarters <= train_end: each predictor alone, the headline model and the four combined;
    standard errors clustered by borrower. (Borrowers with 2+ lender families: robustness.)"""
    out = []
    for outcome, o in OUTCOMES.items():
        for spec, cols in specs().items():
            d = model_frame(m, outcome, cols)
            d = d[d["qsort"] <= train_end]
            out.append(coefficient_table(d, cols, o["kind"], {"outcome": outcome, "sample": "all",
                                                              "spec": spec, "model": BASELINE[o["kind"]]}))
    for h, outcome in NA_HORIZONS.items():      # longer horizons, all borrowers (every labelled quarter)
        if h == 1:
            continue
        for spec, cols in specs().items():
            d = model_frame(m, outcome, cols, sample="na_risk_set")
            out.append(coefficient_table(d, cols, "binary", {"outcome": outcome, "sample": "all", "spec": spec,
                                                             "model": "logit"}))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ main evaluation: rolling, by horizon

RULES = {"mark below 90": lambda d: d["price"] < 0.90, "mark below 95": lambda d: d["price"] < 0.95,
         "drawdown 5+ points": lambda d: d["max_drawdown"] >= 0.05,
         "mark below 90, drawdown 5+ or PIK switch":
             lambda d: (d["price"] < 0.90) | (d["max_drawdown"] >= 0.05) | (d["pik_switch"] == 1)}


def rolling_nonaccrual(m: pd.DataFrame) -> pd.DataFrame:
    """Rolling out-of-sample scores for non-accrual entry within h = 1, 2, 4 quarters: each predictor
    alone, the headline model and all four (logit), and all four with boosted trees, the most
    flexible benchmark. (The MLP is compared in the holdout; in the rolling folds it could run only
    at h = 4, the early folds having too few entries per weight.) One row per test borrower-quarter,
    horizon, spec and model, with the entry quarter. The tables use the headline model."""
    out = []
    for h, outcome in NA_HORIZONS.items():
        for spec, cols in specs().items():
            d = model_frame(m, outcome, cols, sample="na_risk_set")
            models = ["logit"] + (["gbt"] if spec == "combined" else [])
            p = rolling_predictions(d, cols, "binary", h, models)
            out.append(p.assign(h=h, spec=spec))
    p = pd.concat(out, ignore_index=True)
    return p.merge(m[["borrower", "qsort", "first_entry_k"]], on=["borrower", "qsort"], how="left")


def common_quarters(p: pd.DataFrame) -> list[int]:
    """Test quarters scored at every horizon (the fair window for comparing horizons)."""
    sets = [set(g["qsort"]) for _, g in p.groupby("h")]
    return sorted(set.intersection(*sets))


def rolling_tables(p: pd.DataFrame, m: pd.DataFrame) -> dict:
    """Tables of the rolling evaluation: metrics (every horizon, spec, model; full window and the
    common window), review-list capture, calibration, simple rules against the model at the same
    flag rate, and how early entrants are ranked (h = 1)."""
    common = common_quarters(p)
    metrics, capture = [], []
    for (h, spec, model), g in p.groupby(["h", "spec", "model"]):
        for window, w in [("full", g), ("common", g[g["qsort"].isin(common)])]:
            tag = {"h": h, "spec": spec, "model": model, "window": window}
            metrics.append({**tag, **ranking_metrics(w, boot=spec in BOOT_SPECS),
                            "distinct_entries": distinct_entries(w)})
            capture.append(review_capture(w).assign(**tag))
    metrics, capture = pd.DataFrame(metrics), pd.concat(capture, ignore_index=True)
    main = p[(p["spec"] == "headline") & (p["model"] == "logit")]
    calib = pd.concat([calibration_table(g).assign(h=h) for h, g in main.groupby("h")], ignore_index=True)
    ctx = main.merge(m[["borrower", "qsort", "price", "max_drawdown", "pik_switch"]], on=["borrower", "qsort"])
    rules = []
    for h, g in ctx.groupby("h"):
        r = quarter_rank(g)
        for name, rule in RULES.items():
            f = rule(g).to_numpy()
            fm = (r <= f.mean()).to_numpy()
            y = g["y"].to_numpy()
            rules.append({"h": h, "rule": name, "flag_rate": f.mean(), "rule_capture": y[f].sum() / y.sum(),
                          "rule_precision": y[f].mean(), "model_capture": y[fm].sum() / y.sum(),
                          "model_precision": y[fm].mean()})
    return {"metrics": metrics, "capture": capture, "calibration": calib, "rules": pd.DataFrame(rules),
            "earliness": earliness(main[main["h"] == 1]), "cross": horizon_cross_check(main, m, common),
            "common_quarters": common}


def horizon_cross_check(main: pd.DataFrame, m: pd.DataFrame, common: list[int], target: int = 4) -> pd.DataFrame:
    """The headline logit trained for each horizon, scored against one target (entry within
    `target` quarters) on the common test quarters: does the training horizon change who is at
    the top of the list?"""
    y = m.set_index(["borrower", "qsort"])[NA_HORIZONS[target]]
    out = []
    for h, g in main[main["qsort"].isin(common)].groupby("h"):
        g = g.assign(y=y.reindex(pd.MultiIndex.from_frame(g[["borrower", "qsort"]])).to_numpy())
        g = g[g["y"].notna()]
        r = quarter_rank(g)
        top = (r <= 0.05).to_numpy()
        out.append({"trained_for_h": h, "target_h": target, "n": len(g), "positives": int(g["y"].sum()),
                    **{k: v for k, v in ranking_metrics(g, boot=False).items() if k in ("roc_auc", "pr_auc")},
                    "top5pct_capture": g["y"].to_numpy()[top].sum() / g["y"].sum(),
                    "top5pct_precision": g["y"].to_numpy()[top].mean()})
    return pd.DataFrame(out)


def earliness(p1: pd.DataFrame, lags=(0, 1, 2, 3)) -> pd.DataFrame:
    """For the borrowers that enter non-accrual (h = 1 scores), their rank within the quarter at the
    predictor quarter and 1-3 quarters before it: how early a review list would have held them."""
    p1 = p1.assign(rank=quarter_rank(p1))
    ent = p1[p1["y"] == 1][["borrower", "qsort"]]
    rk = p1.set_index(["borrower", "qsort"])["rank"]
    out = []
    for lag in lags:
        r = rk.reindex(pd.MultiIndex.from_arrays([ent["borrower"], ent["qsort"] - lag])).dropna()
        out.append({"quarters_before_predictor_quarter": lag, "entries": len(ent), "scored": len(r),
                    "in_top_5pct": np.mean(r <= 0.05), "in_top_10pct": np.mean(r <= 0.10),
                    "in_top_20pct": np.mean(r <= 0.20), "median_rank": r.median()})
    return pd.DataFrame(out)


def rolling_valuation(m: pd.DataFrame) -> pd.DataFrame:
    """Rolling out-of-sample fit (OLS) for the two valuation outcomes, each predictor alone, the
    headline model and all four: pooled R2 against the training mean, and its range across quarters."""
    out = []
    for outcome in ("price_change", "val_ret"):
        for spec, cols in specs().items():
            p = rolling_predictions(model_frame(m, outcome, cols), cols, "continuous", 1, ["ols"])
            out.append({"outcome": outcome, "spec": spec, **rolling_regression_metrics(p)})
    return pd.DataFrame(out)


def out_of_sample(m: pd.DataFrame, split: str, train_end: int, test_q: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every predictor alone, the headline model and the four combined, with the baseline model;
    for non-accrual, the four combined also with the extensions (ridge, lasso, boosted trees,
    MLP). The valuation outcomes get no extensions here: the rolling test covers them over nine
    quarters. Returns (metrics, non-accrual test predictions of every model)."""
    rows, preds = [], []
    for outcome, o in OUTCOMES.items():
        kind = o["kind"]
        for spec, cols in specs().items():
            d = model_frame(m, outcome, cols, labelled=False)
            ext = kind == "binary" and spec == "combined"
            for name in [BASELINE[kind]] + (EXTENSIONS if ext else []):
                tag = {"split": split, "outcome": outcome, "sample": "all", "spec": spec}
                if name == "mlp" and not mlp_allowed(len(cols), int(d.loc[d["qsort"] <= train_end, "y"].sum())):
                    rows.append({**tag, "model": name, "skipped": "too few events per MLP weight"})
                    continue
                res, p = evaluate(kind, name, d, cols, train_end, test_q)
                rows.append({**tag, **res, "params": json.dumps(res["params"])})
                if outcome == "y_na":
                    preds.append(p.assign(split=split, spec=spec))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def robustness(m: pd.DataFrame, train_end: int, holdout_end: int, holdout_q: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alternative outcome and predictor definitions with the baseline model: in-sample
    coefficients on predictor quarters <= train_end, and the holdout metrics."""
    coefs, summary = [], []
    for variant, outcome, cols, multi in ROBUSTNESS:
        kind = _kind(outcome)
        d = model_frame(m, outcome, cols, sample=_sample(outcome), only_multi=multi, labelled=False)
        lab = d[(d["qsort"] <= train_end) & d["y"].notna()]
        c = coefficient_table(lab, cols, kind, {"variant": variant, "outcome": outcome})
        coefs.append(c)
        res, _ = evaluate(kind, BASELINE[kind], d, cols, holdout_end, holdout_q, boot=False)   # intervals unused here
        row = {"variant": variant, "outcome": outcome, "predictors": ", ".join(cols), "n_obs": len(lab),
               "n_borrowers": lab["borrower"].nunique()}
        if kind == "binary":
            row.update({"n_events": int(lab["y"].sum()), "pseudo_r2": c["pseudo_r2"].iloc[0],
                        "auc_in_sample": c["auc_in_sample"].iloc[0], "holdout_events": res.get("events"),
                        "holdout_roc_auc": res.get("roc_auc"), "holdout_pr_auc": res.get("pr_auc"),
                        "holdout_recall": res.get("recall"), "holdout_precision": res.get("precision")})
        else:
            row.update({"r2_in_sample": c["r2"].iloc[0], "holdout_r2": res.get("r2"),
                        "holdout_mae": res.get("mae"), "holdout_rmse": res.get("rmse")})
        summary.append(row)
    return pd.DataFrame(summary), pd.concat(coefs, ignore_index=True)
