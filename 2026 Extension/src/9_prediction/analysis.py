"""Tables for the early-warning model (src/9_prediction): the signs one by one (event rates by bucket, the
PIK-switch hypotheses, the mark path before an entry), the in-sample coefficients, the sign
selection, the out-of-sample test, robustness, the scorecard itself and the one-input baselines for
each outcome. Classification outcome: an accruing borrower enters non-accrual within the next two
quarters; markdown outcomes: the next-quarter change in the mark and val_ret."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import HORIZON, LOW_MARK, MARKDOWN, MIN_GAIN, OUTCOME
from models.logit import coefficient_table
from models.ranking import equal_weight_score
from models.linear import fit_linear
from frames import model_frame
from evaluation import _predictions, auc_ci, auc_gain, decile_table, performance, regression_scores, \
    rolling_predictions, score_decile, split_predictions, wilson
from features import BUCKETS, UNITS
from specs import BASELINE, INPUTS, MARKDOWN_OUTCOMES, ROBUSTNESS, SCORECARD, SIGN_NAMES, CANDIDATES, specs


def labelled_risk_set(m: pd.DataFrame, fit_end: int) -> pd.DataFrame:
    return m[m["na_risk_set"] & m[OUTCOME].notna() & (m["qsort"] <= fit_end)]


def predictor_summary(m: pd.DataFrame, fit_end: int) -> pd.DataFrame:
    """Each sign on the labelled risk set: all borrowers, then by outcome."""
    d = labelled_risk_set(m, fit_end)
    cols = ["price", "d_price", "pik_switch", "dispersion", "crowding", "max_drawdown", "d_pik_pp", "dispersion_range"]
    out = []
    for group, g in [("all", d), ("no entry in t+1..t+2", d[d[OUTCOME] == 0]), ("entry in t+1..t+2", d[d[OUTCOME] == 1])]:
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
    """Mark path around a non-accrual entry: for borrowers entering at e (t + 1 or t + 2, the first
    flagged quarter), their price and PIK switch at quarters e - horizon .. e (k = -horizon .. 0);
    for borrowers that stay accruing, the same around e = t + 1."""
    d = labelled_risk_set(m, fit_end)
    full = m.set_index(["borrower", "qsort"])
    out = []
    for group, g in [("entered non-accrual", d[d[OUTCOME] == 1]), ("stayed accruing", d[d[OUTCOME] == 0])]:
        for k in range(-horizon, 1):
            e = g["entry_q"].fillna(g["qsort"] + 1).astype(int)
            idx = pd.MultiIndex.from_arrays([g["borrower"], e + k])
            px = full["price"].reindex(idx)
            sw = full["pik_switch"].reindex(idx)
            out.append({"group": group, "k": k, "borrowers": len(g), "n_priced": int(px.notna().sum()),
                        "median_price": px.median(), "p25_price": px.quantile(0.25), "p75_price": px.quantile(0.75),
                        "share_pik_switch": sw.mean()})
    return pd.DataFrame(out)


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
            if spec.startswith("baseline + "):
                row["kept"] = row["gain_vs_baseline"] >= MIN_GAIN
                row["gain_interval_above_zero"] = row["gain_lo"] > 0
        out.append(row)
    return pd.DataFrame(out)


def by_quarter(roll: pd.DataFrame) -> pd.DataFrame:
    """Events and ROC-AUC of each rolling test quarter: mark alone, the baseline and the scorecard,
    with the scorecard's ROC-AUC on the quarters it was fitted on (the overfitting check)."""
    names = {"mark alone": "mark alone", "baseline": "baseline", "scorecard": "scorecard"}
    r = roll[roll["spec"].isin(names)].assign(spec=lambda x: x["spec"].map(names))
    t = r.groupby(["cal_q", "outcome_q", "spec"]).apply(lambda x: roc_auc_score(x["y"], x["pred"])).unstack("spec")
    n = r[r["spec"] == "scorecard"].groupby(["cal_q", "outcome_q"]).agg(
        n=("y", "size"), events=("y", "sum"), train_events=("train_events", "first"),
        scorecard_train_auc=("train_auc", "first"))
    return n.join(t[list(names.values())]).reset_index()


def oos_test(m: pd.DataFrame, train_end: int, test_qs: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The out-of-sample test: every model fitted once on quarters <= train_end and scored on
    `test_qs`. The scorecard against the mark alone, the baseline and the equal-weighted ranking
    of the scorecard's signs. train_roc_auc is each model's ROC-AUC on the quarters it was fitted
    on. Returns (one row per model, the decile tables of the scorecard and the equal-weighted
    ranking, the predictions)."""
    runs = [("mark alone", ["price"], "logit", OUTCOME), ("baseline: mark and its change", BASELINE, "logit", OUTCOME),
            ("scorecard", SCORECARD, "logit", OUTCOME),
            ("equal-weighted ranking, same signs", SCORECARD, "equal", OUTCOME)]
    preds = {}
    for label, cols, name, outcome in runs:
        d = model_frame(m, cols, outcome, labelled=False)
        if name == "equal":
            d = d[d["qsort"].isin(test_qs)]
            p = _predictions(d, equal_weight_score(d, cols).to_numpy())
        else:
            p = split_predictions(d, cols, train_end, test_qs)
        preds[label] = p[p["y"].notna()]
    card = preds["scorecard"]
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
    non-accrual within the next two quarters (SCORECARD logit fitted on the labelled quarters up
    to score_q - HORIZON, whose windows have closed), its
    rank and its score decile (10 = riskiest tenth). The model counts a mark below 70 as 70, so
    the deepest borrowers can tie; ties are ordered by the lower mark. `y` is the outcome, empty
    until both quarters of its window are filed."""
    d = model_frame(m, SCORECARD, labelled=False)
    p = split_predictions(d, SCORECARD, score_q - HORIZON, [score_q]).rename(columns={"pred": "probability"})
    p = p.merge(m[["borrower", "qsort"] + [c for c in context if c not in p]], on=["borrower", "qsort"], how="left")
    p = p.sort_values(["probability", "price"], ascending=[False, True]).reset_index(drop=True)
    p.insert(0, "rank", range(1, len(p) + 1))
    p["decile"] = score_decile(p.rename(columns={"probability": "pred"}))
    first = ["rank", "borrower", "cal_q", "outcome_q", "probability", "decile"]
    return p[first + [c for c in context if c in p] + ["y"]]


def simple_models(m: pd.DataFrame, train_end: int, test_qs: list[int]) -> pd.DataFrame:
    """One simple model per input per outcome, fitted on signals up to train_end and scored on
    test_qs, as in the scorecard's test. Non-accrual within two quarters: logistic regression,
    ROC-AUC. Markdown over the next quarter: linear regression, in- and out-of-sample R2 and the
    out-of-sample RMSE against the training mean. `coef` is per `per` units of the input (the first
    column of the input); the logit's score is roc_auc [95% interval]."""
    out = []
    for name, cols in INPUTS.items():
        d = model_frame(m, cols)
        p = split_predictions(d, cols, train_end, test_qs)
        auc, lo, hi = auc_ci(p)
        out.append({"input": name, "outcome": "non-accrual within 2 quarters (logit)",
                    "n_train": int((d["qsort"] <= train_end).sum()), "n_test": len(p),
                    "roc_auc": auc, "roc_auc_lo": lo, "roc_auc_hi": hi})
        for label, y in MARKDOWN_OUTCOMES.items():
            d = model_frame(m, cols, outcome=y)
            tr, te = d[d["qsort"] <= train_end], d[d["qsort"].isin(test_qs)]
            pred_te, pred_tr, coef = fit_linear(tr, te, cols)
            test = regression_scores(te["y"], pred_te, tr["y"].mean())
            out.append({"input": name, "outcome": label + " (linear)", "n_train": len(tr), "n_test": len(te),
                        "coef": coef, "per": UNITS.get(cols[0], 1.0),
                        "r2_in_sample": regression_scores(tr["y"], pred_tr, tr["y"].mean())["r2"],
                        "r2_out_of_sample": test["r2"], "rmse_out_of_sample": test["rmse"],
                        "rmse_train_mean": test["rmse_benchmark"]})
    return pd.DataFrame(out)
