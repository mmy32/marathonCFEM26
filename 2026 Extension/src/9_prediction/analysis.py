"""Tables for the early-warning scorecard (src/9_prediction): the signs one by one (event rates by
bucket, the PIK-switch hypotheses, the mark path before an event), the in-sample coefficients,
the sign selection, the out-of-sample test, robustness and the scorecard itself. One outcome
throughout (config.OUTCOME): an accruing borrower has a credit event within the next two quarters,
i.e. non-accrual or a lender's concession (PIK switch, debt-for-equity, write-down). Any other
outcome appears only as a labelled robustness row."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import credit_events as CE
from config import HORIZON, LOW_MARK, MARKDOWN, MIN_GAIN, OUTCOME
from models.logit import coefficient_table, table_probabilities
from frames import design, model_frame
from evaluation import _boot_counts, auc_ci, auc_gain, capture_gap, decile_table, performance, \
    pr_auc_ci, pr_auc_gain, quarter_rank, ratio_ci, rolling_predictions, score_decile, split_predictions
from features import BUCKETS, BUCKETS_LEFT_CLOSED, PREDICTOR_CAPS
from specs import BASELINE, NO_CAPS, ROBUSTNESS, SCORECARD, SIGN_NAMES, CANDIDATES, specs


def labelled_risk_set(m: pd.DataFrame, fit_end: int) -> pd.DataFrame:
    return m[m["na_risk_set"] & m[OUTCOME].notna() & (m["qsort"] <= fit_end)]


def predictor_summary(m: pd.DataFrame, fit_end: int) -> pd.DataFrame:
    """Each sign on the labelled risk set: all borrowers, then by outcome."""
    d = labelled_risk_set(m, fit_end)
    cols = ["price", "fall_1q", "pik_switch", "pik_past_year", "dispersion", "crowding", "max_drawdown", "d_pik_pp",
            "dispersion_std"]
    out = []
    for group, g in [("all", d), ("no event in t+1..t+2", d[d[OUTCOME] == 0]), ("event in t+1..t+2", d[d[OUTCOME] == 1])]:
        for c in cols:
            s = g[c]
            out.append({"predictor": c, "group": group, "n": len(s), "missing_share": s.isna().mean(),
                        "mean": s.mean(), "p10": s.quantile(0.1), "median": s.median(), "p90": s.quantile(0.9)})
    return pd.DataFrame(out)


def event_rates(m: pd.DataFrame, fit_end: int) -> pd.DataFrame:
    """Event rate (borrower-resampled 95% interval) by bucket of each sign (labelled risk set)."""
    d = labelled_risk_set(m, fit_end)
    parts = []
    for col, (bins, labels) in BUCKETS.items():
        b = pd.cut(d[col], bins=bins, labels=labels, right=col not in BUCKETS_LEFT_CLOSED).astype(object)
        if col == "dispersion":
            b = b.where(d["has_dispersion"] == 1, "single lender")
        parts.append(d.assign(predictor=col, bucket=b))
    order = {(col, lab): i for col in BUCKETS for i, lab in enumerate(["single lender"] + BUCKETS[col][1])}
    for col in ["pik_switch", "pik_past_year"]:                     # the 0/1 signs
        parts.append(d.assign(predictor=col, bucket=np.where(d[col] == 1, "switched", "no switch")))
        order.update({(col, "no switch"): 0, (col, "switched"): 1})
    groups = pd.concat(parts).groupby(["predictor", "bucket"])
    t = groups[OUTCOME].agg(n="size", events="sum", rate="mean").reset_index()
    ci = {k: ratio_ci(g[OUTCOME], np.ones(len(g)), g["borrower"]) for k, g in groups}
    keys = list(zip(t["predictor"], t["bucket"]))
    t["rate_ci_low"], t["rate_ci_high"] = [ci[k][0] for k in keys], [ci[k][1] for k in keys]
    t["order"] = [order[k] for k in keys]
    t["base_rate"] = d[OUTCOME].mean()
    return t.sort_values(["predictor", "order"]).drop(columns="order").reset_index(drop=True)


def pik_hypotheses(m: pd.DataFrame, fit_end: int) -> pd.DataFrame:
    """Is a PIK switch a stronger warning on a low mark (below LOW_MARK) or after a markdown of
    MARKDOWN or more in the quarter? Event rates with borrower-resampled 95% intervals."""
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
        lo, hi = ratio_ci(g, np.ones(len(g)), d.loc[mask, "borrower"])
        out.append({"group": name, "n": len(g), "events": int(g.sum()), "rate": g.mean() if len(g) else np.nan,
                    "rate_ci_low": lo, "rate_ci_high": hi})
    return pd.DataFrame(out)


PROFILE_GROUPS = ["entered non-accrual", "concession, no non-accrual", "no credit event"]


def event_profile(m: pd.DataFrame, fit_end: int, horizon: int = 4) -> pd.DataFrame:
    """Mark path around a credit event: for borrowers with an event at e (t + 1 or t + 2, the first
    quarter of the event), their mark at quarters e - horizon .. e (k = -horizon .. 0), split into
    those flagged non-accrual in the window and those with a concession only; for borrowers
    without an event, the same around e = t + 1."""
    d = labelled_risk_set(m, fit_end)
    full = m.set_index(["borrower", "qsort"])
    ev, flag = d[OUTCOME] == 1, d["flag_in_window"].astype(bool)
    out = []
    for group, g in zip(PROFILE_GROUPS, [d[ev & flag], d[ev & ~flag], d[~ev]]):
        for k in range(-horizon, 1):
            e = g["event_q"].fillna(g["qsort"] + 1).astype(int)
            px = full["price"].reindex(pd.MultiIndex.from_arrays([g["borrower"], e + k]))
            out.append({"group": group, "k": k, "borrowers": len(g), "n_priced": int(px.notna().sum()),
                        "median_price": px.median(), "p25_price": px.quantile(0.25), "p75_price": px.quantile(0.75)})
    return pd.DataFrame(out)


def coefficients(m: pd.DataFrame, fit_end: int) -> pd.DataFrame:
    """In-sample logits of every spec on predictor quarters <= fit_end, standard errors clustered
    by borrower, with the ROC-AUC and PR-AUC on the rows fitted on. For the scorecard, both also get
    a borrower-resampled 95% interval (the fit held fixed)."""
    out = []
    for spec, cols in specs().items():
        d = model_frame(m, cols)
        d = d[d["qsort"] <= fit_end]
        c = coefficient_table(d, cols, {"spec": spec, "scorecard": cols == SCORECARD})
        if cols == SCORECARD:
            p = d[["borrower", "y"]].assign(pred=table_probabilities(d, cols, c))
            _, c["auc_in_sample_lo"], c["auc_in_sample_hi"] = auc_ci(p)
            _, c["pr_auc_in_sample_lo"], c["pr_auc_in_sample_hi"] = pr_auc_ci(p)
        out.append(c)
    return pd.concat(out, ignore_index=True)


def collinearity(m: pd.DataFrame, fit_end: int) -> pd.DataFrame:
    """Multicollinearity of the scorecard's signs on the rows it is fitted on, capped as in the fit:
    Pearson correlations (one column per sign), Spearman's largest off the diagonal, and each
    sign's R² on the other three with VIF = 1 / (1 - R²), the factor by which the overlap
    inflates its coefficient's variance (1 = none; 5-10 is the usual warning line)."""
    d = model_frame(m, SCORECARD)
    X = pd.DataFrame(design(d[d["qsort"] <= fit_end], SCORECARD), columns=SCORECARD)
    r = X.corr()
    vif = np.diag(np.linalg.inv(r.to_numpy()))
    rho = X.corr("spearman").to_numpy()
    np.fill_diagonal(rho, 0.0)
    out = r.rename_axis("predictor").reset_index()
    out["spearman_max_abs"] = np.abs(rho).max(axis=1)
    out["r2_on_others"] = 1 - 1 / vif
    out["vif"] = vif
    out["n_obs"] = len(X)
    return out


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
    next to it. PR-AUC and its gain over the baseline are reported beside, not used by the rule."""
    r = roll[roll["qsort"] <= dev_end]
    base = r[r["spec"] == "baseline"]
    out = []
    for spec, cols in specs().items():
        p = r[r["spec"] == spec]
        auc, lo, hi = auc_ci(p)
        ap, ap_lo, ap_hi = pr_auc_ci(p)
        row = {"spec": spec, "test_quarters": f"{p['cal_q'].min()} to {p['cal_q'].max()}", "n": len(p),
               "events": int(p["y"].sum()), "roc_auc": auc, "roc_auc_lo": lo, "roc_auc_hi": hi}
        if spec != "baseline":
            row["gain_vs_baseline"], row["gain_lo"], row["gain_hi"] = auc_gain(p, base)
            if spec.startswith("baseline + "):
                row["kept"] = row["gain_vs_baseline"] >= MIN_GAIN
                row["gain_interval_above_zero"] = row["gain_lo"] > 0
        row.update({"pr_auc": ap, "pr_auc_lo": ap_lo, "pr_auc_hi": ap_hi})
        if spec != "baseline":
            row["pr_gain_vs_baseline"], row["pr_gain_lo"], row["pr_gain_hi"] = pr_auc_gain(p, base)
        out.append(row)
    return pd.DataFrame(out)


def by_quarter(roll: pd.DataFrame) -> pd.DataFrame:
    """Events and ROC-AUC of each rolling test quarter: mark alone, the baseline and the scorecard,
    with the scorecard's ROC-AUC on the quarters it was fitted on (the overfitting check), and its
    calibration in that quarter: realized / predicted events with a borrower-resampled 95%
    interval."""
    shown = ["mark alone", "baseline", "scorecard"]
    r = roll[roll["spec"].isin(shown)]
    t = r.groupby(["cal_q", "outcome_q", "spec"]).apply(lambda x: roc_auc_score(x["y"], x["pred"])).unstack("spec")
    card = r[r["spec"] == "scorecard"]
    n = card.groupby(["cal_q", "outcome_q"]).agg(
        n=("y", "size"), events=("y", "sum"), train_events=("train_events", "first"),
        scorecard_train_auc=("train_auc", "first"), expected=("pred", "sum"))
    ci = {k: ratio_ci(g["y"], g["pred"], g["borrower"]) for k, g in card.groupby(["cal_q", "outcome_q"])}
    n["scorecard_realized_to_predicted"] = n["events"] / n.pop("expected")
    n["ratio_lo"], n["ratio_hi"] = [ci[k][0] for k in n.index], [ci[k][1] for k in n.index]
    return n.join(t[shown]).reset_index()


def oos_test(m: pd.DataFrame, train_end: int, test_qs: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The out-of-sample test: every model fitted once on quarters <= train_end and scored on
    `test_qs`. The scorecard against the mark alone (the benchmark) and the mark with its change.
    train_roc_auc is each model's ROC-AUC on the quarters it was fitted on. Returns (one row per
    model, the decile tables of the scorecard and the mark alone, the predictions)."""
    runs = [("mark alone", ["price"]), ("baseline: mark and its change", BASELINE), ("scorecard", SCORECARD)]
    preds = {}
    for label, cols in runs:
        p = split_predictions(model_frame(m, cols, labelled=False), cols, train_end, test_qs)
        preds[label] = p[p["y"].notna()]
    card = preds["scorecard"]
    rows, deciles = [], []
    for label, cols in runs:
        p = preds[label]
        row = {"model": label, "outcome": OUTCOME, **performance(p), "train_roc_auc": p["train_auc"].iloc[0]}
        if label != "scorecard":
            row["auc_minus_scorecard"], row["diff_lo"], row["diff_hi"] = auc_gain(p, card)
        rows.append(row)
        if label in (BENCHMARK, "scorecard"):
            deciles.append(decile_table(p).assign(model=label))
    out = pd.concat([p.assign(model=k) for k, p in preds.items()], ignore_index=True)
    return pd.DataFrame(rows), pd.concat(deciles, ignore_index=True), out


# The benchmark the scorecard is compared with on the same rows (column "mark" of oos_windows): the
# mark alone, a logit on the mark level fitted and scored exactly as the scorecard is.
BENCHMARK = "mark alone"
BENCHMARKS = {BENCHMARK: "mark"}


def oos_windows(m: pd.DataFrame, roll: pd.DataFrame, train_end: int, test_preds: pd.DataFrame) -> dict:
    """The two out-of-sample windows, one row per labelled borrower-quarter with the scorecard's
    score (pred) and the mark alone's (mark): the development quarters (each scored by a fit on the
    windows closed by then, as in sign_selection) and the test (`test_preds` from oos_test)."""
    keys = ["borrower", "qsort"]
    out = {}
    for w, card, mark in [("development", roll[(roll["spec"] == "scorecard") & (roll["qsort"] <= train_end)],
                           roll[(roll["spec"] == BENCHMARK) & (roll["qsort"] <= train_end)]),
                          ("test", test_preds[test_preds["model"] == "scorecard"],
                           test_preds[test_preds["model"] == BENCHMARK])]:
        out[w] = (card[keys + ["cal_q", "y", "pred"]].merge(mark[keys + ["pred"]].rename(columns={"pred": "mark"}),
                                                           on=keys, validate="one_to_one")
                  .merge(m[keys + ["y_na"]], on=keys, validate="one_to_one"))
    return out


def _quarters(p: pd.DataFrame) -> dict:
    return {"quarters": f"{p['cal_q'].min()} to {p['cal_q'].max()}", "n_quarters": p["qsort"].nunique()}


def capture_vs_mark(windows: dict) -> pd.DataFrame:
    """The scorecard's watchlists against the mark alone's: events caught in the riskiest 1%, 2%, ...,
    100% of each quarter's borrowers, and the scorecard's extra share of the events with a 95%
    interval (evaluation.capture_gap), in each window of oos_windows."""
    return pd.concat([capture_gap(p, p.assign(pred=p["mark"])).assign(window=w, **_quarters(p))
                      for w, p in windows.items()], ignore_index=True)


def benchmark_comparison(windows: dict, shares: tuple = (0.10, 0.20)) -> pd.DataFrame:
    """The scorecard and each benchmark, in each window: ROC-AUC, PR-AUC, events caught on the
    riskiest 10% and 20% lists, and the scorecard's gain over the benchmark on each (ROC-AUC, PR-AUC,
    and the extra share of events caught on each list), with borrower-resampled 95% intervals; the
    gains' intervals are paired (both scored on the same resamples)."""
    rows = []
    for w, p in windows.items():
        for name, col in {"scorecard": "pred", **BENCHMARKS}.items():
            b = p.assign(pred=p[col])
            auc, lo, hi = auc_ci(b)
            ap, ap_lo, ap_hi = pr_auc_ci(b)
            row = {"window": w, **_quarters(p), "model": name, "n": len(b), "events": int(b["y"].sum()),
                   "roc_auc": auc, "roc_auc_lo": lo, "roc_auc_hi": hi, "pr_auc": ap, "pr_auc_lo": ap_lo,
                   "pr_auc_hi": ap_hi}
            if name != "scorecard":
                row["card_auc_gain"], row["card_auc_gain_lo"], row["card_auc_gain_hi"] = auc_gain(p, b)
                row["card_pr_gain"], row["card_pr_gain_lo"], row["card_pr_gain_hi"] = pr_auc_gain(p, b)
            cg = capture_gap(p, b, np.array(shares)).set_index("share")
            for s in shares:
                k = f"{round(s * 100)}pct"
                row[f"caught_{k}"] = int(cg.loc[s, "caught_base"])
                if name != "scorecard":
                    row[f"card_gain_{k}"], row[f"card_gain_{k}_lo"], row[f"card_gain_{k}_hi"] = \
                        cg.loc[s, ["gap_share", "gap_lo", "gap_hi"]]
            rows.append(row)
    return pd.DataFrame(rows)


DISAGREEMENT_GROUPS = ["scorecard only", "mark alone only", "both", "neither"]


def list_disagreement(windows: dict, shares: tuple = (0.10, 0.20)) -> pd.DataFrame:
    """Who each watchlist adds: on the riskiest `share` of each quarter's borrowers, by the scorecard
    and by the mark alone, every borrower-quarter is in one group (on both lists, on the
    scorecard's only, on the mark's only, on neither). Per group: borrowers, credit events, the event
    rate with a 95% interval, events with a non-accrual. One more row per list: the event rate of
    the scorecard-only picks minus that of the mark-only picks, with a paired interval (both rates
    on the same borrower resamples)."""
    rows = []
    for w, p in windows.items():
        y, na = p["y"].to_numpy(float), (p["y_na"] == 1).to_numpy(float)
        codes, uniq = pd.factorize(p["borrower"])
        for s in shares:
            a = (quarter_rank(p) <= s).to_numpy()
            b = (quarter_rank(p.assign(pred=p["mark"])) <= s).to_numpy()
            groups = dict(zip(DISAGREEMENT_GROUPS, [a & ~b, ~a & b, a & b, ~a & ~b]))
            for g, k in groups.items():
                lo, hi = ratio_ci(y[k], np.ones(k.sum()), p["borrower"].to_numpy()[k])
                rows.append({"window": w, **_quarters(p), "list": s, "group": g, "n": int(k.sum()),
                             "events": int(y[k].sum()), "event_rate": y[k].mean() if k.any() else np.nan,
                             "rate_lo": lo, "rate_hi": hi,
                             "events_with_non_accrual": int((y * na)[k].sum())})
            cols = np.column_stack([y * groups["scorecard only"], groups["scorecard only"],
                                    y * groups["mark alone only"], groups["mark alone only"]])
            G = np.zeros((len(uniq), 4))
            np.add.at(G, codes, cols)
            diff = lambda t: t[0] / t[1] - t[2] / t[3] if t[1] > 0 and t[3] > 0 else np.nan
            vals = [diff(c @ G) for c in _boot_counts(len(uniq))]
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            rows.append({"window": w, **_quarters(p), "list": s, "group": "difference: scorecard only - mark alone only",
                         "event_rate": diff(G.sum(0)), "rate_lo": lo, "rate_hi": hi})
    return pd.DataFrame(rows)


def ranks_by_outcome(m: pd.DataFrame, train_end: int, test_qs: list[int]) -> pd.DataFrame:
    """Where the scorecard of the out-of-sample test ranked every accruing borrower of the test
    quarters, labelled or not, by what happened next: a credit event (split into those flagged
    non-accrual and those with a concession only), no event, left every BDC's books (by exit
    bucket, primary cutoff; refinanced is empty by construction) or status unknown. Deciles are
    taken within the quarter over all of them. roc_auc_vs_no_event ranks each group against the
    borrowers with no event: how well the one scorecard finds each kind of event, and whether it
    also flags the exits it was not fitted on (partly by construction for the impaired: an exit
    marked well below par is impaired). labelled_share on the last row is the test's coverage."""
    d = model_frame(m, SCORECARD, labelled=False)
    p = split_predictions(d, SCORECARD, train_end, test_qs)
    p["decile"] = score_decile(p)
    p = p.merge(m[["borrower", "qsort", "exited", "exit_class", "window_complete", "flag_in_window"]],
                on=["borrower", "qsort"])
    p = p[p["window_complete"].astype(bool)]
    exited, flag = p["exited"].astype(bool), p["flag_in_window"].astype(bool)
    left = ("left the books: " + p["exit_class"].astype(str)).to_numpy(dtype=object)
    p["what_happened"] = np.select([(p["y"] == 1) & flag, p["y"] == 1, p["y"] == 0, exited],
                                   ["credit event: non-accrual", "credit event: concession only", "no credit event", left],
                                   "status unknown")
    order = (["credit event: non-accrual", "credit event: concession only", "no credit event"]
             + [f"left the books: {k}" for k in CE.EXIT_CLASSES] + ["status unknown"])
    g = p.assign(top10=p["decile"] == 10, top20=p["decile"] >= 9).groupby("what_happened")
    t = g.agg(borrower_quarters=("pred", "size"), in_riskiest_tenth=("top10", "sum"),
              in_riskiest_fifth=("top20", "sum"), mean_probability=("pred", "mean")).reindex(order)
    none = p[p["what_happened"] == "no credit event"]
    t["roc_auc_vs_no_event"] = np.nan
    for k in order:
        x = p[p["what_happened"] == k]
        if k != "no credit event" and len(x) and len(none):
            t.loc[k, "roc_auc_vs_no_event"] = roc_auc_score(np.r_[np.ones(len(x)), np.zeros(len(none))],
                                                            np.r_[x["pred"], none["pred"]])
    t.loc["all"] = [len(p), (p["decile"] == 10).sum(), (p["decile"] >= 9).sum(), p["pred"].mean(), np.nan]
    counts = ["borrower_quarters", "in_riskiest_tenth", "in_riskiest_fifth"]
    t[counts] = t[counts].fillna(0).astype(int)
    t.insert(2, "share_in_riskiest_tenth", t["in_riskiest_tenth"] / t["borrower_quarters"])
    t["share_in_riskiest_fifth"] = t["in_riskiest_fifth"] / t["borrower_quarters"]
    t.loc["all", "labelled_share"] = p["y"].notna().mean()
    return t.rename_axis("what_happened").reset_index()


def robustness(m: pd.DataFrame, fit_end: int, train_end: int, test_qs: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alternative sign and outcome definitions: in-sample coefficients on predictor quarters
    <= fit_end, and on the out-of-sample test the ROC-AUC, PR-AUC, realized / predicted events
    (overall and in the riskiest tenth) and the highest predicted probability."""
    coefs, summary = [], []
    for variant, outcome, cols, multi in ROBUSTNESS:
        caps = {} if variant == NO_CAPS else PREDICTOR_CAPS
        d = model_frame(m, cols, outcome, only_multi=multi)
        c = coefficient_table(d[d["qsort"] <= fit_end], cols, {"variant": variant, "outcome": outcome}, caps)
        coefs.append(c)
        p = split_predictions(d, cols, train_end, test_qs, caps)
        top = p[score_decile(p) == 10]
        summary.append({"variant": variant, "outcome": outcome, "predictors": ", ".join(cols),
                        "n_obs": c["n_obs"].iloc[0], "n_events": c["n_events"].iloc[0],
                        "auc_in_sample": c["auc_in_sample"].iloc[0], "test_n": len(p),
                        "test_events": int(p["y"].sum()), "test_roc_auc": roc_auc_score(p["y"], p["pred"]),
                        "test_pr_auc": average_precision_score(p["y"], p["pred"]),
                        "test_realized_to_predicted": p["y"].sum() / p["pred"].sum(),
                        "test_top_decile_realized_to_predicted": top["y"].sum() / top["pred"].sum(),
                        "test_max_probability": p["pred"].max()})
    return pd.DataFrame(summary), pd.concat(coefs, ignore_index=True)


def _capped(d: pd.DataFrame) -> pd.Series:
    """True where a cap changes at least one of the scorecard's predictors."""
    hit = pd.Series(False, index=d.index)
    for c, (lo, hi) in PREDICTOR_CAPS.items():
        if c in SCORECARD:
            hit |= (d[c] < lo if lo is not None else False) | (d[c] > hi if hi is not None else False)
    return hit


def caps_comparison(m: pd.DataFrame, train_end: int, test_qs: list[int]) -> pd.DataFrame:
    """The scorecard on the out-of-sample test with and without the caps, side by side: ranking
    (ROC-AUC, PR-AUC, share of events in the riskiest tenth), calibration (realized / predicted
    events overall, by score decile group, and on the borrower-quarters a cap changes), the
    highest predicted probability, and the false negatives: events outside the riskiest 5, 10, 20
    and 30% of each quarter's borrowers, on the test and on the development quarters (rolling,
    as in the sign selection). Borrower-resampled 95% intervals; the ROC-AUC row last is the
    uncapped fit minus the capped one, on the same resamples."""
    d = model_frame(m, SCORECARD)
    d = d.assign(capped=_capped(d))
    fits = {"capped": PREDICTOR_CAPS, "uncapped": {}}
    preds = {}
    for k, caps in fits.items():
        p = split_predictions(d, SCORECARD, train_end, test_qs, caps)
        preds[k] = p.merge(d[["borrower", "qsort", "capped"]], on=["borrower", "qsort"]).assign(
            decile=score_decile)

    def ratio(p, k):
        s = p[k]
        return (s["y"].sum() / s["pred"].sum(), *ratio_ci(s["y"], s["pred"], s["borrower"]))

    cp = preds["capped"]
    n_cap, e_cap = int(cp["capped"].sum()), int(cp.loc[cp["capped"], "y"].sum())
    metrics = {
        "ROC-AUC": lambda p: auc_ci(p),
        "PR-AUC": lambda p: tuple(performance(p)[c] for c in ["pr_auc", "pr_auc_lo", "pr_auc_hi"]),
        "share of events in the riskiest tenth": lambda p: (
            p.loc[p["decile"] == 10, "y"].sum() / p["y"].sum(),
            *ratio_ci(p["y"] * (p["decile"] == 10), p["y"], p["borrower"])),
        "realized / predicted: all": lambda p: ratio(p, p["y"].notna()),
        "realized / predicted: riskiest tenth": lambda p: ratio(p, p["decile"] == 10),
        "realized / predicted: decile 9": lambda p: ratio(p, p["decile"] == 9),
        "realized / predicted: deciles 1-8": lambda p: ratio(p, p["decile"] <= 8),
        f"realized / predicted: rows a cap changes ({n_cap}, {e_cap} events)": lambda p: ratio(p, p["capped"]),
        "mean predicted probability: rows a cap changes": lambda p: (p.loc[p["capped"], "pred"].mean(), np.nan, np.nan),
        "realized event rate: rows a cap changes": lambda p: (
            p.loc[p["capped"], "y"].mean(), *ratio_ci(p.loc[p["capped"], "y"], np.ones(n_cap), p.loc[p["capped"], "borrower"])),
        "highest predicted probability": lambda p: (p["pred"].max(), np.nan, np.nan),
        "borrower-quarters predicted at 50% or more": lambda p: ((p["pred"] >= 0.5).sum(), np.nan, np.nan),
    }
    rows = []
    for name, f in metrics.items():
        row = {"metric": name}
        for k, p in preds.items():
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = f(p)
        rows.append(row)
    dev = {k: rolling_predictions(d[d["qsort"] <= train_end], SCORECARD, caps=caps) for k, caps in fits.items()}
    for label, ps in [("test", preds), ("development", dev)]:
        for share in [0.05, 0.10, 0.20, 0.30]:
            row = {"metric": f"{label}: events missed outside the riskiest {share:.0%} "
                             f"(of {int(ps['capped']['y'].sum())})"}
            for k, p in ps.items():
                row[k] = int(((quarter_rank(p) > share) & (p["y"] == 1)).sum())
            rows.append(row)
    g, lo, hi = auc_gain(preds["uncapped"], cp)
    rows.append({"metric": "ROC-AUC, uncapped minus capped", "uncapped": g, "uncapped_lo": lo, "uncapped_hi": hi})
    return pd.DataFrame(rows)


def scorecard(m: pd.DataFrame, score_q: int, context: list[str]) -> pd.DataFrame:
    """The scorecard for quarter score_q: every accruing borrower's probability of a credit
    event within the next two quarters (SCORECARD logit fitted on the labelled quarters up
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
