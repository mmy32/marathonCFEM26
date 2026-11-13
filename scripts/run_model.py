#!/usr/bin/env python3
"""Early-warning models (BDC Credit Risk Capstone outline, 2026-11-12). Builds the borrower-quarter
panel (warning signs at t: mark level, rolling max drawdown, PIK switch, and valuation dispersion,
tested but not in the headline model; outcomes: new non-accrual within 1, 2 and 4 quarters, price
change and fair-value return at t + 1), fits the headline logit (mark, drawdown, PIK switch) and its
comparisons, and runs the out-of-sample tests: the main rolling evaluation (every quarter scored by
a model fitted on earlier, fully observed quarters; review-list capture) and the outline's single
chronological split. Reads FINAL_DATASET.csv; writes
data/model/model_panel.csv, data/model/rolling_predictions.csv and outputs/model/. About 8
minutes; `--cached` reuses the saved panel (skips the 2-minute panel build). See
docs/14_early_warning_model.md."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "1_data_processing"))
from paths import FINAL_DIR, MODEL_DATA_DIR, MODEL_DIR, add_code_dirs_to_path  # noqa: E402
add_code_dirs_to_path()

import model_analysis as A  # noqa: E402
import model_charts as C  # noqa: E402
from model_fit import HEADLINE, NA_HORIZONS, fit_predict, model_frame  # noqa: E402
from model_panel import build_model_panel, q_label, q_next, q_sort, sample_waterfall  # noqa: E402

warnings.filterwarnings("ignore")

TRAIN_END = "2026Q1"     # outline section 6: training predictor quarters up to 2026Q1 (outcomes up to 2026Q2)
TEST_Q = "2026Q2"        # outline test: predictor quarter 2026Q2, outcome 2026Q3
HOLDOUT_END = "2025Q4"   # the same design one quarter earlier, scored while 2026Q3 is not in the data
HOLDOUT_Q = "2026Q1"
CONTEXT = ["price", "max_drawdown", "pik_switch", "dispersion", "d_price", "n_families", "fv_debt", "sector"]
REVIEW_SHARE = 0.05      # the forecast's review list: the riskiest 5% of accruing borrowers


def forecast(m: pd.DataFrame) -> pd.DataFrame:
    """Scores for the borrowers accruing in the latest quarter, whose outcomes are not filed yet:
    probability of entering non-accrual within 1, 2 and 4 quarters (the headline model: mark,
    drawdown and PIK switch, logistic, fitted on every quarter whose outcome window is fully
    observed), rank by the 4-quarter score, and the top-REVIEW_SHARE review list. The model counts
    a mark below 70 as 70 and a drawdown above 20 points as 20, so the deepest borrowers can tie;
    ties are ordered by the lower mark."""
    q = m["qsort"].max()
    f = None
    for h, outcome in NA_HORIZONS.items():
        d = model_frame(m, outcome, HEADLINE, sample="na_risk_set", labelled=False)
        train, test = d[d["y"].notna() & (d["qsort"] <= q - h)], d[d["qsort"] == q]
        pred, _, _ = fit_predict("binary", "logit", {}, train, test, HEADLINE)
        col = test[["borrower", "cal_q"]].assign(**{f"p_h{h}": pred})
        f = col if f is None else f.merge(col, on=["borrower", "cal_q"], how="outer")
    h = max(NA_HORIZONS)
    f = f.merge(m[["borrower", "cal_q"] + CONTEXT], on=["borrower", "cal_q"], how="left")
    f = f.sort_values([f"p_h{h}", "price"], ascending=[False, True]).reset_index(drop=True)
    f.insert(5, f"rank_h{h}", range(1, len(f) + 1))
    f.insert(6, "review_list", f[f"rank_h{h}"] <= round(REVIEW_SHARE * len(f)))
    return f


def write_rolling(m: pd.DataFrame, out: Path):
    """Main evaluation: rolling out of sample for non-accrual within 1, 2, 4 quarters and for the
    valuation outcomes."""
    p = A.rolling_nonaccrual(m)
    MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    p.to_csv(MODEL_DATA_DIR / "rolling_predictions.csv", index=False)
    T = A.rolling_tables(p, m)
    common = q_label(T["common_quarters"])
    T["metrics"].to_csv(out / "rolling_metrics.csv", index=False)
    T["capture"].to_csv(out / "review_list_capture.csv", index=False)
    T["calibration"].to_csv(out / "rolling_calibration.csv", index=False)
    T["rules"].to_csv(out / "rolling_rules_vs_model.csv", index=False)
    T["earliness"].to_csv(out / "early_warning.csv", index=False)
    T["cross"].to_csv(out / "horizon_cross_check.csv", index=False)
    C.horizon_review_lists(p, T["common_quarters"], f"{common[0]}-{common[-1]}", out / "horizon_review_lists.png")
    C.rolling_auc_by_quarter(p, out / "rolling_auc_by_quarter.png")
    mt = T["metrics"]
    print(mt[mt["spec"].isin(["headline", "combined", "price"])][
        ["h", "spec", "model", "window", "test_quarters", "n", "positives", "distinct_entries", "base_rate", "roc_auc",
         "roc_auc_lo", "roc_auc_hi", "pr_auc", "pr_lift"]].round(3).to_string(index=False))
    cap = T["capture"]
    print(cap[(cap["spec"] == "headline") & (cap["model"] == "logit")][
        ["h", "window", "share_flagged", "flagged_per_quarter", "capture", "precision", "lift"]].round(3)
          .to_string(index=False))
    for k in ("rules", "earliness", "cross"):
        print(T[k].round(3).to_string(index=False))
    v = A.rolling_valuation(m)
    v.to_csv(out / "rolling_valuation.csv", index=False)
    print(v.round(4).to_string(index=False))


def load_panel(cached: bool) -> pd.DataFrame:
    path = MODEL_DATA_DIR / "model_panel.csv"
    if cached:
        print(f"Reading {path} ...")
        # round_trip: pandas' default parser misreads the last bit of some floats, which moved
        # borrowers sitting exactly on a bucket edge; this one reads back exactly what was written
        return pd.read_csv(path, float_precision="round_trip")
    from monitor_panel import build_monitor_panel
    print("Building the tranche-quarter panel ...")
    tr, rows = build_monitor_panel(FINAL_DIR / "FINAL_DATASET.csv")
    print("Building the borrower-quarter model panel ...")
    m = build_model_panel(tr, rows)
    MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    m.to_csv(path, index=False)
    print(f"  {len(m):,} borrower-quarters -> {path}")
    return m


def review_holdout(preds: pd.DataFrame, m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide holdout predictions (the headline logit, then the four-signal models), and the cases to
    discuss: every entry the headline model missed at its threshold (false negatives) and the 20
    highest-scored non-entrants."""
    p = preds[preds["spec"] == "combined"].assign(flag=lambda x: x["flag"].astype(int))
    wide = p.pivot_table(index=["borrower", "cal_q", "outcome_q", "y"], columns="model", values=["pred", "flag"],
                         aggfunc="first")
    wide.columns = [f"{'p' if a == 'pred' else 'flag'}_all4_{b}" for a, b in wide.columns]
    head = preds[(preds["spec"] == "headline") & (preds["model"] == "logit")][["borrower", "cal_q", "pred", "flag"]]
    wide = (head.rename(columns={"pred": "p_headline", "flag": "flag_headline"})
            .merge(wide.reset_index(), on=["borrower", "cal_q"], how="left")
            .merge(m[["borrower", "cal_q"] + CONTEXT], on=["borrower", "cal_q"], how="left")
            .sort_values("p_headline", ascending=False))
    cols = ["borrower", "cal_q", "outcome_q", "y", "p_headline", "flag_headline"]
    wide = wide[cols + [c for c in wide.columns if c not in cols]]
    missed = wide[(wide["y"] == 1) & ~wide["flag_headline"].astype(bool)].assign(case="missed entry (false negative)")
    fps = wide[wide["y"] == 0].head(20).assign(case="highest-scored non-entrant (false positive)")
    return wide, pd.concat([missed, fps])


def main(cached: bool = False):
    out = MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    m = load_panel(cached)
    first_q = m.loc[m["in_sample"], "qsort"].min()
    train_end, test_q = q_sort(TRAIN_END), q_sort(TEST_Q)
    hold_end, hold_q = q_sort(HOLDOUT_END), q_sort(HOLDOUT_Q)
    window = f"{m.loc[m['qsort'] == first_q, 'cal_q'].iloc[0]}-{TRAIN_END}"
    entries = f"{m.loc[m['qsort'] == first_q, 'outcome_q'].iloc[0]}-{m.loc[m['qsort'] == train_end, 'outcome_q'].iloc[0]}"

    print("Samples ...")
    wf = sample_waterfall(m)
    wf.to_csv(out / "sample_waterfall.csv", index=False)
    print(wf.to_string(index=False))
    ss = A.sample_summary(m, {"training (outline)": (first_q, train_end), "holdout: training": (first_q, hold_end),
                              "holdout: test": (hold_q, hold_q), "test (outline)": (test_q, test_q)})
    ss.to_csv(out / "sample_summary.csv", index=False)
    print(ss.to_string(index=False))

    print("Descriptives ...")
    A.predictor_summary(m, train_end).to_csv(out / "predictor_summary.csv", index=False)
    er = A.event_rates(m, train_end)
    er.to_csv(out / "event_rates_by_bucket.csv", index=False)
    C.event_rate_by_predictor(er, out / "event_rate_by_predictor.png", window)
    ph = A.pik_hypotheses(m, train_end)
    ph.to_csv(out / "pik_switch_hypotheses.csv", index=False)
    print(ph.round(4).to_string(index=False))
    ep = A.event_profile(m, train_end)
    ep.to_csv(out / "event_profile.csv", index=False)
    C.event_profile(ep, out / "event_profile.png", entries)

    print("In-sample coefficients (clustered by borrower) ...")
    coefs = A.coefficients(m, train_end)
    coefs.to_csv(out / "coefficients.csv", index=False)
    C.odds_ratios(coefs, out / "odds_ratios_nonaccrual.png", window)
    show = coefs[coefs["spec"].isin(["headline", "combined"]) & (coefs["sample"] == "all") & (coefs["term"] != "const")]
    print(show[["outcome", "spec", "term", "per", "coef", "se_cluster", "p_value", "odds_ratio", "n_obs",
                "n_events"]].round(4).to_string(index=False))

    print("Main evaluation: rolling out of sample, non-accrual within 1, 2 and 4 quarters ...")
    write_rolling(m, out)

    print(f"Outline split (single holdout): train to {HOLDOUT_END}, test {HOLDOUT_Q} -> {q_next(HOLDOUT_Q)} ...")
    metrics, preds = A.out_of_sample(m, "holdout", hold_end, hold_q)
    labelled = m.loc[(m["qsort"] == test_q) & m["na_risk_set"], "y_na"].notna().any()
    if labelled:
        print(f"Test: train to {TRAIN_END}, test {TEST_Q} -> {q_next(TEST_Q)} ...")
        mt, pt = A.out_of_sample(m, "test", train_end, test_q)
        metrics, preds = pd.concat([metrics, mt]), pd.concat([preds, pt])
    metrics.to_csv(out / "oos_metrics.csv", index=False)
    cols = ["split", "outcome", "sample", "spec", "model", "n_test", "events", "roc_auc", "roc_auc_lo", "roc_auc_hi",
            "pr_auc", "recall", "precision", "flag_rate", "r2", "r2_vs_train_mean", "mae", "rmse"]
    print(metrics[[c for c in cols if c in metrics]].round(4).to_string(index=False))
    hp = preds[preds["split"] == "holdout"]
    wide, review = review_holdout(hp, m)
    wide.to_csv(out / "holdout_predictions.csv", index=False)
    review.to_csv(out / "holdout_review.csv", index=False)
    C.holdout_curves(hp, out / "holdout_roc_pr.png", q_next(HOLDOUT_Q))
    C.holdout_auc(metrics[metrics["split"] == "holdout"], out / "holdout_auc.png", q_next(HOLDOUT_Q))

    f = forecast(m)
    latest = f["cal_q"].iloc[0]
    print(f"Forecast: {len(f):,} borrowers accruing in {latest}, {int(f['review_list'].sum())} on the review list ...")
    f.to_csv(out / f"forecast_{latest}.csv", index=False)
    print(f.head(15)[["borrower", "p_h1", "p_h2", "p_h4", "price", "max_drawdown", "pik_switch", "dispersion"]].round(3)
          .to_string(index=False))

    print("Robustness ...")
    rs, rc = A.robustness(m, train_end, hold_end, hold_q)
    rs.to_csv(out / "robustness_summary.csv", index=False)
    rc.to_csv(out / "robustness_coefficients.csv", index=False)
    print(rs.round(4).to_string(index=False))
    print(f"Done: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cached", action="store_true", help="reuse data/model/model_panel.csv")
    main(ap.parse_args().cached)
