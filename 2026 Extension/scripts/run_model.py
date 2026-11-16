#!/usr/bin/env python3
"""Early-warning scorecard: ranks borrowers by the chance of entering non-accrual within the next
two quarters. Builds the borrower-quarter panel (signs at t: the mark, its change over the quarter,
PIK switch, valuation gap, crowding; outcome: any BDC flags the borrower non-accrual at t + 1 or
t + 2; borrowers that leave every BDC's books before then have no label and their exits are counted
by type, and a mark never enters the outcome), chooses the signs on the development quarters, tests
the scorecard out of sample on signals from the second half of 2025, and applies it to the latest
quarter. Reads FINAL_DATASET.csv; writes
data/model/ (the panel and the out-of-sample predictions) and outputs/model/. About 6
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

import analysis as A  # noqa: E402
import plots as C  # noqa: E402
from common import q_next, q_sort, window_label  # noqa: E402
from config import HORIZON, RESERVED_Q, SCORE_Q, TEST_QS, TRAIN_END  # noqa: E402
from frames import model_frame  # noqa: E402
from evaluation import decile_table, performance, split_predictions  # noqa: E402
from panel import build_model_panel, event_counts, exit_table, sample_waterfall  # noqa: E402
from specs import BASELINE, CANDIDATES, SCORECARD, SIGN_NAMES  # noqa: E402

warnings.filterwarnings("ignore")

CONTEXT = ["price", "d_price", "pik_switch", "max_drawdown", "dispersion", "crowding", "fv_debt", "sector"]


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


def main(cached: bool = False):
    out = MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    m = load_panel(cached)
    train_end, test_qs, score_q = q_sort(TRAIN_END), [q_sort(q) for q in TEST_QS], q_sort(SCORE_Q)
    fit_end = score_q - HORIZON                     # the scorecard is fitted on every window closed by SCORE_Q
    first = m.loc[m["in_sample"], "cal_q"].min()
    window = f"{first}-{TEST_QS[-1]}"
    test_events = f"{q_next(TEST_QS[0])}-{q_next(TEST_QS[-1], HORIZON)}"

    print("Samples, events and exits ...")
    wf = sample_waterfall(m[m["qsort"] <= score_q])
    wf.to_csv(out / "sample_waterfall.csv", index=False)
    print(wf.to_string(index=False))
    ec = event_counts(m[m["qsort"] <= score_q])
    ec.to_csv(out / "event_counts.csv", index=False)
    print(ec.round(4).to_string(index=False))
    xt = exit_table(m[m["qsort"] <= score_q])
    xt.to_csv(out / "exits.csv", index=False)
    print(xt.round(3).to_string(index=False))

    print("The signs one by one ...")
    A.predictor_summary(m, fit_end).to_csv(out / "predictor_summary.csv", index=False)
    er = A.event_rates(m, fit_end)
    er.to_csv(out / "event_rates_by_bucket.csv", index=False)
    C.event_rate_by_predictor(er, out / "event_rate_by_predictor.png", window)
    ph = A.pik_hypotheses(m, fit_end)
    ph.to_csv(out / "pik_switch_hypotheses.csv", index=False)
    print(ph.round(4).to_string(index=False))
    ep = A.event_profile(m, fit_end)
    ep.to_csv(out / "event_profile.csv", index=False)
    C.event_profile(ep, out / "event_profile.png", f"{q_next(first)}-{SCORE_Q}")

    print("In-sample coefficients (clustered by borrower) ...")
    coefs = A.coefficients(m, fit_end)
    coefs.to_csv(out / "coefficients.csv", index=False)
    C.odds_ratios(coefs, out / "odds_ratios_nonaccrual.png", window)
    print(coefs[coefs["term"] != "const"][["spec", "term", "per", "coef", "se_cluster", "p_value", "odds_ratio",
                                           "n_obs", "n_events"]].round(4).to_string(index=False))

    print(f"Sign selection: rolling out of sample, signs up to {TRAIN_END} ...")
    roll = A.rolling(m, fit_end)
    MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    roll.to_csv(MODEL_DATA_DIR / "rolling_predictions.csv", index=False)
    sel = A.sign_selection(roll, train_end)
    sel.to_csv(out / "sign_selection.csv", index=False)
    print(sel.round(4).to_string(index=False))
    kept = [k for k, cols in CANDIDATES.items() if sel.set_index("spec").loc[f"baseline + {SIGN_NAMES[k]}", "kept"]]
    if set(SCORECARD) != set(BASELINE + sum((CANDIDATES[k] for k in kept), [])):
        raise SystemExit(f"The rule keeps {kept}; specs.SCORECARD is {SCORECARD}. Update SCORECARD.")
    bq = A.by_quarter(roll)
    bq.to_csv(out / "rolling_by_quarter.csv", index=False)
    C.rolling_auc_by_quarter(bq, out / "rolling_auc_by_quarter.png", TEST_QS[0])
    print(bq.round(3).to_string(index=False))

    print(f"Out-of-sample test: fitted on signs up to {TRAIN_END}, entries in {test_events} ...")
    perf, dec, preds = A.oos_test(m, train_end, test_qs)
    perf.to_csv(out / "oos_performance.csv", index=False)
    dec.to_csv(out / "oos_deciles.csv", index=False)
    preds.to_csv(MODEL_DATA_DIR / "oos_predictions.csv", index=False)
    C.score_deciles(dec, out / "oos_score_deciles.png", test_events, f"signs up to {TRAIN_END}")
    print(perf.round(4).to_string(index=False))
    print(dec.round(4).to_string(index=False))

    print("Robustness ...")
    rs, rc = A.robustness(m, fit_end, train_end, test_qs)
    rs.to_csv(out / "robustness_summary.csv", index=False)
    rc.to_csv(out / "robustness_coefficients.csv", index=False)
    print(rs.round(4).to_string(index=False))

    card = A.scorecard(m, score_q, CONTEXT)
    card.to_csv(out / f"scorecard_{SCORE_Q}.csv", index=False)
    print(f"Scorecard: {len(card):,} borrowers accruing in {SCORE_Q}, {card['probability'].sum():.0f} entries expected "
          f"in {window_label([score_q])[0]}")
    print(card.head(15).drop(columns=["cal_q", "outcome_q", "y"]).round(3).to_string(index=False))
    if m["qsort"].max() >= q_sort(RESERVED_Q):      # the reserved quarter is in the data: the blind test
        blind_q = q_sort(RESERVED_Q) - HORIZON
        d = model_frame(m, SCORECARD)
        p = split_predictions(d, SCORECARD, blind_q - HORIZON, [blind_q])
        print(f"Reserved test: signals in {q_next(RESERVED_Q, -HORIZON)}, entries in {window_label([blind_q])[0]} ...")
        res = pd.DataFrame([{"model": "scorecard", **performance(p)}])
        res.to_csv(out / f"reserved_test_{RESERVED_Q}.csv", index=False)
        decile_table(p).to_csv(out / f"reserved_test_{RESERVED_Q}_deciles.csv", index=False)
        print(res.round(4).to_string(index=False))
    print(f"Done: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cached", action="store_true", help="reuse data/model/model_panel.csv")
    main(ap.parse_args().cached)
