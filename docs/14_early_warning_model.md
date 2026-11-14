# Early-Warning Scorecard: Which Borrowers Have a Credit Event Next Quarter

## In 30 seconds

Lenders mark a loan down before they stop booking its interest. The scorecard is a logistic
regression on three signs from each quarter's filings:

| Warning sign | Odds of a credit event next quarter |
|---|---|
| Its loans are marked low | each 10 points lower: about **4.5x** |
| The mark fell this quarter | each 5-point fall: about **1.4x** |
| It just switched to paying interest in kind (PIK) | about **3x** |

A **credit event** is non-accrual, or leaving every BDC's books marked below 80 (an impaired
exit, the nearest thing in the data to a realized loss). A markdown is an input, never the
outcome.

**Out of sample** (fitted on 2023-2025Q3, tested on events in 2026Q1-Q2): ROC-AUC **0.89**
[0.85, 0.93]; the riskiest tenth of borrowers held 90 of the 114 events. On non-accrual alone,
0.86 [0.81, 0.91].

**Deliverables:** `scorecard_2026Q2.csv` (every accruing borrower ranked), `oos_performance.csv`
(the performance table) and `scripts/run_model.py`, which rebuilds both from `FINAL_DATASET.csv`.

```
python scripts/run_model.py            # about 6 minutes
python scripts/run_model.py --cached   # reuse the saved panel
python tests/test_model.py             # 21 tests
```

## Departures from the sponsor's note

- **Next quarter, not two to four quarters,** by the team's choice. One window everywhere.
- **No restructuring event:** the filings have no field for it. A PIK switch, the closest thing,
  is an input.
- **Test signals are from 2025Q4-2026Q1** so the events stay in 2026Q1-Q2.
- **The test is not fully blind:** the rule for keeping signs was fixed after it was seen.
  2026Q3 is held back as the blind test.

---

## 1. Events per quarter (`event_counts.csv`)

| Events in | Borrowers at risk | **Events** | of which impaired exits | Paid-off exits | No label |
|---|---|---|---|---|---|
| 2023Q3 | 3,399 | **35** | 8 | 124 | 36 |
| 2023Q4 | 3,584 | **39** | 17 | 196 | 84 |
| 2024Q1 | 4,000 | **28** | 6 | 133 | 135 |
| 2024Q2 | 4,227 | **50** | 6 | 184 | 30 |
| 2024Q3 | 4,491 | **34** | 5 | 200 | 52 |
| 2024Q4 | 4,776 | **56** | 17 | 357 | 78 |
| 2025Q1 | 4,852 | **45** | 7 | 213 | 74 |
| 2025Q2 | 5,167 | **43** | 10 | 223 | 43 |
| 2025Q3 | 5,404 | **40** | 4 | 225 | 41 |
| 2025Q4 | 5,782 | **66** | 8 | 322 | 50 |
| 2026Q1 | 5,838 | **57** | 10 | 284 | 85 |
| 2026Q2 | 6,082 | **57** | 13 | 284 | 51 |
| **All** | 57,602 | **550** | 111 | 2,745 | 759 |

Every exit is classified by the borrower's last mark: **paid off** (95 or above: repaid if at
maturity, refinanced if before), **impaired** (below 80: an event) or **unclassified** (80-95).
Unclassified exits and borrowers whose status is unknown have no label (1.3%).

## 2. Which signs are kept (`sign_selection.csv`)

Each sign is added alone to the baseline (mark and its change) and kept only if it adds at least
half a point of out-of-sample ROC-AUC on the development quarters (events up to 2025Q4).

| Model | ROC-AUC | Gain over the baseline [95% CI] | Kept |
|---|---|---|---|
| Baseline: mark and its change | 0.870 | | |
| + PIK switch | 0.876 | +0.006 [-0.001, 0.014] | **yes** |
| + fall over the past year | 0.873 | +0.002 [-0.001, 0.007] | no |
| + crowding | 0.868 | -0.002 [-0.007, 0.003] | no |
| + valuation gap | 0.867 | -0.003 [-0.011, 0.004] | no |

## 3. Out-of-sample performance (`oos_performance.csv`, `oos_score_deciles.png`)

Fitted once on signals up to 2025Q3; tested on 11,784 borrowers and 114 events in 2026Q1-Q2.

| Model | ROC-AUC [95% CI] | Events in the riskiest tenth |
|---|---|---|
| **Scorecard** | **0.888 [0.85, 0.93]** | **90 of 114** |
| Mark alone | 0.881 [0.83, 0.92] | 91 of 114 |
| Equal-weighted ranking of the same signs | 0.855 [0.81, 0.89] | 80 of 114 |
| Scorecard, non-accrual only (91 events) | 0.862 [0.81, 0.91] | 69 of 91 |

| Score decile | Realized event rate [95% CI] | Predicted |
|---|---|---|
| 10 (riskiest) | 7.6% [6.3, 9.3] | 8.4% |
| 9 | 0.6% [0.3, 1.2] | 0.6% |
| 1-8 | 0.2% [0.1, 0.3] | 0.3% |
| All | 0.97% [0.81, 1.16] | 1.15% |

- **The fitted weights beat equal weights** by 0.034 [0.013, 0.057].
- **Most of the skill is in the mark;** the other two signs add under a point.
- **Predictions match outcomes** within the intervals, running slightly high.
- **The risk sits in the top tenth.** Below it the scorecard does not separate borrowers.
- **No overfitting:** ROC-AUC is 0.874 on the training quarters and 0.888 on the test, and 0.83
  to 0.93 in each of nine quarters scored one at a time (`rolling_by_quarter.csv`).

## 4. The scorecard for 2026Q2 (`scorecard_2026Q2.csv`)

All 6,267 borrowers accruing in 2026Q2, ranked, with their probability and the signs behind it.
About 62 events are expected in 2026Q3, 43 of them in the riskiest tenth. At the top: Rural
Sourcing Holdings (50%: marked at 74, down 12 points, just switched to PIK). When 2026Q3 is filed,
a rerun scores this list against it (`reserved_test_2026Q3.csv`).

## Limitations

- **Few events:** 550 in 12 quarters, 114 in the test. Gaps of a point of AUC are noise.
- **Sudden defaults are missed:** a fifth of the events were outside the riskiest tenth.
- **Part of the headline is by construction:** an impaired exit is defined by a low mark. That is
  why 0.86 on non-accrual alone is shown next to 0.89.
- **Exit types are inferred** from the last mark and maturity; about 1 in 5 exits reappear later
  (filing gaps or name changes).
- **Non-accrual is missing** where filings report it only in HTML, and understated for some BDCs.

---

## Method

Code in `src/9_model/` (`model_panel.py`: panel and outcome; `model_fit.py`: logit and metrics;
`model_analysis.py`: tables; `model_charts.py`: charts). Outputs in `outputs/model/`.

- **Unit:** a borrower in a quarter, every BDC's loans pooled. Kept if it is not a CLO vehicle,
  has a mark on at least half its par, holds $1m+ of debt, and its loans were priced the quarter
  before.
- **Outcome:** for a borrower accruing at t, 1 if any BDC flags its debt non-accrual at t + 1 or
  it left impaired; 0 if it is still accruing or left paid off.
- **Mark:** fair value / par (fair value / cost for debt without par), capped below at 70.
  **Change:** on the same loans only, capped at 20 points. **PIK switch:** a loan that paid cash
  starts paying in kind, or its PIK share rises 10+ points.
- **Tested signs:** valuation gap (std of lender families' marks on the same loan), crowding
  (number of lender families), fall over the past year.
- **Stages:** development 2023Q2-2025Q3 signals (choose signs); test 2025Q4-2026Q1 signals
  (performance table); 2026Q2 signals scored for 2026Q3, held back.
- **Intervals:** borrower-resampled bootstrap for AUC; Wilson for rates; borrower-clustered
  standard errors for coefficients.

Further checks are written by the same run: each sign alone (`event_rates_by_bucket.csv`,
`event_profile.png`, `pik_switch_hypotheses.csv`), coefficients (`coefficients.csv`) and
alternative definitions (`robustness_summary.csv`, all within the intervals above).
