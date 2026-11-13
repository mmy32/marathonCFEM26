# Early-Warning Models: Non-Accrual and Valuation Deterioration

## The model in 30 seconds

Lenders mark a loan down before they stop booking its interest. So a borrower is more likely to go
on non-accrual within the next year when:

| Warning sign | Odds of non-accrual within 4 quarters |
|---|---|
| Its loans are marked low | each 10 points lower: about **2.6x** |
| The marks fell over the past year | each 5-point fall: about **1.6x** |
| It just switched to paying interest in kind (PIK) | **3x** |

The model is a logistic regression on these three signs. A mark below 70 counts as 70, and a fall
of more than 20 points counts as 20: beyond those points risk keeps rising, but much more slowly.
Tested out of sample (each quarter scored by a model fitted only on earlier quarters), it ranks
borrowers well: ROC-AUC 0.81 for the year ahead, 0.85 for next quarter. Of the riskiest 5% of
borrowers each quarter, about 1 in 4 goes on non-accrual within a year, seven times the average.
Its blind spot is borrowers still marked near par that default suddenly.

**Left out on purpose:**

- **Valuation dispersion** (lenders disagreeing on value), the outline's fourth sign. It adds
  nothing out of sample (ROC-AUC 0.808 with it and without it). It also needs two extra
  explanations: it exists only for borrowers with 2+ lender families, and the indicator for that
  has its own sign.
- **Complex models.** Boosted trees, a small neural network, ridge and lasso do no better
  (within 0.01).

```
python scripts/run_model.py            # about 8 minutes
python scripts/run_model.py --cached   # reuse data/model/model_panel.csv (skips the 2-minute panel build)
python tests/test_model.py             # 19 tests
```

---

## Results (run of 2026-11-12)

### 1. Each sign alone (`event_rate_by_predictor.png`, `event_profile.png`)

Accruing borrowers 2023Q2-2026Q1: 54,222 borrower-quarters, 8,802 borrowers, 441 entries into
non-accrual the next quarter (0.81%).

| Bucket | Entry rate next quarter |
|---|---|
| Mark below 80 / 80-90 / 98 and up | 14.3% / 4.8% / 0.2% |
| Drawdown over 10 points / none | 9.7% / 0.3% |
| PIK switch / none | 4.3% / 0.7% |
| Dispersion over 5 points / single lender | 6.6% / 0.9% |

Borrowers that enter non-accrual have a median mark of 96 a year before, then 94, 92, 86 in the
last quarter before, and 72 at entry. Borrowers that stay accruing hold at 99. One entry in five
comes from a borrower still marked at 98 or above.

### 2. The headline model (`coefficients.csv`, `odds_ratios_nonaccrual.png`)

Logistic regression on all three signs together. Odds multipliers [95% CI, clustered by borrower]:

| Sign | Next quarter | Within 2 quarters | Within 4 quarters |
|---|---|---|---|
| Mark 10 points lower | 3.1x [2.6, 3.6] | 2.9x [2.5, 3.4] | 2.6x [2.2, 3.1] |
| Mark fell 5 points in the past year | 1.4x [1.3, 1.6] | 1.5x [1.3, 1.6] | 1.6x [1.4, 1.8] |
| Switched to PIK this quarter | 3.3x [2.4, 4.6] | 2.6x [2.0, 3.5] | 3.0x [2.3, 3.9] |

Looking further ahead, the current mark matters a little less and its fall a little more.

**PIK switch, outline 4.3** (`pik_switch_hypotheses.csv`). A switch on a mark below 90 leads to
non-accrual next quarter 13.9% of the time, against 7.6% for a low mark without a switch. A
switch on a mark of 90 or above is a weak signal (2.0%).

### 3. Out of sample: rolling test (`rolling_metrics.csv`, `review_list_capture.csv`)

Every quarter is scored by a model fitted only on earlier quarters whose outcomes were already
filed. Intervals resample borrowers.

**By horizon**, on the three quarters all horizons share (2024Q4-2025Q2), ROC-AUC:

| Entry | Base rate | Mark alone | **Three signs** | All four | Boosted trees (all four) | Top 5% list: entries caught / share that enters |
|---|---|---|---|---|---|---|
| Next quarter | 0.7% | 0.834 | **0.852** [0.80, 0.90] | 0.859 | 0.847 | 56% / 8% |
| Within 2 quarters | 1.7% | 0.799 | **0.819** [0.78, 0.86] | 0.821 | 0.824 | 44% / 15% |
| Within 4 quarters | 3.8% | 0.784 | **0.808** [0.77, 0.84] | 0.808 | 0.812 | 35% / 27% |

- **Further ahead is harder,** but the list is more useful: a quarter of the year-ahead top 5%
  enter non-accrual, against 8% next quarter.
- **The drawdown and PIK switch earn their place at 4 quarters:** +0.024 ROC-AUC over the mark
  alone [0.006, 0.045]. At next quarter they add +0.007 [-0.002, 0.018], within noise.
- **Dispersion adds nothing:** three signs vs all four differ by -0.001 [-0.013, 0.012] at 4
  quarters and +0.002 [-0.003, 0.008] at next quarter.
- **The training horizon doesn't change who is on the list** (`horizon_cross_check.csv`). Against
  entry within 4 quarters, models trained for 1, 2 and 4 quarters reach ROC-AUC 0.809, 0.809 and
  0.808. The signs give one risk ordering; the horizon only sets which question the probabilities
  answer.

**Next quarter, all 9 test quarters** (2024Q1-2026Q1, 369 entries; `rolling_auc_by_quarter.png`):
ROC-AUC 0.848 [0.824, 0.874], between 0.74 and 0.91 in every quarter (mark alone 0.841).

| Review list each quarter | Borrowers | Next quarter: caught / share that enters | Within 4 quarters: caught / share that enters |
|---|---|---|---|
| Top 1% | ~45 | 23% / 19% | 13% / 48% |
| Top 5% | ~230 | 58% / 10% | 35% / 27% |
| Top 10% | ~460 | 69% / 6% | 53% / 20% |
| Top 20% | ~920 | 80% / 3% | 72% / 14% |

- **Probabilities are accurate** (`rolling_calibration.csv`): 380 entries predicted against 369
  that happened (next quarter). Top tenth: 5.7% predicted vs 5.8% actual (next quarter), 19.4% vs
  20.1% (4 quarters). The very top 1% runs high (69% predicted, 48% actual, at 4 quarters).
- **Early warning** (`early_warning.csv`): 69% of entrants are on the top-10% list the quarter
  before, 57% two quarters before, and 44% a year before.
- **Against simple rules** (`rolling_rules_vs_model.csv`), flagging the same number of borrowers:
  "mark below 90" catches 60% of next-quarter entries and the model 62%. "Drawdown 5+ points"
  catches 61% and the model 67%. Most of the skill is in the mark; the model ranks within it.
- **Blind spot:** 74 of the 369 entrants were outside the top 20%. 88% of these were still marked
  at 98 or above, with no fall and no PIK switch.

**Horizon.** Entry within 4 quarters is the recommended target: a year of warning, a list where
1 in 4 enters, and the same ranking as the next-quarter model, which holds up over 9 test
quarters. Its own test is thinner (3 quarters). Next quarter stays as the outline's test.

### 4. The outline's single split (`oos_metrics.csv`, `holdout_auc.png`, `holdout_roc_pr.png`)

Train to 2025Q4, test 2026Q1 borrowers on 2026Q2 entries (5,777 borrowers, 44 entries). The
outline's own test (2026Q2 to 2026Q3) runs automatically once 2026Q3 is in FINAL_DATASET.

| Model | ROC-AUC [95% CI] |
|---|---|
| Mark alone | 0.84 [0.75, 0.91] |
| **Three signs (logistic)** | **0.85 [0.77, 0.93]** |
| All four: logistic / ridge / lasso | 0.85 / 0.85 / 0.86 |
| All four: boosted trees / neural network | 0.86 / 0.86 |

At the threshold chosen in validation to catch 80% of entries, the model catches 38 of 44
(86%) but flags a third of all borrowers. That is why the review list is the headline instead.
The six entries it missed (Simon Data, Service Master, Lonestar EMS, Everbridge, Quirch Foods,
Vesta Housing; `holdout_review.csv`) were all marked at 99-105 with no warning sign. The
highest-scored borrowers that stayed accruing (RQM Buyer, Eagle Midco, MPH Acquisition) had
just switched to PIK: lenders gave relief instead of stopping accrual.

### 5. Valuation outcomes (`rolling_valuation.csv`)

| Outcome | In-sample R2 | Rolling out-of-sample R2 (9 quarters) |
|---|---|---|
| `price_change` (same loans, t to t + 1) | 0.043 | 0.040 (by quarter -0.03 to 0.07) |
| `val_ret` (outline formula) | 0.004 | 0.003 |

Next quarter's mark is only weakly predictable, and almost entirely through the drawdown: a
5-point fall predicts a further 0.7-point markdown, so marks have momentum. `val_ret` moves with
new lending and repayments, not with marks. The signs warn of **credit events**, not of next
quarter's marks.

### 6. Robustness (`robustness_summary.csv`, three-sign model, holdout ROC-AUC)

| Variant | Holdout ROC-AUC |
|---|---|
| Headline | 0.854 |
| + dispersion (std / range) | 0.852 / 0.855 |
| PIK as change in PIK share | 0.844 |
| Market-adjusted mark and drawdown | 0.854 |
| + seniority mix / + borrower size | 0.855 / 0.861 |
| Outcome: first-ever entry (34 entries) / same-lender entries | 0.825 / 0.854 |
| Borrowers with 2+ lender families, + dispersion (13 entries) | 0.891 |

Nothing moves outside the interval. Larger borrowers enter slightly more often at the same mark
(partly mechanical: more lenders, more chances one flags it).

### 7. Forecast from 2026Q2 (`forecast_2026Q2.csv`)

6,328 borrowers accruing in 2026Q2 get probabilities of entering non-accrual within 1, 2 and 4
quarters. On average these are 1.0%, 2.1% and 4.3%, about 270 entries expected by 2027Q2. The
review list is the top 5% by the 4-quarter score: 316 borrowers, each at 14% or more and 37% on
average.

At the top are borrowers that just switched to PIK on a falling mark: Rural Sourcing Holdings
(80% within a year), REP SEKO Merger Sub (78%) and Invincible Boat (68%). Next comes a group
marked below 70 that fell 20+ points, all at the model's ceiling (65%) and ordered by mark:
Flexsys (20), Horizon Facilities (27), Vibrantz (31), Alliance Environmental (33), Veracode (36),
Datacom (37), ICS (38), SonicWall (38), Ivanti (40). When 2026Q3 is filed, a rerun tests these.

---

## Limitations

- **Short history:** 9 test quarters at next quarter, 3 at 4 quarters. Each new quarter of filings
  adds one at every horizon.
- **Sudden defaults are invisible:** one entrant in five gave no warning in the marks.
- **Marks are the lenders' own:** a low mark and a later non-accrual flag can be the same
  judgment, seen twice. The model reads lenders' view from outside; it does not see the borrower.
- **Non-accrual is unknown** for filings that report it only in HTML (14% of fair value), and it is
  understated for Ares Capital and similar BDCs (rows without a rate were removed in cleaning).
- **Exits are not outcomes:** a borrower no longer held has no label, whether repaid or written off.
- **CLO tranches are excluded by their seniority label,** which misses a few.

---

## Method

Code: `src/9_model/`. Reads only `FINAL_DATASET.csv`, through the tranche-quarter panel of
`src/8_risk_monitor/monitor_panel.py` (`11_tranche_matching.md`).

| Module | What it does |
|---|---|
| `model_panel.py` | Borrower-quarter panel: signs, outcomes, sample flags |
| `model_fit.py` | Model frames, logit / OLS with borrower-clustered errors, extensions, validation, rolling predictions and their metrics |
| `model_analysis.py` | Descriptive tables, rolling evaluation by horizon, in-sample, single-split and robustness runs |
| `model_charts.py` | The seven charts (deck style) |

Writes `data/model/model_panel.csv` and `data/model/rolling_predictions.csv` (git-ignored), and
the tables and charts named above to `outputs/model/`. Also written there: `sample_waterfall.csv`
and `sample_summary.csv` (counts after each rule and per window), `predictor_summary.csv`, and
`robustness_coefficients.csv`.

**Unit and sample.** A borrower (validated name) in a quarter, with every BDC's loans pooled. It
is kept when:

- it is not a CLO / structured vehicle (under half its par in CLO tranches);
- it has a price (fair value / par on priceable loans, at most 1.5) covering at least half its
  funded par;
- it holds at least $1m of debt fair value;
- it was held the quarter before, so 2023Q1 drops out as a predictor quarter.

**Outcomes.**

- **New non-accrual within h quarters** (`y_na`, `y_na_h2`, `y_na_h4`). For a borrower accruing
  at t (at least one BDC's status known, none flagging it), it is 1 if any BDC flags any of its
  debt as non-accrual in t + 1 .. t + h. It is 0 if the borrower is still held and accruing at
  t + h, and missing otherwise. Status is read from all debt rows, including loans marked at zero.
  A row is labelled only once t + h is in the data. Of the 441 next-quarter entries, 428 come from a BDC that reported the
  borrower accruing the quarter before.
- **Valuation outcomes.** `val_ret = (FV(t+1) - FV(t)) / FV(t)` on debt fair value (the
  outline's formula, no cost). `price_change` is the change in fair value / par on the same loans
  held in both quarters, so new lending and repayments do not move it (`price_change_raw`, the
  pooled difference, is a robustness check). Both are capped at their training 1st / 99th
  percentiles: `val_ret` reaches +21,000% when lending grows.

**Signs.** All measured with what is filed for quarter t:

- **Mark:** fair value / par.
- **Rolling max drawdown:** the largest fall of the mark over the past four quarters, measured on
  the same loans, so a new loan bought at a discount is not a fall.
- **PIK switch:** a loan that paid cash the quarter before starts paying in kind, or its PIK share
  of the coupon rises 10+ points.
- **Dispersion** (tested, not in the headline): the std of the manager families' marks on the same
  tranche, for tranches held by 2+ families.

PIK rates are never missing in FINAL_DATASET.

**Testing.**

- **Rolling:** for each test quarter q, fit on the quarters up to q - h, whose outcomes are all
  filed by q. A quarter is tested once it has three training quarters. Boosted trees use fixed
  settings (depth 2, learning rate 0.1). Tests check that no fold sees an outcome filed after its
  test quarter.
- **Outline split:** for the extensions, expanding-window validation over the last four training
  quarters chooses the hyperparameters and a threshold that catches 80% of validation entries.
- **Inference:** standard errors are clustered by borrower.
