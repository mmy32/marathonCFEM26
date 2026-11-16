# Early-Warning Scorecard: Which Borrowers Enter Non-Accrual Within Two Quarters

## In 30 seconds

Lenders mark a loan down before they stop booking its interest. The scorecard is a logistic
regression on five signs from each quarter's filings:

| Warning sign | Odds of entering non-accrual within two quarters |
|---|---|
| Its loans are marked low | each 10 points lower: about **2.8x** |
| It just switched to paying interest in kind (PIK) | about **2.8x** |
| The mark fell over the past year | each further 5 points: about **1.3x** |
| The mark fell this quarter | each 5-point fall: about **1.3x** |
| Its lenders disagree on its value | each point between the highest and lowest mark: about **1.02x** (not significant) |

The outcome is **non-accrual within the next two quarters**: some BDC flags the borrower's debt
non-accrual at t + 1 or t + 2. A markdown is an input, never the outcome.

**Out of sample** (fitted on signals up to 2025Q1, tested on signals from the second half of 2025,
entries in 2025Q4-2026Q2): ROC-AUC **0.83** [0.78, 0.87]; the riskiest tenth of borrowers held
125 of the 190 entries. **Almost all of that comes from the mark:** the mark and its change
alone score the same 0.83.

**Deliverables:** `scorecard_2026Q2.csv` (every accruing borrower ranked), `oos_performance.csv`
(the performance table) and `scripts/run_model.py`, which rebuilds both from `FINAL_DATASET.csv`
(itself built from the filings by `src/1_data_processing`).

```
python scripts/run_model.py            # about 6 minutes
python scripts/run_model.py --cached   # reuse the saved panel
python tests/test_model.py             # 31 tests
```

## How the sponsor's definitions are applied

- **Non-accrual is the outcome; restructurings are counted beside it** (section 2). The filings
  have no restructuring field, so a restructuring is read from a change in terms on the same loan
  at the same BDC from one quarter to the next: a switch to PIK, a spread step-up, a maturity
  extension, a debt-for-equity swap, a cost write-down. A realized loss is not an event: it shows
  up as an impaired exit.
- **Every exit goes in one of four buckets:** repaid, refinanced, impaired, unresolved (section
  2). The cutoffs were fixed before any count was run and every count is shown at a second
  cutoff. Exits never enter the outcome.
- **The mark is fair value / par**, and fair value / cost for debt reported without par. A partly
  drawn revolver is left out of the mark: its par is the commitment, not what was lent.
- **Only what was public at the time.** A quarter's signs are public when its filing is, 45 to 90
  days after quarter-end, and every fit uses only outcome windows that had closed by then. An
  entry at t + 1 becomes public one filing later and one at t + 2 two filings later, so the
  scorecard warns one to two filings ahead.
- **Two quarters, the short end of the note's two to four,** so that the test can use signals
  from the second half of 2025 and every window still closes inside the data. The test's entries
  therefore fall in 2025Q4-2026Q2 rather than only in the first half of 2026: the window of a
  2025Q3 signal starts in 2025Q4.
- **The rule for keeping signs and the caps were fixed on 2026-11-13,** after an earlier
  one-quarter version of the test had been seen. They were applied unchanged to this outcome.
  On 2026-11-16 the valuation gap was redefined from the standard deviation of lenders' marks to
  the range, so that "points apart" is literally true and it matches the midterm deck; the rule
  keeps it either way, and the test ROC-AUC is the same 0.829. 2026Q3 is held back as the blind
  test.

---

## 1. Entries per quarter (`event_counts.csv`)

| Signals in | Window | Borrowers at risk | **Entered non-accrual** | Still accruing | Left the books (no label) | Status unknown |
|---|---|---|---|---|---|---|
| 2023Q2 | 2023Q3-Q4 | 3,399 | **48** | 2,908 | 382 | 61 |
| 2023Q3 | 2023Q4-2024Q1 | 3,584 | **39** | 3,113 | 381 | 51 |
| 2023Q4 | 2024Q1-Q2 | 4,000 | **60** | 3,480 | 326 | 134 |
| 2024Q1 | 2024Q2-Q3 | 4,227 | **70** | 3,699 | 408 | 50 |
| 2024Q2 | 2024Q3-Q4 | 4,491 | **64** | 3,787 | 575 | 65 |
| 2024Q3 | 2024Q4-2025Q1 | 4,776 | **74** | 4,009 | 604 | 89 |
| 2024Q4 | 2025Q1-Q2 | 4,852 | **70** | 4,251 | 460 | 71 |
| 2025Q1 | 2025Q2-Q3 | 5,167 | **70** | 4,580 | 472 | 45 |
| 2025Q2 | 2025Q3-Q4 | 5,404 | **91** | 4,729 | 544 | 40 |
| 2025Q3 | 2025Q4-2026Q1 | 5,782 | **103** | 4,958 | 640 | 81 |
| 2025Q4 | 2026Q1-Q2 | 5,838 | **87** | 5,075 | 603 | 73 |
| **All** | | 51,520 | **776** | 44,589 | 5,395 | 760 |

45,365 borrower-quarters are labelled (entered or still accruing); 1.71% enter non-accrual within
two quarters. Of the 44,589 still accruing, 7,455 had a restructuring in the window (mostly a
maturity extension); they stay labelled 0. The 5,395 that left the books are split into the four
exit buckets in section 2.

Windows overlap (a borrower's quarters t and t + 1 share a quarter), so standard errors are
clustered and intervals resampled by borrower.

## 2. Restructurings and exits

Both are read from how each BDC's schedule changes between consecutive filings
(`src/9_model/credit_events.py`). A "loan" is one BDC's tranche followed over time (the tranche
chains of doc 11), for operating borrowers only (no CLO tranches). A loan's mark is fair value /
par.

**Cutoffs**, fixed on 2026-11-16 before any count was run:

| Rule | Primary | Second |
|---|---|---|
| Repaid: last mark at or near par | 95 or above | 98 or above |
| Impaired: last mark well below par | 80 or below | 90 or below |
| Spread step-up | +50 bp | +100 bp |
| Maturity extension | +90 days | +365 days |
| Write-down: cost cut while par stays | 5 points of par | 10 points |
| PIK switch | PIK starts, or the PIK share of the coupon rises 10+ points | PIK starts on a cash-pay loan |

### Restructurings (`restructuring_events.csv`, `restructuring_vs_nonaccrual.csv`)

224,672 loans were seen in two consecutive filings (2023Q2-2026Q2):

| Restructuring | Loans, primary | Loans, second cutoff | Accruing borrowers with it at t | **Entered non-accrual within two quarters** |
|---|---|---|---|---|
| PIK switch | 3,789 | 3,031 | 1,230 | **7.2%** |
| Spread step-up | 3,543 | 1,390 | 1,256 | 2.9% |
| Maturity extension | 7,099 | 5,805 | 2,367 | 1.6% |
| Replaced by equity | 32 | 32 | 4 | 0 of 4 |
| Write-down | 145 | 55 | 46 | 1 of 46 |
| **Any** | 12,861 | 9,453 | 4,217 | 3.1% |
| None | | | 41,148 | 1.6% |

- **A PIK switch is the restructuring that precedes non-accrual,** at 4.5 times the rate of
  borrowers without one. It is already a sign in the scorecard.
- **A maturity extension says nothing:** extended borrowers enter non-accrual at the same rate
  as the rest. Most are routine amend-and-extend deals of healthy borrowers, and they are the
  largest type.
- **Swaps and write-downs are rare in the data:** most BDCs' schedules as parsed carry few equity
  rows, and a write-off usually takes the loan out of the schedule (an impaired exit), not down
  in it.
- **Robustness:** with "non-accrual or any restructuring within two quarters" as the outcome
  (8,516 events), the scorecard's test ROC-AUC falls to 0.59 (`robustness_summary.csv`). The
  restructurings it adds are mostly the benign extensions above, so non-accrual stays the
  outcome.

### Exits (`exits_unlabelled.csv`)

The 5,395 borrower-quarters in section 1 that left every BDC's books have no label. None is
dropped: each goes in one bucket, from its loans that left in the exit quarter (impaired if any
was, repaid if all were; the loan rules are in the backup below).

| Bucket | Exits, primary | Share | Second cutoff | Share |
|---|---|---|---|---|
| Repaid | 4,064 | 75% | 3,629 | 67% |
| Refinanced | 0 | | 0 | |
| Impaired | 208 | 4% | 408 | 8% |
| Unresolved | 1,123 | 21% | 1,358 | 25% |

- **Refinanced is zero by construction:** a refinanced borrower is still on some BDC's books, so
  it stays in the sample and gets a label.
- **Impaired exits are not events:** a realized loss is reported as an impaired exit. The
  scorecard flags them anyway: in the test quarters it put 36 of the 37 impaired exits in its
  riskiest tenth, against 15 of the 950 repaid (section 4).
- **The unresolved fall into three roughly equal groups:** the lender skipped its next filing
  (443), the last mark was between 80 and 95 or missing (344), or no loan could be traced to the
  exit quarter (336).

### Backup: every loan exit (`exits.csv`), and two data filters

A loan exits when its chain ends while the data go on. Each loan exit goes in exactly one bucket,
tested in this order:

1. **Impaired:** it was non-accrual at its last filing (or its BDC flagged the borrower the next
   quarter), its last mark was well below par, or the BDC holds new equity in the borrower the
   next quarter.
2. **Refinanced:** the same BDC has a new tranche to the borrower the next quarter, or another
   BDC still holds the borrower.
3. **Repaid:** its last mark was at or near par, and its BDC filed the next quarter (if it did
   not, the loan's absence says nothing).
4. **Unresolved:** everything else.

Where the reported par is a commitment rather than what was lent (a partly drawn revolver or
delayed-draw loan: cost under 80% of par), fair value / par is not a price, and the exit's mark is
fair value / cost (1,081 of the 26,357 exits, 4%).

| Bucket | Loans, primary | Share | Loans, second cutoff | Share |
|---|---|---|---|---|
| Repaid | 8,439 | 32% | 7,525 | 29% |
| Refinanced | 14,841 | 56% | 14,189 | 54% |
| - another BDC, already lending | 8,409 | 32% | 7,954 | 30% |
| - another BDC, new loan that quarter | 5,435 | 21% | 5,279 | 20% |
| - new tranche, same BDC | 997 | 4% | 956 | 4% |
| Impaired | 1,773 | 7% | 2,738 | 10% |
| - non-accrual | 958 | 4% | 958 | 4% |
| - marked well below par | 793 | 3% | 1,759 | 7% |
| - replaced by equity | 22 | 0.1% | 21 | 0.1% |
| Unresolved | 1,304 | 5% | 1,905 | 7% |
| - mark between the cutoffs, or none | 753 | 3% | 1,394 | 5% |
| - BDC did not file the next quarter | 551 | 2% | 511 | 2% |
| **All** | 26,357 | | 26,357 | |

"Another BDC still holds the borrower" is the sponsor's rule as written. It is broad: in 8,409
exits the other BDC was already lending before, so the loan may simply have been repaid while the
borrower's other loans continue. Counting only exits where another BDC started a new loan that
quarter, refinanced would fall to 6,432 (24%), and the rest would be repaid, impaired or
unresolved by their marks.

Two filters guard against filing artifacts in the restructurings. Both were added after looking
at examples, and both are the same at either cutoff. A spread step-up counts only if the all-in
rate rose with it (by the step less 75 bp of base-rate movement), because a relabel from prime to
SOFR raises the reported spread by about three points while the all-in rate falls. A write-down
counts only on a loan already marked at least that far below its cost, which is what a realized
loss looks like; a revolver paid down at par also cuts cost while par (the commitment) stays.

## 3. Which signs are kept (`sign_selection.csv`)

Each sign is added alone to the baseline (mark and its change) and kept if it adds at least half
a point of out-of-sample ROC-AUC on the development quarters (signals 2024Q2-2025Q1, 278 entries).

| Model | ROC-AUC | Gain over the baseline [95% CI] | Kept |
|---|---|---|---|
| Baseline: mark and its change | 0.802 | | |
| + PIK switch | 0.808 | +0.007 [-0.003, 0.024] | **yes** |
| + fall over the past year | 0.807 | +0.006 [-0.002, 0.015] | **yes** |
| + valuation gap | 0.807 | +0.006 [-0.009, 0.020] | **yes** |
| + crowding | 0.801 | -0.001 [-0.006, 0.005] | no |
| Scorecard: all three kept signs | 0.814 | +0.012 [-0.004, 0.032] | |

No gain's interval clears zero; the rule keeps three signs by small margins.

## 4. Out-of-sample performance (`oos_performance.csv`, `oos_score_deciles.png`)

Fitted once on signals up to 2025Q1 (every window closed by 2025Q3); tested on signals from
2025Q3-Q4: 10,223 borrowers, 190 entries in 2025Q4-2026Q2.

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Entries caught in the riskiest 10% | in the riskiest 20% |
|---|---|---|---|---|
| **Scorecard** | **0.829 [0.78, 0.87]** | **0.189 [0.14, 0.25]** | **125 of 190** | **149 of 190** |
| Mark and its change | 0.830 [0.78, 0.87] | 0.188 [0.13, 0.25] | 130 of 190 | 145 of 190 |
| Mark alone | 0.824 [0.77, 0.87] | 0.164 [0.12, 0.22] | 128 of 190 | 145 of 190 |
| Equal-weighted ranking of the same signs | 0.798 [0.75, 0.84] | 0.117 [0.09, 0.16] | 103 of 190 | 132 of 190 |
| Random ranking | 0.50 | 0.019 (the entry rate) | 19 of 190 | 38 of 190 |

PR-AUC scores only how well entries are found, so it suits a rare event; its random baseline is
the entry rate, not 0.5. The scorecard's 0.19 is about 10 times random.

| Score decile | Realized entry rate [95% CI] | Predicted | Realized ÷ predicted [95% CI] |
|---|---|---|---|
| 10 (riskiest) | 12.2% [9.8, 15.0] | 9.8% | 1.25 [1.01, 1.52] |
| 9 | 2.3% [1.5, 3.5] | 1.3% | 1.80 [1.13, 2.72] |
| 1-8 | 0.50% [0.33, 0.69] | 0.72% | 0.70 [0.46, 0.95] |
| All | 1.86% [1.56, 2.15] | 1.69% | **1.10 [0.93, 1.29]** |

Realized ÷ predicted is 1 when the model predicts the right number of entries; its interval, like
the others, resamples borrowers with the predictions held fixed (`oos_deciles.csv`).

- **The mark does the work.** The scorecard ties the mark and its change (difference -0.001
  [-0.021, 0.019]). The three added signs helped slightly on the development quarters and not
  on the test. The scorecard is kept because the rule, fixed in advance, chose it; a version with
  only the PIK switch added scored 0.836 (`robustness_summary.csv`), within the same interval.
- **Fitted weights vs equal weights:** +0.030 [-0.002, 0.061], better but not clearly.
- **Right on average, too flat across deciles:** the predicted entry rate is within the interval
  of the realized one overall, but the riskiest tenth had 25% more entries than predicted and
  the bottom eight tenths 30% fewer. Entries rose in 2025-2026 relative to the training years.
- **The risk sits in the top fifth.** Below it the scorecard barely separates borrowers.
- **No overfitting:** ROC-AUC is 0.827 on the training quarters and 0.829 on the test, and 0.78
  to 0.85 in each of seven quarters scored one at a time, each by a model refitted on the windows
  closed by then (`rolling_by_quarter.csv`, `rolling_auc_by_quarter.png`).

**The borrowers the test leaves out** (`oos_ranks_by_outcome.csv`). The test scores only labelled
borrowers. Ranking every accruing borrower of 2025Q3-Q4 instead, labelled or not:

| What happened next | Borrower-quarters | In the riskiest tenth | In the riskiest fifth |
|---|---|---|---|
| Entered non-accrual | 190 | 125 (66%) | 149 (78%) |
| Still accruing | 10,033 | 904 (9%) | 1,914 (19%) |
| Left the books: impaired | 37 | **36 (97%)** | 37 (100%) |
| Left the books: unresolved | 256 | 69 (27%) | 122 (48%) |
| Left the books: repaid | 950 | 15 (2%) | 76 (8%) |
| Status unknown | 154 | 12 (8%) | 25 (16%) |

Impaired exits are not events, but the scorecard flags them as strongly as the entries. This is
partly by construction: an exit marked well below par counts as impaired, and a low mark is the
scorecard's main sign.

## 5. The scorecard for 2026Q2 (`scorecard_2026Q2.csv`)

All 6,267 borrowers accruing in 2026Q2, ranked, with their probability and the signs behind it.
About 120 entries are expected in 2026Q3-Q4, 70 of them in the riskiest tenth. At the top: Rural
Sourcing Holdings (68%: marked at 74, down 12 points this quarter and 25 over the year, just
switched to PIK, its two lenders' marks 22 points apart). Once 2026Q3 is filed, a rerun tests the
scorecard blind on the 2026Q1 signals (`reserved_test_2026Q3.csv`); this list itself is scored
once 2026Q4 is filed.

## Limitations

- **Few events:** 776 in 11 windows, 190 in the test. Gaps of a point of AUC are noise.
- **Sudden defaults are missed:** a third of the entries were outside the riskiest tenth.
- **Losses without non-accrual are not in the outcome:** a borrower sold or written off without
  first being flagged leaves the books unlabelled. It is counted as an impaired exit when it left
  marked well below par (208 exits, 408 at the second cutoff; section 2).
- **Restructurings are inferred,** from term changes on loans the matcher links across quarters.
  An extension or a step-up that the matcher reads as a new loan counts as a refinancing instead.
- **Non-accrual is missing** where filings report it only in HTML, and understated for some BDCs.

---

## Method

Code in `src/9_model/` (`model_panel.py`: panel, outcome and exits; `model_fit.py`: logit and
metrics; `model_analysis.py`: tables; `model_charts.py`: charts). Outputs in `outputs/model/`.

- **Unit:** a borrower in a quarter, every BDC's loans pooled. Kept if it is not a CLO vehicle,
  has a mark on at least half its par, holds $1m+ of debt, and its loans were priced the quarter
  before.
- **Outcome:** for a borrower accruing at t, 1 if any BDC flags its debt non-accrual at t + 1 or
  t + 2; 0 if it is held and known accruing in both; no label if it leaves every BDC's books
  before the window closes, its status is unknown, or the window runs past the data.
- **Exits and restructurings:** section 2; the cutoffs and filters are constants at the top of
  `credit_events.py`.
- **Mark:** fair value / par (fair value / cost for debt without par), capped below at 70.
  **Change:** on the same loans only, capped at 20 points. **PIK switch:** a loan that paid cash
  starts paying in kind, or its PIK share rises 10+ points. **Fall over the past year:** largest
  peak-to-trough fall of the same-loan mark over four quarters, capped at 20. **Valuation gap:**
  highest minus lowest lender family's mark on the same loan, as in the midterm deck (0 with an
  indicator for single-lender borrowers).
- **Tested and dropped:** crowding (number of lender families).
- **Stages:** a quarter's window closes two quarters later, so each fit uses only windows closed
  before the quarter it scores. Development: signals 2023Q2-2025Q1 (choose signs, rolling test
  2024Q2-2025Q1). Test: signals 2025Q3-2025Q4 (performance table; 2025Q2 is skipped because its
  window overlaps the test). Scorecard: signals 2026Q2, fitted on 2023Q2-2025Q4.
- **Intervals:** borrower-resampled bootstrap for every performance measure and rate (AUC,
  recall, event rates), since a borrower appears in several quarters and one default can fall in
  two overlapping windows; borrower-clustered standard errors for coefficients.

Further checks are written by the same run: each sign alone (`event_rates_by_bucket.csv` and
`event_rate_by_predictor.png`, all five signs and crowding; `event_profile.png`,
`pik_switch_hypotheses.csv`), coefficients (`coefficients.csv`) and alternative definitions
(`robustness_summary.csv`: test ROC-AUC 0.82 to 0.84 for every variant of the non-accrual
outcome, all within the interval above; 0.59 with restructurings added to it). Entrants' median
mark falls from 96 four quarters before entry to 85 the quarter before; borrowers that stay
accruing hold at 99 (`event_profile.csv`).
