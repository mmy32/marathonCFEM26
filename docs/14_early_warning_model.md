# Early-Warning Scorecard: Which Borrowers Have a Credit Event Within Two Quarters

## In 30 seconds

Lenders mark a loan down before they stop booking its interest or change its terms. The scorecard
is a logistic regression on four signs from each quarter's filings:

| Warning sign | Odds of a credit event within two quarters |
|---|---|
| It just switched to paying interest in kind (PIK) | about **6.5x** |
| Its loans are marked low | each 10 points lower: about **2.1x** |
| The mark fell over the past year | each further 5 points: about **1.4x** |
| The mark fell this quarter | each 5-point fall: about **1.06x** (not significant) |

The outcome is **a credit event within the next two quarters**: at t + 1 or t + 2 some BDC flags
the borrower's debt non-accrual, or a lender makes a concession on one of its loans (switches its
cash interest to PIK, swaps the debt for equity, or writes down its cost). A markdown is an input,
never the outcome.

**Out of sample** (fitted on outcomes through 2025Q3, tested on the borrowers accruing in 2025Q3
and their events in 2025Q4-2026Q1): ROC-AUC **0.76** [0.73, 0.79]; the riskiest tenth of borrowers
held 136 of the 321 events and the riskiest fifth 207. The mark alone scores 0.74; the two added
signs are worth about two points.

**Two kinds of event, found unequally.** The same scorecard finds the non-accruals well (ROC-AUC
0.82; 65 of 103 in the riskiest tenth) and the concessions less well (0.73; 72 of 218): most
concessions are PIK switches on loans still marked near par.

**Deliverables:** `scorecard_2026Q2.csv` (every accruing borrower ranked), `oos_performance.csv`
(the performance table) and `scripts/run_model.py`, which rebuilds both from `FINAL_DATASET.csv`
(itself built from the filings by `src/1_data_processing`).

```
python scripts/run_model.py            # about 6 minutes
python scripts/run_model.py --cached   # reuse the saved panel
python tests/test_model.py             # 33 tests
python scripts/check_look_ahead.py     # look-ahead check (section 6), about 8 minutes
```

## How the sponsor's definitions are applied

- **A credit event is non-accrual or a restructuring in which the lender gives something up.**
  The filings have no restructuring field, so a restructuring is read from a change in terms on
  the same loan at the same BDC from one quarter to the next. The sponsor lists five such changes
  (section 2). Three are concessions and are in the outcome: a switch to PIK (cash interest
  deferred), a debt-for-equity swap, a cost write-down.
- **A spread step-up and a maturity extension are counted beside the outcome, not in it.** A
  step-up pays the lender more, and most extensions are routine amend-and-extend deals. With all
  five in the outcome there would be 8,516 events instead of 2,995, and 3,858 of them would be an
  extension alone on loans marked at par (median 99.8, against 99.5 for borrowers with no event).
  That version is a labelled robustness row (ROC-AUC 0.56). This choice was made on 2026-11-16
  after the counts of all five had been seen.
- **A realized loss is in the outcome where the schedule shows one.** A write-down or a swap into
  equity happens while the borrower stays on the books. When a loan simply leaves, the schedule
  does not say what was recovered, so that loss is reported as an impaired exit, as the sponsor
  asks, and never enters the outcome.
- **Every exit goes in one of four buckets:** repaid, refinanced, impaired, unresolved (section
  2). The cutoffs were fixed before any count was run and every count is shown at a second
  cutoff.
- **The mark is fair value / par**, and fair value / cost for debt reported without par. A partly
  drawn revolver is left out of the mark: its par is the commitment, not what was lent.
- **Only what was public at the time** (`filing_lags.csv`). Every one of the 1,986 filings was
  public within 93 days of its quarter-end: quarterly reports in about 40 days, annual reports
  (the fourth quarter) in about 65. A quarter's scorecard can be made on the day of its last
  filing and uses nothing filed later: its signs come from that quarter's filings, and every fit
  uses only outcome windows that had closed by then. An event at t + 1 becomes public one filing
  later and one at t + 2 two filings later, so the scorecard warns one to two filings ahead.
- **Two quarters is the primary, short-horizon specification,** the short end of the sponsor's
  two to four. Every stage scores one quarter of borrowers, so every scored window is two quarters
  long. A four-quarter version is not run: its latest testable signals would be 2025Q2, fitted on
  five quarters. It becomes possible once 2026Q3 and 2026Q4 are loaded.
- **No look-ahead in the data pipeline** (`scripts/check_look_ahead.py`, section 6): the panel
  rebuilt from data cut after 2025Q1 or 2025Q3 gives the same signs, sample and scores as the
  full build.
- **The 2025Q3 test is not untouched.** The rule for keeping signs and the caps were fixed on
  2026-11-13 after an earlier version of the test had been seen. The outcome was widened from
  non-accrual alone on 2026-11-16, after that outcome's test result (0.81) had been seen; the
  alternatives were compared on the development quarters only, and the rule and the caps were
  applied unchanged. The data stay at 2023Q1-2026Q2 for now, so this is the final test. A blind
  test needs 2026Q3: a rerun with it loaded scores the borrowers of 2026Q1 on their events in
  2026Q2-Q3, and nothing in the model should change after that run.

---

## 1. Events per quarter (`event_counts.csv`)

| Signals in | Window | Borrowers at risk | **Credit events** | with non-accrual | concession only | No event | Left the books (no label) | Status unknown |
|---|---|---|---|---|---|---|---|---|
| 2023Q2 | 2023Q3-Q4 | 3,399 | **215** | 48 | 167 | 2,746 | 377 | 61 |
| 2023Q3 | 2023Q4-2024Q1 | 3,584 | **223** | 39 | 184 | 2,934 | 379 | 48 |
| 2023Q4 | 2024Q1-Q2 | 4,000 | **242** | 60 | 182 | 3,304 | 323 | 131 |
| 2024Q1 | 2024Q2-Q3 | 4,227 | **268** | 70 | 198 | 3,503 | 407 | 49 |
| 2024Q2 | 2024Q3-Q4 | 4,491 | **282** | 64 | 218 | 3,578 | 569 | 62 |
| 2024Q3 | 2024Q4-2025Q1 | 4,776 | **294** | 74 | 220 | 3,796 | 600 | 86 |
| 2024Q4 | 2025Q1-Q2 | 4,852 | **266** | 70 | 196 | 4,061 | 454 | 71 |
| 2025Q1 | 2025Q2-Q3 | 5,167 | **284** | 70 | 214 | 4,370 | 469 | 44 |
| 2025Q2 | 2025Q3-Q4 | 5,404 | **300** | 91 | 209 | 4,525 | 540 | 39 |
| 2025Q3 | 2025Q4-2026Q1 | 5,782 | **321** | 103 | 218 | 4,746 | 636 | 79 |
| 2025Q4 | 2026Q1-Q2 | 5,838 | **300** | 87 | 213 | 4,869 | 598 | 71 |
| **All** | | 51,520 | **2,995** | 776 | 2,219 | 42,432 | 5,352 | 741 |

45,427 borrower-quarters are labelled (an event, or held and accruing through the window); 6.6%
have a credit event within two quarters.

- **By type** (a borrower can have several): 776 non-accrual, 2,339 PIK switch, 104 write-down,
  14 debt-for-equity. 2,121 of the 2,995 events are a PIK switch and nothing else.
- **Labelled 0:** of the 42,432 with no event, 5,298 had a spread step-up or a maturity extension
  in the window.
- **No label:** the 5,352 that left the books without an event are split into the four exit
  buckets in section 2.

**The two kinds of event look different beforehand** (`event_profile.csv`, `event_profile.png`).
Borrowers that enter non-accrual are marked at a median of 96 four quarters before, 86 the quarter
before and 73 in the quarter itself. Borrowers with a concession only stay at 97 to 98 throughout,
close to the 99 of borrowers with no event. Two thirds of the concession-only events come from
borrowers marked at 95 or above, against one third of the non-accruals.

Windows overlap (a borrower's quarters t and t + 1 share a quarter), so standard errors are
clustered and intervals resampled by borrower.

## 2. Term changes and exits

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

### Term changes (`restructuring_events.csv`, `restructuring_vs_nonaccrual.csv`)

224,672 loans were seen in two consecutive filings (2023Q2-2026Q2):

| Term change | In the outcome | Loans, primary | Loans, second cutoff | Accruing borrowers with it at t | Entered non-accrual within two quarters |
|---|---|---|---|---|---|
| PIK switch | **yes** | 3,789 | 3,031 | 1,230 | **7.2%** |
| Replaced by equity | **yes** | 32 | 32 | 4 | 0 of 4 |
| Write-down | **yes** | 145 | 55 | 46 | 1 of 46 |
| Spread step-up | no | 3,543 | 1,390 | 1,256 | 2.9% |
| Maturity extension | no | 7,099 | 5,805 | 2,367 | 1.6% |
| **Any** | | 12,861 | 9,453 | 4,217 | 3.1% |
| None | | | | 41,148 | 1.6% |

- **A PIK switch is the term change that precedes non-accrual,** at 4.5 times the rate of
  borrowers with no term change.
- **A maturity extension says nothing:** extended borrowers enter non-accrual at the same rate
  as the rest, and extensions are the largest type.
- **Swaps and write-downs are rare in the data:** most BDCs' schedules as parsed carry few equity
  rows, and a write-off usually takes the loan out of the schedule (an impaired exit), not down
  in it.

### Exits (`exits_unlabelled.csv`)

Two different things leave. A **loan exit** is one BDC's loan disappearing from its schedule; the
borrower may still be held, by that BDC or another, and then it stays in the sample and gets a
label. A **borrower exit** is a borrower leaving every BDC's books. The 5,352 borrower-quarters in
section 1 are borrower exits without an event. None is dropped: each goes in one bucket, from its
loans that left in the exit quarter (impaired if any was, repaid if all were; the loan rules are in
the backup below).

| Bucket | Exits, primary | Share | Second cutoff | Share |
|---|---|---|---|---|
| Repaid | 4,045 | 76% | 3,615 | 68% |
| Refinanced | 0 | | 0 | |
| Impaired | 202 | 4% | 397 | 7% |
| Unresolved | 1,105 | 21% | 1,340 | 25% |

- **Refinanced is zero for borrowers by construction.** Under the sponsor's rule a refinanced
  loan's borrower is still on some BDC's books, so it is not a borrower exit. Refinancing shows up
  at the loan level: 56% of loan exits (backup).
- **Impaired means a probable loss, not a measured one.** All 202 are impaired because their loans
  left marked at 80 or below (median 67). The schedule does not report what was recovered, so no
  exit can be confirmed as a realized loss, and none enters the outcome. The scorecard flags them
  anyway: in the test quarter it put all 16 impaired exits in its riskiest tenth, against 9 of
  the 479 repaid (section 4). This is partly by construction, since a low mark is its main sign.
- **The unresolved fall into three groups:** the lender skipped its next filing (443), the last
  mark was between 80 and 95 or missing (336), or no loan could be traced to the exit quarter
  (326).
- An exit is counted once per window it falls in, so the 202 borrower-quarters are 118 separate
  exits.

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

Two filters guard against filing artifacts in the term changes. Both were added after looking at
examples, and both are the same at either cutoff. A spread step-up counts only if the all-in rate
rose with it (by the step less 75 bp of base-rate movement), because a relabel from prime to SOFR
raises the reported spread by about three points while the all-in rate falls. A write-down counts
only on a loan already marked at least that far below its cost, which is what a realized loss
looks like; a revolver paid down at par also cuts cost while par (the commitment) stays.

## 3. Which signs are kept (`sign_selection.csv`)

Each sign is added alone to the baseline (mark and its change) and kept if it adds at least half
a point of out-of-sample ROC-AUC on the development quarters (signals 2024Q2-2025Q1, 1,126
events).

| Model | ROC-AUC | Gain over the baseline [95% CI] | Kept |
|---|---|---|---|
| Baseline: mark and its change | 0.714 | | |
| + PIK switch | 0.744 | +0.029 [0.019, 0.040] | **yes** |
| + fall over the past year | 0.729 | +0.014 [0.003, 0.025] | **yes** |
| + valuation gap | 0.719 | +0.004 [-0.005, 0.013] | no |
| + crowding | 0.713 | -0.002 [-0.011, 0.007] | no |
| Scorecard: both kept signs | 0.753 | +0.039 [0.025, 0.052] | |

- Both kept signs' gains clear zero. The valuation gap falls just under the rule's half point.
- The mark alone scores 0.727, above the baseline: the quarter's change adds nothing by itself.
  It stays because the sponsor's baseline is the mark and its recent change.

**The fitted scorecard** (`coefficients.csv`, spec "scorecard"; `odds_ratios.png`). This fit
uses all 45,427 labelled borrower-quarters with outcomes through 2026Q2 (7,722 borrowers, 2,995
events). Coefficients are per point of par (0.01 of the mark), so they read directly off a
schedule. The CSV reports them per 10 points (the mark) and per 5 points (the two changes);
divide by 10 and by 5. Standard errors are clustered by borrower. The mark and the moves are
capped (Method).

| Term | Coefficient | SE | z | p | Odds ratio |
|---|---|---|---|---|---|
| Intercept | 4.13 | 0.55 | 7.5 | <0.001 | |
| Mark, per point of par | -0.073 | 0.006 | -13.0 | <0.001 | 0.93 |
| Change in the mark this quarter, per point | -0.012 | 0.007 | -1.8 | 0.067 | 0.99 |
| Fall over the past year, per point | 0.061 | 0.008 | 7.6 | <0.001 | 1.06 |
| Switched to PIK this quarter (0/1) | 1.87 | 0.085 | 22.1 | <0.001 | 6.48 |

McFadden pseudo R² 0.11; ROC-AUC 0.744 on the rows it was fitted on. Per 10 points lower the
mark multiplies the odds by 2.07 (= 1 / 0.482), and per 5 points of fall over the year by 1.36.
The 30-second table rounds these.

## 4. Out-of-sample performance (`oos_performance.csv`, `oos_score_deciles.png`)

Fitted once on outcomes through 2025Q3 (signals up to 2025Q1, 2,074 events); tested on the 5,067
borrowers accruing in 2025Q3, of which 321 had a credit event in 2025Q4-2026Q1.

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Events caught in the riskiest 10% | in the riskiest 20% |
|---|---|---|---|---|
| **Scorecard** | **0.759 [0.73, 0.79]** | **0.231 [0.20, 0.27]** | **136 of 321** | **207 of 321** |
| Mark and its change | 0.746 [0.72, 0.78] | 0.223 [0.19, 0.27] | 129 of 321 | 193 of 321 |
| Mark alone | 0.739 [0.71, 0.77] | 0.209 [0.18, 0.25] | 133 of 321 | 195 of 321 |
| Equal-weighted ranking of the same signs | 0.741 [0.71, 0.77] | 0.218 [0.18, 0.26] | 125 of 321 | 179 of 321 |
| Random ranking | 0.50 | 0.063 (the event rate) | 32 of 321 | 64 of 321 |

PR-AUC scores only how well events are found; its random baseline is the event rate, not 0.5. The
scorecard's 0.23 is about 3.6 times random.

| Score decile | Realized event rate [95% CI] | Predicted | Realized ÷ predicted [95% CI] |
|---|---|---|---|
| 10 (riskiest) | 26.9% [23.0, 30.0] | 26.8% | 1.00 [0.86, 1.14] |
| 9 | 14.0% [11.1, 17.1] | 6.9% | 2.02 [1.59, 2.48] |
| 1-8 | 2.8% | 4.4% | 0.64 [0.54, 0.76] |
| All | 6.3% [5.7, 7.0] | 6.9% | **0.92 [0.84, 1.02]** |

Deciles 1-8 one by one (about 507 borrowers each), events and realized rate, from decile 8 down
to 1: 23 (4.5%), 17 (3.4%), 11 (2.2%), 8 (1.6%), 11 (2.2%), 15 (3.0%), 12 (2.4%), 17 (3.4%).
Each is predicted at 3.8% to 5.1%.

Realized ÷ predicted is 1 when the model predicts the right number of events; its interval, like
the others, resamples borrowers with the predictions held fixed (`oos_deciles.csv`).

- **The scorecard is a systematic ranking, not a large gain over the mark.** It beats the mark
  alone by 0.020 [0.002, 0.041] and the mark and its change by 0.013 [-0.004, 0.034].
- **Fitted weights vs equal weights:** +0.018 [0.001, 0.038].
- **Calibration: right in total, too flat in the middle.** The total number of events and the
  riskiest tenth are predicted correctly. The ninth decile had twice the predicted events and the
  bottom eight a third fewer. Use the ranking; outside the riskiest tenth the probabilities are
  rough.
- **The risk sits in the top fifth:** 64% of the events, at 27% and 14% event rates against 3%
  below.
- **No overfitting:** ROC-AUC is 0.745 on the training quarters and 0.759 on the test, and 0.73
  to 0.77 in each of seven quarters scored one at a time, each by a model refitted on the windows
  closed by then (`rolling_by_quarter.csv`, `rolling_auc_by_quarter.png`).

**Every borrower of the test quarter, labelled or not** (`oos_ranks_by_outcome.csv`). The test
scores the 5,067 labelled borrowers, 88% of the 5,782 accruing in 2025Q3. Ranking all of them:

| What happened next | Borrowers | In the riskiest tenth | In the riskiest fifth | ROC-AUC against "no event" |
|---|---|---|---|---|
| Credit event: non-accrual | 103 | **65 (63%)** | 79 (77%) | **0.82** |
| Credit event: concession only | 218 | 72 (33%) | 130 (60%) | 0.73 |
| No credit event | 4,746 | 379 (8%) | 824 (17%) | |
| Left the books: impaired | 16 | **16 (100%)** | 16 (100%) | 0.98 |
| Left the books: unresolved | 141 | 33 (23%) | 65 (46%) | 0.61 |
| Left the books: repaid | 479 | 9 (2%) | 30 (6%) | 0.39 |
| Status unknown | 79 | 4 (5%) | 12 (15%) | 0.37 |

- **Non-accrual is found as well as before** the outcome was widened (0.82 here; 0.81 for the
  earlier scorecard fitted on non-accrual alone).
- **Concessions are found less well.** They are mostly PIK switches on loans marked near par,
  where the mark has little to say.
- **Unresolved exits rank between repaid and impaired,** as a mix of the two would.

## 5. The scorecard for 2026Q2 (`scorecard_2026Q2.csv`)

All 6,267 borrowers accruing in 2026Q2, ranked, with their probability and the signs behind it.
About 440 events are expected in 2026Q3-Q4, 180 of them in the riskiest tenth. At the top: Rural
Sourcing Holdings (88%: marked at 74, down 12 points this quarter and 25 over the year, just
switched to PIK). Once 2026Q3 is loaded, a rerun runs the blind test: a fit on outcomes through
2026Q1, scored on the borrowers of 2026Q1 and their events in 2026Q2-Q3
(`reserved_test_2026Q3.csv`); this list itself is scored once 2026Q4 is filed.

## Weaknesses and limitations

Reviewed on 2026-11-17 ahead of presenting. The first five are the questions to expect; each says
how to present it and what would settle it.

1. **Predicting PIK with PIK.** 2,121 of the 2,995 events are a PIK switch alone, and a PIK switch
   is the strongest sign (6.5x). 31% of borrowers that just switched have another PIK event within
   two quarters, against 4% of the rest. Part of that is one restructuring reported in stages: at
   the primary cutoff every 10-point rise in a loan's PIK share counts as a new switch. *Not yet
   checked:* rerun with the PIK event at the second cutoff (cash to PIK only, a loan's first
   switch) and see whether the PIK sign and the 0.76 hold. If they do not, the outcome should use
   the stricter definition.
2. **A small gain over the mark.** 0.759 against 0.739 for the mark alone, +0.020 [0.002, 0.041].
   *Present it as* a systematic ranking in which the mark does most of the work, not as a model
   that beats simple rules.
3. **No blind test in the data used.** The 2025Q3 test was seen before the sign rule, the caps and
   the outcome were settled (see "How the sponsor's definitions are applied"), and with the data
   held at 2023Q1-2026Q2 there is no other. *Present it* as an out-of-sample test by time, not a
   blind one. *Settled by* the 2026Q3 test (section 5).
4. **Probabilities are rough outside the riskiest tenth.** The total and the riskiest tenth are
   right; the ninth decile had twice the predicted events and the bottom eight a third fewer
   (section 4). *Present* deciles and ranks, not probabilities.
5. **Impaired exits are not evidence.** All 16 impaired exits of the test quarter fall in the
   riskiest tenth, but an exit is impaired because its last mark was low, and a low mark is the
   main sign. *Do not use them as validation.*

Further limitations:

- **Concessions are harder to see coming** than non-accruals (0.73 against 0.82): most are PIK
  switches on loans still marked near par.
- **Sudden events are missed:** 185 of the 321 test events were outside the riskiest tenth,
  including 38 of the 103 non-accruals.
- **Few non-accruals:** 776 in 11 windows, 103 in the test.
- **One horizon:** two quarters only (see "How the sponsor's definitions are applied").
- **Losses at exit are not in the outcome:** a loan sold or written off leaves the schedule
  without a recovery figure. It is counted as an impaired exit when it left marked well below par
  (202 exits, 397 at the second cutoff; section 2).
- **Term changes are inferred,** from loans the matcher links across quarters. A change that the
  matcher reads as a new loan counts as a refinancing instead.
- **Non-accrual is missing** where filings report it only in HTML, and understated for some BDCs.

## 6. Look-ahead check (`scripts/check_look_ahead.py`, `look_ahead_check.csv`)

The check cuts `FINAL_DATASET.csv` after a quarter T, rebuilds the whole panel from it as if the
data ended there, and compares it with the panel built from all the data. Nothing known at T may
change when later quarters are added.

| Cut after | Signs, sample flags, non-accrual status at quarters <= T | Outcomes of windows closed by T | Scorecard scores for T |
|---|---|---|---|
| 2025Q1 (end of development) | 59,202 rows, all identical | 43,690 rows, all identical | 5,167 scores, identical |
| 2025Q3 (the test quarter) | 76,345 rows; 1 debt-for-equity flag differs | 59,202 rows; labels identical, 1 event quarter differs | 5,782 scores, identical |

**What the two differences are.** The chain linker bridges a loan's gap of up to two missed
filings, so whether a loan ended, and was replaced by equity, is settled only two quarters later.
In the cut data FS Whitewater's loan seems to end in 2025Q2 and be swapped for equity in 2025Q3.
In the full data it comes back, and the swap falls a quarter later. This touches exits and the
14 equity swaps only, never a sign. No label the model was fitted on changed.

**Checked by reading the code:**

- Every sign looks back only (same-loan changes from t - 1 to t, the drawdown over t - 4 to t, a
  PIK switch against the previous filing).
- The valuation gap and crowding use one quarter's filings.
- Non-accrual flags come from each filing's own footnotes.
- No value is filled in from another quarter.
- Each fit's scaler and weights are fitted on its training rows only.
- Each BDC has one filing per quarter, the original: no later amendment replaces it.
- The 25 BDCs that stopped filing stay in for the quarters they filed (no survivorship).

**Not covered by the check:**

- **Labels made once over all quarters.** Borrower names, seniority and sectors were labelled
  over all quarters before FINAL_DATASET, and the BDC-to-manager-family map is today's. These
  decide which rows are one borrower; they do not move a mark or a flag.
- **Choices made after seeing the test** (weakness 3).

---

## Method

Code in `src/9_model/` (`model_panel.py`: panel, outcome and exits; `credit_events.py`: term
changes and loan exits; `model_fit.py`: logit and metrics; `model_analysis.py`: tables;
`model_charts.py`: charts). Outputs in `outputs/model/`.

- **Unit:** a borrower in a quarter, every BDC's loans pooled. Kept if it is not a CLO vehicle,
  has a mark on at least half its par, holds $1m+ of debt, and its loans were priced the quarter
  before.
- **Outcome** (`y_event`): for a borrower accruing at t, 1 if at t + 1 or t + 2 any BDC flags its
  debt non-accrual or any lender's loan to it has a concession (`credit_events.CONCESSIONS`: PIK
  switch, debt-for-equity, write-down, at the primary cutoffs); 0 if it is held and known accruing
  in both quarters with no concession; no label if it leaves every BDC's books before the window
  closes without an event, its status is unknown, or the window runs past the data. A borrower
  that had a concession at t is still at risk.
- **Exits and term changes:** section 2; the cutoffs and filters are constants at the top of
  `credit_events.py`.
- **Mark:** fair value / par (fair value / cost for debt without par), capped below at 70.
  **Change:** on the same loans only, capped at 20 points. **PIK switch:** a loan that paid cash
  starts paying in kind, or its PIK share rises 10+ points. **Fall over the past year:** largest
  peak-to-trough fall of the same-loan mark over four quarters, capped at 20.
- **Tested and dropped:** the valuation gap (highest minus lowest lender family's mark on the
  same loan) and crowding (number of lender families).
- **The caps make no difference to missed events** (`caps_comparison.csv`). Events outside each
  quarter's riskiest 5 / 10 / 20 / 30% of borrowers, capped against uncapped: test (321 events)
  242 / 185 / 114 / 91 against 241 / 185 / 113 / 91; development, rolling (1,126 events) 810 /
  640 / 436 / 346 against 816 / 640 / 442 / 353. ROC-AUC is 0.759 either way. The caps change 80
  test borrowers (19 events) and hold the highest probability at 93% instead of 98%.
- **Stages:** a quarter's window closes two quarters later, so each fit uses only windows closed
  before the quarter it scores. Choosing signs: rolling, each of signals 2024Q2-2025Q1 scored by
  a fit on the windows closed by then; the four rounds' predictions are pooled into one ROC-AUC
  per model. Test: fit on outcomes through 2025Q3, scored on the borrowers of 2025Q3 (events
  2025Q4-2026Q1). Blind test (not in the data used; needs 2026Q3): fit on outcomes through
  2026Q1, scored on the borrowers of 2026Q1 (events 2026Q2-Q3). Scorecard: fit on outcomes through
  2026Q2 (signals 2023Q2-2025Q4), scored on the borrowers of 2026Q2.

  | Stage | Fitted on (signal quarters) | Rows fitted | Events fitted | Scored |
  |---|---|---|---|---|
  | Choosing signs, first round | 2023Q2-2023Q4 | 9,664 | 680 | borrowers of 2024Q2 |
  | Choosing signs, last round | 2023Q2-2024Q3 | 21,385 | 1,524 | borrowers of 2025Q1 |
  | Test | 2023Q2-2025Q1 | 30,366 | 2,074 | borrowers of 2025Q3 (5,067; 321 events) |
  | Scorecard | 2023Q2-2025Q4 | 45,427 | 2,995 | borrowers of 2026Q2 (6,267) |
- **Intervals:** borrower-resampled bootstrap for every performance measure and rate (AUC,
  recall, event rates), since a borrower appears in several quarters and one event can fall in
  two overlapping windows; borrower-clustered standard errors for coefficients.

Further checks are written by the same run:

- **Each sign alone:** `event_rates_by_bucket.csv` and `event_rate_by_predictor.png` (the four
  signs and the two dropped), `pik_switch_hypotheses.csv`, `coefficients.csv`, `odds_ratios.png`.
- **Alternative signs** (`robustness_summary.csv`): test ROC-AUC 0.72 to 0.77 across variants of
  the signs, and 0.77 on the 125 events among borrowers with 2+ lender families.
- **Other outcomes, labelled as such in the same file:** non-accrual alone (the outcome until
  2026-11-16) 0.82 on 103 events, or 0.81 with the five signs the rule kept for it; non-accrual
  or any of the five term changes 0.56 on 892 events.
