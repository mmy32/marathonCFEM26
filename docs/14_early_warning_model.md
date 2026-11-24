# Early-Warning Scorecard: Which Borrowers Have a Credit Event Within Two Quarters

## In 30 seconds

Lenders mark a loan down before they stop booking its interest or change its terms. The scorecard
is a logistic regression on five signs from each quarter's filings:

| Warning sign | Odds of a credit event within two quarters |
|---|---|
| It just switched to paying interest in kind (PIK) | about **4.3x** |
| It switched to PIK in the three quarters before | about **3.9x** |
| Its loans are marked low | each 10 points lower: about **2.0x** |
| The mark fell over the past year | each further 5 points: about **1.3x** |
| The mark fell this quarter | each 5-point fall: about **1.1x** |

The outcome is **a credit event within the next two quarters**: at t + 1 or t + 2 some BDC flags
the borrower's debt non-accrual, or a lender makes a concession on one of its loans (switches its
cash interest to PIK, swaps the debt for equity, or writes down its cost). A markdown is an input,
never the outcome.

**Out of sample** (fitted on outcomes through 2025Q4, tested on the borrowers accruing in 2025Q4
and their events in 2026Q1-Q2): ROC-AUC **0.75** [0.72, 0.79]; the riskiest tenth of borrowers
held 118 of the 283 events and the riskiest fifth 171. The mark alone scores 0.73; the scorecard
beats it by **+0.022** [0.002, 0.041] in ROC-AUC and **+0.062** [0.032, 0.095] in PR-AUC. On the
five development quarters, each scored out of sample, it scores 0.783 against 0.723.

**Two kinds of event, found unequally.** The same scorecard finds the non-accruals well (ROC-AUC
0.84; 57 of 87 in the riskiest tenth) and the concessions less well (0.71; 61 of 196): most
concessions are PIK switches on loans still marked near par. The second PIK sign mostly helps with
these: a borrower that switched to PIK is likely to switch again (weakness 1).

**Deliverables:** `scorecard_2026Q2.csv` (every accruing borrower ranked), `oos_performance.csv`
(the performance table) and `scripts/run_model.py`, which rebuilds both from `FINAL_DATASET.csv`
(itself built from the filings by `src/1_data_processing`).

```
python scripts/run_model.py            # about 6 minutes
python scripts/run_model.py --cached   # reuse the saved panel
python tests/test_model.py             # 42 tests (and tests/test_pik_metrics.py, 3)
python scripts/check_look_ahead.py     # look-ahead check (section 6), about 8 minutes
```

## How the sponsor's definitions are applied

- **A credit event is non-accrual or a restructuring in which the lender gives something up.**
  The filings have no restructuring field, so a restructuring is read from a change in terms on
  the same loan at the same BDC from one quarter to the next. The sponsor lists five such changes
  (section 2). Three are concessions and are in the outcome: a switch to PIK (cash interest
  deferred), a debt-for-equity swap, a cost write-down.
- **A PIK switch is read from the schedule, with two filing artifacts removed** (2026-11-23,
  `pik_metrics.pik_switch_clean`): a rise in the PIK share counts only if the PIK rate itself
  rose, not when the total coupon fell (as when the cash part goes missing from a filing); and
  PIK that comes back after one filing at zero, with the total coupon unchanged, is the PIK split
  missing from that filing, not a new switch. This removed 301 of 3,789 loan-level switches and
  207 of 2,995 event rows. The same flag gives the two PIK signs.
- **A spread step-up and a maturity extension are counted beside the outcome, not in it.** A
  step-up pays the lender more, and most extensions are routine amend-and-extend deals. With all
  five in the outcome there would be 8,345 events instead of 2,788, and 3,876 of them would be an
  extension alone on loans marked at par (median 99.9, against 99.5 for borrowers with no event).
  That version is a labelled robustness row (ROC-AUC 0.58). This choice was made on 2026-11-16
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
  rebuilt from data cut after 2025Q2 or 2025Q4 (the end of development and the test quarter)
  gives the same signs, sample, outcomes and scores as the full build.
- **The 2025Q4 test is not untouched.** The rule for keeping signs and the caps were fixed on
  2026-11-13 after an earlier version of the test had been seen. The outcome was widened from
  non-accrual alone on 2026-11-16, after that outcome's test result (0.81) had been seen; the
  alternatives were compared on the development quarters only, and the rule and the caps were
  applied unchanged. The split itself moved on 2026-11-18, from development up to 2025Q1 and a
  2025Q3 test to development up to 2025Q2 and a 2025Q4 test, after 2025Q4 had been seen in the
  rolling check; the rule keeps the same signs either way. The past-year PIK switch was added as
  a candidate on 2026-11-23, after the test had been seen: a check of why the ninth score decile
  ran above its prediction found that borrowers with an earlier PIK switch had events at about
  four times the predicted rate. The rule, applied unchanged, keeps it. The PIK clean-up and a
  check of recency weighting (section 3, not adopted) followed on the same day, after a code audit
  had looked at the test quarter again. The data stay at 2023Q1-2026Q2 for
  now, so this is the final test. A blind
  test needs 2026Q3: a rerun with it loaded scores the borrowers of 2026Q1 on their events in
  2026Q2-Q3, and nothing in the model should change after that run.

---

## 1. Events per quarter (`event_counts.csv`)

| Signals in | Window | Borrowers at risk | **Credit events** | with non-accrual | concession only | No event | Left the books (no label) | Status unknown |
|---|---|---|---|---|---|---|---|---|
| 2023Q2 | 2023Q3-Q4 | 3,399 | **205** | 48 | 157 | 2,756 | 377 | 61 |
| 2023Q3 | 2023Q4-2024Q1 | 3,584 | **198** | 39 | 159 | 2,958 | 379 | 49 |
| 2023Q4 | 2024Q1-Q2 | 4,000 | **214** | 60 | 154 | 3,331 | 324 | 131 |
| 2024Q1 | 2024Q2-Q3 | 4,227 | **243** | 70 | 173 | 3,528 | 407 | 49 |
| 2024Q2 | 2024Q3-Q4 | 4,491 | **268** | 64 | 204 | 3,589 | 572 | 62 |
| 2024Q3 | 2024Q4-2025Q1 | 4,776 | **282** | 74 | 208 | 3,807 | 600 | 87 |
| 2024Q4 | 2025Q1-Q2 | 4,852 | **247** | 70 | 177 | 4,079 | 455 | 71 |
| 2025Q1 | 2025Q2-Q3 | 5,167 | **259** | 70 | 189 | 4,395 | 469 | 44 |
| 2025Q2 | 2025Q3-Q4 | 5,404 | **287** | 91 | 196 | 4,537 | 541 | 39 |
| 2025Q3 | 2025Q4-2026Q1 | 5,782 | **302** | 103 | 199 | 4,765 | 636 | 79 |
| 2025Q4 | 2026Q1-Q2 | 5,838 | **283** | 87 | 196 | 4,886 | 598 | 71 |
| **All** | | 51,520 | **2,788** | 776 | 2,012 | 42,631 | 5,358 | 743 |

45,419 borrower-quarters are labelled (an event, or held and accruing through the window); 6.1%
have a credit event within two quarters.

- **By type** (a borrower can have several): 776 non-accrual, 2,117 PIK switch, 104 write-down,
  14 debt-for-equity. 1,914 of the 2,788 events are a PIK switch and nothing else.
- **What lies behind the event rows** (`outcome_breakdown.csv`, each row in the first group that
  applies): 776 non-accrual (200 of them also with a concession), 1,929 PIK switch without
  non-accrual, 72 write-down only, 11 debt-for-equity only. An event counts in the rows of both
  quarters before it, so the 2,788 rows are 1,140 borrowers, 447 non-accrual entries and 1,405
  separate concessions.
- **Labelled 0:** of the 42,631 with no event, 5,332 had a spread step-up or a maturity extension
  in the window (4,186 an extension, 1,627 a step-up, 481 both).
- **No label:** the 5,358 that left the books without an event are split into the four exit
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
(`src/9_prediction/credit_events.py`). A "loan" is one BDC's tranche followed over time (the tranche
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
| PIK switch | PIK starts, or the PIK rate rises with its share of the coupon up 10+ points | PIK starts on a cash-pay loan |

At both cutoffs, PIK that comes back after one filing at zero with the total coupon unchanged (within
25 bp) is not a switch: the filing in between dropped the PIK split (added 2026-11-23, with the rule
that the PIK rate itself must rise).

### Term changes (`restructuring_events.csv`, `restructuring_vs_nonaccrual.csv`)

224,672 loans were seen in two consecutive filings (2023Q2-2026Q2):

| Term change | In the outcome | Loans, primary | Loans, second cutoff | Accruing borrowers with it at t | Entered non-accrual within two quarters |
|---|---|---|---|---|---|
| PIK switch | **yes** | 3,488 | 2,864 | 1,122 | **7.3%** |
| Replaced by equity | **yes** | 32 | 32 | 4 | 0 of 4 |
| Write-down | **yes** | 145 | 55 | 46 | 1 of 46 |
| Spread step-up | no | 3,543 | 1,390 | 1,256 | 2.9% |
| Maturity extension | no | 7,099 | 5,805 | 2,367 | 1.6% |
| **Any** | | 12,577 | 9,293 | 4,120 | 3.1% |
| None | | | | 41,245 | 1.6% |

- **A PIK switch is the term change that precedes non-accrual,** at 4.6 times the rate of
  borrowers with no term change.
- **A maturity extension says nothing:** extended borrowers enter non-accrual at the same rate
  as the rest, and extensions are the largest type.
- **Swaps and write-downs are rare in the data:** most BDCs' schedules as parsed carry few equity
  rows, and a write-off usually takes the loan out of the schedule (an impaired exit), not down
  in it.

### Exits (`exits_unlabelled.csv`)

Two different things leave. A **loan exit** is one BDC's loan disappearing from its schedule; the
borrower may still be held, by that BDC or another, and then it stays in the sample and gets a
label. A **borrower exit** is a borrower leaving every BDC's books. The 5,358 borrower-quarters in
section 1 are borrower exits without an event. None is dropped: each goes in one bucket, from its
loans that left in the exit quarter (impaired if any was, repaid if all were; the loan rules are in
the backup below).

| Bucket | Exits, primary | Share | Second cutoff | Share |
|---|---|---|---|---|
| Repaid | 4,048 | 76% | 3,617 | 68% |
| Refinanced | 0 | | 0 | |
| Impaired | 203 | 4% | 399 | 7% |
| Unresolved | 1,107 | 21% | 1,342 | 25% |

- **Refinanced is zero for borrowers by construction.** Under the sponsor's rule a refinanced
  loan's borrower is still on some BDC's books, so it is not a borrower exit. Refinancing shows up
  at the loan level: 56% of loan exits (backup).
- **Impaired means a probable loss, not a measured one.** All 203 are impaired because their loans
  left marked at 80 or below (median of the lowest, 66). The schedule does not report what was
  recovered, so no exit can be confirmed as a realized loss, and none enters the outcome. The
  scorecard flags them anyway: in the test quarter it put all 20 impaired exits in its riskiest
  tenth, against 15 of the 468 repaid (section 4). This is partly by construction, since a low
  mark is its main sign.
- **The unresolved fall into three groups:** the lender skipped its next filing (443), the last
  mark was between 80 and 95 or missing (337), or no loan could be traced to the exit quarter
  (327).
- An exit is counted once per window it falls in, so the 203 borrower-quarters are 119 separate
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
a point of out-of-sample ROC-AUC on the development quarters (signals 2024Q2-2025Q2, 1,343
events).

| Model | ROC-AUC | Gain over the baseline [95% CI] | Kept |
|---|---|---|---|
| Baseline: mark and its change | 0.715 | | |
| + PIK switch | 0.741 | +0.026 [0.017, 0.035] | **yes** |
| + fall over the past year | 0.729 | +0.015 [0.005, 0.024] | **yes** |
| + PIK switch in the past year (added 2026-11-23) | 0.760 | +0.046 [0.033, 0.059] | **yes** |
| + valuation gap | 0.717 | +0.003 [-0.005, 0.010] | no |
| + crowding | 0.714 | -0.001 [-0.009, 0.008] | no |
| Scorecard: the three kept signs | 0.783 | +0.069 [0.054, 0.084] | |

- All three kept signs' gains clear zero. The valuation gap falls just under the rule's half point.
- PR-AUC, reported beside the rule, ranks the five the same way: +0.037 [0.020, 0.051] for the
  past-year PIK switch, +0.027 [0.015, 0.039] for the PIK switch, +0.007 [-0.001, 0.014] for the
  fall over the past year, +0.001 [-0.005, 0.008] for the valuation gap and -0.001 [-0.004,
  0.002] for crowding.
- The rule kept the same signs before the PIK clean-up, and keeps the PIK switch and the fall
  over the past year when the development quarters end in 2025Q1 instead (the past-year PIK
  switch was not checked that way).
- **The past-year PIK switch came after the test** ("How the sponsor's definitions are
  applied"). Its gain is mostly on concession-only events: +0.043 ROC-AUC for those against
  +0.008 for non-accruals, and +0.005 [-0.001, 0.011] with non-accrual alone as the outcome
  (weakness 1).
- The mark alone scores 0.723, above the baseline: the quarter's change adds nothing by itself.
  It stays because the sponsor's baseline is the mark and its recent change.

**Interactions were tested and left out.** Each of the 10 pairwise interactions of the five
signs was added to the scorecard and judged by the same rule on the same quarters. None adds the
half point: the best adds +0.0006 (the mark times the past-year PIK switch), and all ten together
cost 0.004 [0.002, 0.006] of ROC-AUC and 0.011 of PR-AUC. Most are significant in-sample. They all
say the same thing: the signs overlap, so once one fires the next adds less. For example, a PIK
switch this quarter and one in the past year together count far less than the two apart. With
45,000 rows such a bend is significant, but it reorders few borrowers. The six pairs of the
four-sign model (before the past-year PIK switch) gave the same answer.

**Weighting recent quarters more was tested and left out** (2026-11-23, `config.HALF_LIFE`). Each
fit weighted its training quarters by 0.5 ^ (age / half-life), for half-lives of 8, 4 and 2
quarters, judged by the same rule on the same quarters. None changes the ranking: the best gains
+0.0001 ROC-AUC (half-life 4), and the two quarters after development move by 0.0004 at most. They
only lift the predicted level toward the realized one a little: realized ÷ predicted 0.89 with
no weighting, 0.91 at a half-life of 2 (development), and 0.87 against 0.93 in 2025Q3-Q4.

**The fitted scorecard** (`coefficients.csv`, spec "scorecard"; `odds_ratios.png`). This fit
uses all 45,419 labelled borrower-quarters with outcomes through 2026Q2 (7,722 borrowers, 2,788
events). Coefficients are per point of par (0.01 of the mark), so they read directly off a
schedule. The CSV reports them per 10 points (the mark) and per 5 points (the two changes);
divide by 10 and by 5. Standard errors are clustered by borrower. The mark and the moves are
capped (Method).

| Term | Coefficient | SE | z | p | Odds ratio |
|---|---|---|---|---|---|
| Intercept | 3.44 | 0.55 | 6.3 | <0.001 | |
| Mark, per point of par | -0.068 | 0.006 | -12.1 | <0.001 | 0.93 |
| Fall in the mark this quarter, per point (a fall is positive) | 0.026 | 0.008 | 3.4 | <0.001 | 1.03 |
| Fall over the past year, per point | 0.056 | 0.008 | 6.6 | <0.001 | 1.06 |
| Switched to PIK this quarter (0/1) | 1.47 | 0.079 | 18.5 | <0.001 | 4.33 |
| Switched to PIK in the three quarters before (0/1) | 1.37 | 0.081 | 16.9 | <0.001 | 3.94 |

McFadden pseudo R² 0.13; on the rows it was fitted on, ROC-AUC 0.768 [0.753, 0.784] and PR-AUC
0.251 [0.231, 0.274] (borrower resamples, the fit held fixed). Per 10 points lower the
mark multiplies the odds by 1.97 (= 1 / 0.508), and per 5 points of fall over the year by 1.32.
The 30-second table rounds these. The four-sign scorecard (without the past-year PIK switch) is
the robustness row "without the past-year PIK switch": PIK switch 1.77 (5.9x), mark -0.072, test
ROC-AUC 0.732.

**Multicollinearity** (`predictor_correlations.csv`, `predictor_correlations.png`). On the same
rows, capped as in the fit, no VIF exceeds 1.76 (mark 1.71, fall this quarter 1.28, fall over the
past year 1.76, PIK switch 1.04, past-year PIK switch 1.05); the strongest correlation is the mark
with the fall over the past year (-0.62), then the two falls (+0.43); the two PIK signs correlate
+0.17. The quarter's fall moves with the other signs, not its precision: its standard error stays
at 0.007-0.008 per point in every fit, while its coefficient is 0.041 in the baseline, 0.011 once
the fall over the past year is added (p = 0.11; that window contains this quarter's fall), 0.018
with the PIK switch too (p = 0.009), and 0.026 with both PIK signs (p < 0.001).

## 4. Out-of-sample performance (`oos_performance.csv`, `oos_score_deciles.png`)

Fitted once on outcomes through 2025Q4 (signals up to 2025Q2, 35,183 rows, 2,203 events); tested
on the 5,169 borrowers accruing in 2025Q4, of which 283 had a credit event in 2026Q1-Q2.

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Events caught in the riskiest 10% | in the riskiest 20% |
|---|---|---|---|---|
| **Scorecard** | **0.753 [0.72, 0.79]** | **0.239 [0.20, 0.29]** | **118 of 283** | **171 of 283** |
| Mark and its change | 0.730 [0.70, 0.76] | 0.182 [0.15, 0.22] | 121 of 283 | 159 of 283 |
| Mark alone (the benchmark) | 0.731 [0.70, 0.76] | 0.177 [0.15, 0.21] | 116 of 283 | 159 of 283 |
| Random ranking | 0.50 | 0.055 (the event rate) | 28 of 283 | 57 of 283 |

PR-AUC scores only how well events are found; its random baseline is the event rate, not 0.5. The
scorecard's 0.24 is about 4.4 times random.

**The scorecard against the mark alone** (`oos_benchmarks.csv`, `oos_capture.csv`; paired 95%
intervals, 500 borrower resamples). The mark alone is a logistic regression on the mark level,
fitted and scored exactly as the scorecard is, so the gain is what the other four signs add:

| Window | Events | ROC-AUC gain | PR-AUC gain | Riskiest 20%, extra share of events |
|---|---|---|---|---|
| Test (2025Q4) | 283 | **+0.022 [0.002, 0.041]** | **+0.062 [0.032, 0.095]** | +4.2 pts [0.0, 8.3] |
| Development (5 quarters, rolling) | 1,343 | **+0.060 [0.045, 0.076]** | **+0.065 [0.047, 0.083]** | **+8.7 pts [6.0, 11.3]** |
| After development (2025Q3-Q4, rolling) | 585 | **+0.025 [0.008, 0.043]** | **+0.046 [0.025, 0.069]** | **+3.9 pts [0.7, 7.7]** |
| All seven quarters (rolling) | 1,928 | **+0.048 [0.036, 0.063]** | **+0.059 [0.042, 0.076]** | **+7.3 pts [4.6, 9.9]** |

**Quarter by quarter** (each scored by a fit on the windows closed by then; DeLong intervals), the
ROC-AUC gain over the mark alone is +0.052, +0.051, +0.057, +0.080, +0.061 in the five development
quarters, then +0.028 (2025Q3) and +0.022 (2025Q4, the test); every one clears zero. It is smaller
in the latest quarters: the gain comes mostly from the PIK signs, and in 2025Q4 PIK was a weaker
sign (borrowers with a PIK sign had events at 24%, against about 30% before) while non-accrual was
a larger share of events. One quarter can detect a ROC-AUC gain of about 0.023 with 80% power.

| Score decile | Realized event rate [95% CI] | Predicted | Realized ÷ predicted [95% CI] |
|---|---|---|---|
| 10 (riskiest) | 22.9% [19.4, 26.3] | 28.6% | 0.80 [0.68, 0.92] |
| 9 | 10.3% [7.7, 13.0] | 7.4% | 1.39 [1.05, 1.76] |
| 1-8 | 2.7% | 3.7% | 0.73 [0.60, 0.87] |
| All | 5.5% [4.9, 6.0] | 6.6% | **0.83 [0.75, 0.92]** |

Deciles 1-8 one by one (517 borrowers each), events and realized rate, from decile 8 down to 1:
21 (4.1%), 21 (4.1%), 13 (2.5%), 9 (1.7%), 14 (2.7%), 7 (1.4%), 13 (2.5%), 14 (2.7%). Each is
predicted at 3.2% to 4.4%.

Realized ÷ predicted is 1 when the model predicts the right number of events; its interval, like
the others, resamples borrowers with the predictions held fixed (`oos_deciles.csv`).

- **The scorecard is a systematic ranking with a modest gain over the mark.** On the test it beats
  the mark alone by 0.022 [0.002, 0.041] and the mark and its change by 0.023 [0.003, 0.044].
  On the five development quarters, each scored out of sample, it beats the mark alone by 0.060
  [0.045, 0.076] (`oos_benchmarks.csv`). Most of the gain is the two PIK signs.
- **Watch lists against the mark alone** (`oos_capture.csv`, `oos_list_disagreement.csv`): on the
  riskiest 20% the scorecard catches 171 events against 159 on the test and 886 against 769 on the
  development quarters. Where the two lists differ, the scorecard's own picks have the higher
  event rate: on the test's 20% list 9.8% against 4.9% (+4.9 points [0.6, 9.3]), on the
  development quarters' 16.5% against 6.2%. On the test's 10% list the two tie (16% each).
- **Calibration: 17% high in total.** The model expected 339 events and 283 happened (0.83 [0.75,
  0.92]); the riskiest tenth runs high too (0.80). The ninth decile had 1.39 times the predicted
  events and the bottom eight 0.73. The event rate eased from about 6.5% in 2023-24 to 5.5% in this
  window, and the fit pools the earlier quarters. Use the ranking, not the probabilities.
- **The risk sits in the top fifth:** 60% of the events, at 23% and 10% event rates against 3%
  below.
- **No overfitting:** ROC-AUC is 0.769 on the rows the test model was fitted on and 0.753 on the
  test, and 0.75 to 0.80 in each of seven quarters scored one at a time, each by a model refitted
  on the windows closed by then (`rolling_by_quarter.csv`, `rolling_auc_by_quarter.png`).
- **Calibration drifts with the event rate:** in those seven quarters realized ÷ predicted runs
  from 1.00 (2024Q2) to 0.78 (2025Q1). It is 0.91 [0.82, 1.01] in 2025Q3 and 0.83 [0.75, 0.92] in
  the test quarter (`rolling_by_quarter.csv`). Weighting recent quarters more lifts it a little
  but does not change the ranking (section 3).
- **Each watch list's share of events caught,** for the scorecard and the mark alone, with
  intervals, is in `oos_capture.csv` (for example the riskiest 20% of the test quarter: 60% [55,
  66] against 56% [51, 62]). `oos_score_deciles.png` sets the two side by side by score decile.

**Every borrower of the test quarter, labelled or not** (`oos_ranks_by_outcome.csv`). The test
scores the 5,169 labelled borrowers, 89% of the 5,838 accruing in 2025Q4. Ranking all of them:

| What happened next | Borrowers | In the riskiest tenth | In the riskiest fifth | ROC-AUC against "no event" |
|---|---|---|---|---|
| Credit event: non-accrual | 87 | **57 (66%)** | 68 (78%) | **0.84** |
| Credit event: concession only | 196 | 61 (31%) | 105 (54%) | 0.71 |
| No credit event | 4,886 | 402 (8%) | 879 (18%) | |
| Left the books: impaired | 20 | **20 (100%)** | 20 (100%) | 0.98 |
| Left the books: unresolved | 110 | 23 (21%) | 54 (49%) | 0.65 |
| Left the books: repaid | 468 | 15 (3%) | 31 (7%) | 0.37 |
| Status unknown | 71 | 5 (7%) | 10 (14%) | 0.38 |

- **Non-accrual is found as well as by a scorecard built for it** (0.84 here; 0.85 for the same
  five signs fitted on non-accrual alone, `robustness_summary.csv`).
- **Concessions are found less well.** They are mostly PIK switches on loans marked near par,
  where the mark has little to say.
- **Unresolved exits rank between repaid and impaired,** as a mix of the two would.

## 5. The scorecard for 2026Q2 (`scorecard_2026Q2.csv`)

All 6,267 borrowers accruing in 2026Q2, ranked, with their probability and the signs behind it.
About 410 events are expected in 2026Q3-Q4, 176 of them in the riskiest tenth. At the top: Rural
Sourcing Holdings (94%: marked at 74, down 12 points this quarter and 25 over the year, just
switched to PIK). Once 2026Q3 is loaded, a rerun runs the blind test: a fit on outcomes through
2026Q1, scored on the borrowers of 2026Q1 and their events in 2026Q2-Q3
(`reserved_test_2026Q3.csv`); this list itself is scored once 2026Q4 is filed.

## Weaknesses and limitations

Reviewed on 2026-11-17 ahead of presenting. The first five are the questions to expect; each says
how to present it and what would settle it.

1. **Predicting PIK with PIK.** 1,914 of the 2,788 events are a PIK switch alone, and the two PIK
   signs are the strongest (4.3x for a switch this quarter, 3.9x for one in the three quarters
   before). 27% of borrowers that just switched have another PIK event within two quarters,
   against 4% of the rest. Part of that is one restructuring reported in stages: at the primary
   cutoff every rise of the PIK rate that lifts its share 10+ points counts as a new switch, and
   most PIK events after an earlier switch are the same loan switching again. Filing artifacts
   that looked like switches were removed on 2026-11-23 (section 2). The past-year PIK switch
   leans on this most: on the development quarters it adds +0.043 ROC-AUC for concession-only
   events and +0.008 for non-accruals, and with non-accrual alone as the outcome it adds +0.005
   [-0.001, 0.011], under the rule. *Not yet checked:* rerun with the PIK event at the second
   cutoff (cash to PIK only, a loan's first switch) and see whether the PIK signs and the test
   ROC-AUC (0.75) hold. If they do not, the outcome should use the stricter definition.
2. **A modest gain over the mark, smaller in the latest quarters.** On the test 0.753 against
   0.731 for the mark alone, +0.022 [0.002, 0.041]; on the development quarters +0.060 [0.045,
   0.076]. Most of it is the two PIK signs, so it is mostly a gain on concessions (weakness 1), and
   it fell from +0.05 to +0.08 a quarter in 2024-25Q2 to about +0.02 in 2025Q3-Q4 as PIK became a
   weaker sign (section 4). *Present it as* a systematic ranking in which the mark and PIK history
   do most of the work, with the gain over the mark significant on the test and in every quarter.
3. **No blind test in the data used.** The 2025Q4 test was seen in the rolling check before the
   split was chosen, before the past-year PIK switch was added and before the PIK clean-up, and an
   earlier test had been seen before the sign rule, the caps and the outcome were settled (see
   "How the sponsor's definitions are applied"). With the data held at 2023Q1-2026Q2 there is no
   other. *Present it* as an out-of-sample test by time, not a blind one. *Settled by* the 2026Q3
   test (section 5).
4. **Probabilities run high.** In the test quarter the total is 17% high (0.83) and the riskiest
   tenth 20% high (0.80); the ninth decile had 1.39 times the predicted events and the bottom
   eight 0.73 (section 4). *Present* deciles and ranks, not probabilities.
5. **Impaired exits are not evidence.** All 20 impaired exits of the test quarter fall in the
   riskiest tenth, but an exit is impaired because its last mark was low, and a low mark is the
   main sign. *Do not use them as validation.*

Further limitations:

- **Concessions are harder to see coming** than non-accruals (0.71 against 0.84): most are PIK
  switches on loans still marked near par.
- **Sudden events are missed:** 165 of the 283 test events were outside the riskiest tenth,
  including 30 of the 87 non-accruals.
- **The past-year PIK switch looks back less far at the start of the data,** which begins in
  2023Q1: signals of 2023Q2-Q4 see one or two earlier quarters, not three. Every fit includes
  these quarters.
- **Few non-accruals:** 776 in 11 windows, 87 in the test.
- **One horizon:** two quarters only (see "How the sponsor's definitions are applied").
- **Losses at exit are not in the outcome:** a loan sold or written off leaves the schedule
  without a recovery figure. It is counted as an impaired exit when it left marked well below par
  (203 exits, 399 at the second cutoff; section 2).
- **Term changes are inferred,** from loans the matcher links across quarters. A change that the
  matcher reads as a new loan counts as a refinancing instead.
- **Non-accrual is missing** where filings report it only in HTML, and understated for some BDCs.

## 6. Look-ahead check (`scripts/check_look_ahead.py`, `look_ahead_check.csv`)

The check cuts `FINAL_DATASET.csv` after a quarter T, rebuilds the whole panel from it as if the
data ended there, and compares it with the panel built from all the data. Nothing known at T may
change when later quarters are added. The cuts are the end of development and the test quarter
(`config.TRAIN_END` and `config.TEST_QS`: 2025Q2 and 2025Q4). Run on 2026-11-23, with the
past-year PIK switch and the cleaned PIK flag (which looks back two filings):

| Cut after | Signs (all five and the two dropped), sample flags, non-accrual status at quarters <= T | Outcomes of windows closed by T | Scorecard scores for T |
|---|---|---|---|
| 2025Q2 | 67,625 rows, all identical | 51,321 rows, all identical | 5,404 scores, identical |
| 2025Q4 | 85,333 rows, all identical | 67,625 rows, all identical | 5,838 scores, identical |

At the cuts of the earlier split (2025Q1 and 2025Q3, before the past-year PIK switch) one
difference showed at 2025Q3: a debt-for-equity flag and the quarter of one event. The chain linker
bridges a loan's gap of up to two missed filings, so whether a loan ended, and was replaced by
equity, is settled only two quarters later; in data cut at 2025Q3, FS Whitewater's loan seemed to
end in 2025Q2 and be swapped in 2025Q3, while in the full data the swap falls a quarter later. It
touched exits and the equity swaps only, never a sign or a fitted label.

**Checked by reading the code:**

- Every sign looks back only (same-loan changes from t - 1 to t, the drawdown over t - 4 to t, a
  PIK switch against the previous filing, the past-year PIK switch over t - 3 to t - 1).
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

Code in `src/9_prediction/`, one job per file: `config.py` (constants), `features.py` (the signs and the
`FEATURES` registry of units, caps and buckets), `targets.py` (the outcomes: credit event, exits,
restructurings, next-quarter markdown), `credit_events.py` (term changes and loan exits), `panel.py`
(joins them, sample rules, event and exit counts), `specs.py` (which signs go in which model),
`frames.py` (the rows a model sees), `models/` (one file per kind of estimator: `logit.py`,
`linear.py`, and `ranking.py`, equal-weight scores that `run_model.py` no longer uses; lasso, trees
and a neural net go here), `evaluation.py` (time splits and
metrics), `analysis.py` (tables, including the one-input baselines) and `plots.py`. Outputs in
`outputs/model/`.

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
  **Change (fall this quarter):** on the same loans only, entered as a fall (`fall_1q`: a fall is
  positive, a rise negative, like the fall over the past year), capped at 20 points either way. **PIK switch:** a loan that paid cash
  starts paying in kind, or its PIK rate rises with its share of the coupon up 10+ points; PIK
  back after one filing at zero with the coupon unchanged is not a switch
  (`pik_metrics.pik_switch_clean`). **PIK switch in the past year:** 1 if
  the borrower had a PIK switch in any of the three quarters before t (a quarter it was not held,
  or its first quarter, counts as none). **Fall over the past year:** largest
  peak-to-trough fall of the same-loan mark over four quarters, capped at 20.
- **Tested and dropped:** the valuation gap (highest minus lowest lender family's mark on the
  same loan), crowding (number of lender families), the 10 pairwise interactions of the five
  signs, and weighting recent quarters more (section 3).
- **Fit:** an unpenalized logistic regression; every training quarter weighs the same
  (`config.HALF_LIFE = None`; `models/logit.recency_weights` gives the tested alternative).
- **The caps make no difference to missed events** (`caps_comparison.csv`). Events outside each
  quarter's riskiest 5 / 10 / 20 / 30% of borrowers, capped against uncapped: test (283 events)
  202 / 165 / 112 / 91 against 207 / 167 / 110 / 91; development, rolling (1,343 events) 937
  / 735 / 457 / 352 against 947 / 728 / 456 / 356. Test ROC-AUC is 0.753 capped and 0.750
  uncapped. The caps change 96 test borrowers (19 events) and hold the highest probability at 86%
  instead of 97%.
- **Stages:** a quarter's window closes two quarters later, so each fit uses only windows closed
  before the quarter it scores. Choosing signs: rolling, each of signals 2024Q2-2025Q2 scored by
  a fit on the windows closed by then; the five rounds' predictions are pooled into one ROC-AUC
  per model. Test: fit on outcomes through 2025Q4, scored on the borrowers of 2025Q4 (events
  2026Q1-Q2); 2025Q3 is a gap, since its window overlaps the last fitted outcomes. Blind test (not in the data used; needs 2026Q3): fit on outcomes through
  2026Q1, scored on the borrowers of 2026Q1 (events 2026Q2-Q3). Scorecard: fit on outcomes through
  2026Q2 (signals 2023Q2-2025Q4), scored on the borrowers of 2026Q2.

  | Stage | Fitted on (signal quarters) | Rows fitted | Events fitted | Scored |
  |---|---|---|---|---|
  | Choosing signs, first round | 2023Q2-2023Q4 | 9,662 | 617 | borrowers of 2024Q2 |
  | Choosing signs, last round | 2023Q2-2024Q4 | 25,705 | 1,657 | borrowers of 2025Q2 |
  | Test | 2023Q2-2025Q2 | 35,183 | 2,203 | borrowers of 2025Q4 (5,169; 283 events) |
  | Scorecard | 2023Q2-2025Q4 | 45,419 | 2,788 | borrowers of 2026Q2 (6,267) |
- **Intervals:** borrower-resampled bootstrap for every performance measure and rate (AUC,
  recall, event rates), since a borrower appears in several quarters and one event can fall in
  two overlapping windows; borrower-clustered standard errors for coefficients.

Further checks are written by the same run:

- **Each sign alone:** `event_rates_by_bucket.csv` and `event_rate_by_predictor.png` (the five
  signs and the two dropped), `pik_switch_hypotheses.csv`, `coefficients.csv`, `odds_ratios.png`.
- **Alternative signs** (`robustness_summary.csv`): test ROC-AUC 0.73 to 0.76 across variants of
  the signs (0.732 without the past-year PIK switch), and 0.78 on the 121 events among borrowers
  with 2+ lender families.
- **Other outcomes, labelled as such in the same file:** non-accrual alone (the outcome until
  2026-11-16) 0.85 on 87 events, or 0.85 with the five signs the rule kept for it then (before
  the past-year PIK switch was a candidate); non-accrual or any of the five term changes 0.58 on
  808 events.
