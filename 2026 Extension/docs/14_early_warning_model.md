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
| Its lenders disagree on its value | each point apart: about **1.03x** (not significant) |

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
python tests/test_model.py             # 22 tests
```

## Departures from the sponsor's note

- **Non-accrual only.** The note counts non-accrual, restructuring or a realized loss. The
  filings have no restructuring or realized-loss field: no per-loan realized gain or loss tag in
  the 1,792 iXBRL columns, the two firm-specific candidates (`piac:InvestmentOwnedSales`,
  `piac:...TransfersFromRestructuring...`) are empty, and the 2,401 saved footnotes mention a
  restructuring three times (one 2021 deal) and a realized loss never. Inferring a loss from a
  borrower leaving the books marked low would make a markdown the outcome, which the note rules
  out.
- **Two quarters, the short end of the note's two to four,** so that the test can use signals
  from the second half of 2025 and every window still closes inside the data.
- **Exits are typed, not classified as repaid, refinanced or impaired** (section 2). The filings
  show that a borrower left, not why.
- **The rule for keeping signs and the caps were fixed on 2026-11-13,** after an earlier
  one-quarter version of the test had been seen. They were applied unchanged to this outcome.
  2026Q3 is held back as the blind test.

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
two quarters. Windows overlap (a borrower's quarters t and t + 1 share a quarter), so standard
errors are clustered and intervals resampled by borrower.

## 2. Exits (`exits.csv`)

A borrower that leaves every BDC's books before its window closes, without a flag, has no label.
Each exit is typed from its scheduled maturity only:

| Exit type | Exits | Share | Mark at t below 80 | 80-95 | 95 and up |
|---|---|---|---|---|---|
| At scheduled maturity | 220 | 4% | 9 | 18 | 193 |
| Before scheduled maturity, cause not reported | 5,133 | 95% | 164 | 426 | 4,543 |
| Maturity not reported | 42 | 1% | 5 | 8 | 29 |
| **All** | 5,395 | | 178 | 452 | 4,765 |

The mark columns are context, never a classification: 88% left marked at 95 or above, which
suggests repayment, but the filings do not say.

**Refinancing.** A borrower refinanced by another BDC is still held, so it is never an exit: it
stays in the panel and is scored. Within the BDCs, a refinancing is partly visible at the loan level:
of 26,978 loan positions that ended in 2023-2026Q1, 26% ended early in a quarter in which a new
loan to the same borrower appeared (5% at the same BDC, 21% at another). That pattern also fits
an add-on loan or a break in loan matching after an amendment, so it is not proof. A refinancing by a bank or the
syndicated market, an early repayment, a sale and a write-off look the same in the schedule of
investments; only the MD&A text separates them, and not for every BDC.

## 3. Which signs are kept (`sign_selection.csv`)

Each sign is added alone to the baseline (mark and its change) and kept if it adds at least half
a point of out-of-sample ROC-AUC on the development quarters (signals 2024Q2-2025Q1, 278 entries).

| Model | ROC-AUC | Gain over the baseline [95% CI] | Kept |
|---|---|---|---|
| Baseline: mark and its change | 0.802 | | |
| + PIK switch | 0.808 | +0.007 [-0.003, 0.024] | **yes** |
| + fall over the past year | 0.807 | +0.006 [-0.002, 0.015] | **yes** |
| + valuation gap | 0.807 | +0.005 [-0.009, 0.019] | **yes** |
| + crowding | 0.801 | -0.001 [-0.006, 0.005] | no |
| Scorecard: all three kept signs | 0.813 | +0.011 [-0.005, 0.031] | |

No gain's interval clears zero; the rule keeps three signs by small margins.

## 4. Out-of-sample performance (`oos_performance.csv`, `oos_score_deciles.png`)

Fitted once on signals up to 2025Q1 (every window closed by 2025Q3); tested on signals from
2025Q3-Q4: 10,223 borrowers, 190 entries in 2025Q4-2026Q2.

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Entries caught in the riskiest 10% | in the riskiest 20% |
|---|---|---|---|---|
| **Scorecard** | **0.829 [0.78, 0.87]** | **0.190 [0.14, 0.25]** | **125 of 190** | **149 of 190** |
| Mark and its change | 0.830 [0.78, 0.87] | 0.188 [0.13, 0.25] | 130 of 190 | 145 of 190 |
| Mark alone | 0.824 [0.77, 0.87] | 0.164 [0.12, 0.22] | 128 of 190 | 145 of 190 |
| Equal-weighted ranking of the same signs | 0.798 [0.75, 0.84] | 0.117 [0.09, 0.16] | 103 of 190 | 132 of 190 |
| Random ranking | 0.50 | 0.019 (the entry rate) | 19 of 190 | 38 of 190 |

PR-AUC scores only how well entries are found, so it suits a rare event; its random baseline is
the entry rate, not 0.5. The scorecard's 0.19 is about 10 times random.

| Score decile | Realized entry rate [95% CI] | Predicted |
|---|---|---|
| 10 (riskiest) | 12.2% [10.4, 14.4] | 9.8% |
| 9 | 2.3% [1.6, 3.5] | 1.3% |
| 1-8 | 0.5% | 0.7% |
| All | 1.86% [1.61, 2.14] | 1.68% |

- **The mark does the work.** The scorecard ties the mark and its change (difference -0.001
  [-0.022, 0.019]). The three added signs helped slightly on the development quarters and not
  on the test. The scorecard is kept because the rule, fixed in advance, chose it; a version with
  only the PIK switch added scored 0.836 (`robustness_summary.csv`), within the same interval.
- **Fitted weights vs equal weights:** +0.031 [-0.002, 0.062], better but not clearly.
- **Predictions run low in the riskiest tenth** (9.8% predicted, 12.2% realized): entries rose
  in 2025-2026 relative to the training years.
- **The risk sits in the top fifth.** Below it the scorecard barely separates borrowers.
- **No overfitting:** ROC-AUC is 0.827 on the training quarters and 0.829 on the test, and 0.78
  to 0.85 in each of seven quarters scored one at a time (`rolling_by_quarter.csv`).

## 5. The scorecard for 2026Q2 (`scorecard_2026Q2.csv`)

All 6,267 borrowers accruing in 2026Q2, ranked, with their probability and the signs behind it.
About 120 entries are expected in 2026Q3-Q4, 70 of them in the riskiest tenth. At the top: Rural
Sourcing Holdings (70%: marked at 74, down 12 points this quarter and 25 over the year, just
switched to PIK, lenders 15 points apart). Once 2026Q3 is filed, a rerun tests the scorecard
blind on the 2026Q1 signals (`reserved_test_2026Q3.csv`); this list itself is scored once
2026Q4 is filed.

## Limitations

- **Few events:** 776 in 11 windows, 190 in the test. Gaps of a point of AUC are noise.
- **Sudden defaults are missed:** a third of the entries were outside the riskiest tenth.
- **Losses without non-accrual are invisible:** a borrower sold or written off without first
  being flagged leaves the books unlabelled (section 2).
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
- **Exit type:** at scheduled maturity if every loan held at t was due by the quarter the borrower
  was gone; before scheduled maturity otherwise; maturity not reported.
- **Mark:** fair value / par (fair value / cost for debt without par), capped below at 70.
  **Change:** on the same loans only, capped at 20 points. **PIK switch:** a loan that paid cash
  starts paying in kind, or its PIK share rises 10+ points. **Fall over the past year:** largest
  peak-to-trough fall of the same-loan mark over four quarters, capped at 20. **Valuation gap:**
  std of lender families' marks on the same loan (0 with an indicator for single-lender borrowers).
- **Tested and dropped:** crowding (number of lender families).
- **Stages:** a quarter's window closes two quarters later, so each fit uses only windows closed
  before the quarter it scores. Development: signals 2023Q2-2025Q1 (choose signs, rolling test
  2024Q2-2025Q1). Test: signals 2025Q3-2025Q4 (performance table; 2025Q2 is skipped because its
  window overlaps the test). Scorecard: signals 2026Q2, fitted on 2023Q2-2025Q4.
- **Intervals:** borrower-resampled bootstrap for AUC; Wilson for rates; borrower-clustered
  standard errors for coefficients.

Further checks are written by the same run: each sign alone (`event_rates_by_bucket.csv`,
`event_profile.png`, `pik_switch_hypotheses.csv`), coefficients (`coefficients.csv`) and
alternative definitions (`robustness_summary.csv`: test ROC-AUC 0.82 to 0.84, all within the
interval above). Entrants' median mark falls from 96 four quarters before entry to 86 the quarter
before; borrowers that stay accruing hold at 99 (`event_profile.csv`).
