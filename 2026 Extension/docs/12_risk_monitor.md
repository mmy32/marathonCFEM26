# Risk Monitor: benchmarks, outcomes (rungs 1 and 3), evaluation

## Overview

The CFEM guideline note on *Financing the Economy 2026* (ACC / Houlihan Lokey, 22 October
2026) asks for a quarterly monitor of where the private credit stress tail forms, in three
layers: market (panel aggregates against published figures), signal (distribution of the
signals across BDCs), and evaluation (which signals predict which outcomes, at what horizon).
Stage 1 built all three with the data FINAL_DATASET had. Stage 2 (same day) added a BDC-type
tag and non-accrual status to FINAL_DATASET, which filled the remaining benchmarks and added
outcome rung 3. Rung 2 (amendments) is measured in the benchmark table; rung 4 (exit at a loss)
is not built.

```
python scripts/run_risk_monitor.py [--reps 200]     # about 15 minutes
```

Code: `src/8_risk_monitor/` (`monitor_panel.py`, `benchmarks.py`, `evaluation.py`,
`charts.py`). Reads only `FINAL_DATASET.csv`. Writes to `outputs/risk_monitor/`.

| File | Contents |
|---|---|
| `benchmark_table.csv`, `market_series.csv`, `market_layer.png` | Market layer |
| `outcome_rung1_by_quarter.csv` | Eligible loans, events, decile cutoff and share marked down per quarter |
| `signal_tail_by_bdc.csv`, `signal_layer.png` | Signal layer |
| `evaluation_summary.csv`, `evaluation_buckets.csv`, `evaluation_h{1,2,4}.png` | Evaluation layer |
| `evaluation_summary_sector_outcome.csv` | The same at h = 2 with the outcome defined within quarter x sector |

---

## Unit of analysis

One row per BDC loan-quarter: the tranche chains of `11_tranche_matching.md` (273,769 rows).
The same loan held by several BDCs appears once per BDC.

## Outcome, rung 1: peer-relative markdown

- **Valuation return** since the chain's previous quarter:
  `(FV - FV_prev - (COST - COST_prev)) / FV_prev`. Draws, paydowns and PIK capitalisation
  move fair value and cost together and cancel; only re-marks remain. Needs the previous
  quarter to be the one just before.
- **Event:** the return is negative and in the bottom decile of that quarter's cross-section.
  A market-wide re-mark moves the whole distribution without changing ranks: in 2026Q1, 65%
  of loans were marked down and the decile cutoff was -4.4%, against about -1.5% to -2% in
  other quarters; the event rate stays 10%.
- **Within quarter x sector** version (sectors with 200+ loans that quarter): `mk_event_sec`.
- **Forward outcome** `Y_mk_h{h}`: any event on the same chain in quarters t+1..t+h.
  Undefined when the chain has no eligible return in the window or the window runs past
  2026Q2. Base rates: 10% at h = 1, 16% at h = 2, 24% at h = 4.

A cost write-down moves cost with fair value, so it is not a rung-1 event; it belongs in rung 2
(restructuring) with the other T2 events.

## Market layer: benchmark table

| Report figure | Reported | Panel |
|---|---|---|
| Fig 1 / p.39 | Market $4.1tn; BDCs ~$500bn | $528bn at 2026Q2 (12.9%) |
| Fig 15 | MSCI 9.5% in 2025, 0.9% in Q1 2026; CDLI 9.3% in 2025 | 9.6% in 2025; 1.0% in Q1 2026 |
| Fig 10 | Top 20% of managers deployed 88% of 2025 capital | 80% (top 20% of 85 BDC families) |
| Fig 40 | Share of loans with a significant amendment, mean 10.6% | 14.5% stress-type (PIK toggle 4.0%, spread up 7.5%, maturity moved 5.3%); 27.3% incl. spread cuts |
| Fig 18 | Business services 25%, software 21%, industrials 17% | Technology 21.8%, Industrials 21.2%, Healthcare 15.9% (taxonomies differ) |
| Fig 11 | Deals above $1bn ~12% of transactions | 0.4% of borrowers, 11% of par (different unit) |
| Fig 39 | Non-accruals 4.2% listed vs 1.1% private BDCs, June 2026 | Mean across BDCs at cost: listed 4.1%, non-traded 1.0%, private 4.3% (median 1.2%) |
| Fig 37 | 79% of funds <= 2.5% non-accrual, 17% above 5% | 78% of 112 BDCs <= 2.5%, 9% above 5% (share of debt FV, not NAV) |
| Fig 38 | Default rate by industry, healthcare highest at 2.7% | Non-accrual at cost: Transportation 5.2%, Real Estate 2.7%, Comm. & Media 2.4%, Technology 2.2%, Healthcare 2.1% |
| Fig 27 | BDC assets by type | Listed $153bn (48 BDCs), non-traded $233bn (24), private $142bn (99) |
| Fig 25 | Non-listed BDC flows | Pending: needs fund-level issuance and repurchases |

The return and size benchmarks match closely, which supports the panel's fair values. The
listed-vs-non-traded non-accrual split matches Fig 39, which the guideline names as the test
that the status field is read correctly. Fig 37 uses debt fair value as the denominator
because NAV is not in FINAL_DATASET, so its "above 5%" share is lower than a NAV-based one.
Notes on each comparison are in `benchmark_table.csv`. BDC type and non-accrual status come from
FINAL_DATASET's `bdc_type` and `non_accrual` columns (`6_final_dataset.md`, steps 4-5).

## Signal layer

A loan is in the signal tail if any of: PIK toggle at t or t-1; the lowest family mark on
the loan is in the quarter's bottom decile relative to consensus (V2); this lender's distance
from consensus is in the top decile; the number of lender families fell by the most over four
quarters (bottom decile of C4). The chart shows, per quarter, the median and 90th-percentile
share of a BDC's fair value in tail loans, and how many BDCs have more than 10% there.

The rise from 2023 is partly mechanical: C4 needs four quarters of history, and the share of
loans held by 2+ families (needed for V2 and the distance) grows with the panel. The
guideline's growth confound applies; compare quarters from 2024Q1 on.

## Evaluation layer

For each signal and horizon (1, 2, 4 quarters):

- event rate by quintile of the signal's within-quarter percentile (0/1 for the toggle), with
  n and events per bucket;
- AUC of the signal alone, oriented so higher = riskier (below 0.5 = the opposite);
- AUC of the baseline and of baseline + signal (logistic), the gain, and a 90% interval from
  200 quarter-bootstrap draws (predictions fixed, not refit);
- out of sample: fitted on windows ending by 2025Q4, scored on windows starting in 2026 (the
  1H26 stress episode; h = 1 and 2 only).

Baseline: price percentile, two-quarter price change percentile, PIK at origination (with an
unknown flag), lien group, log of the borrower's BDC-held par (the guideline's size
confounder). Each signal is tested on the rows where it exists (dispersion: loans held by 2+
families, about 45% of rows), with the baseline refit there.

### Results, h = 2 (peer-relative markdown within two quarters, base rate 16%)

| Signal | AUC alone | Gain over baseline [90%] | Reading |
|---|---|---|---|
| Price (baseline) | 0.68 | | Lowest price quintile: 39% event rate. Marks are persistent |
| Two-quarter price change (baseline) | 0.63 | | Baseline AUC with both and controls: 0.69 |
| P1 PIK share | 0.55 | +0.004 [+0.002, +0.005] | Top quintile 22% |
| P1 acquired PIK share | 0.55 | +0.003 [+0.002, +0.005] | |
| P2 PIK toggle | 0.52 | +0.002 [+0.002, +0.003] | 37% event rate after a toggle vs 16%; rare (2.4% of rows) |
| V1 dispersion | 0.62 | +0.002 [-0.000, +0.004] | Top quintile 25% |
| V2 low mark | 0.60 | +0.002 [+0.001, +0.004] | |
| Lender's mark minus consensus | 0.50 | +0.006 [+0.004, +0.009] | U-shaped: both outliers mark down more later |
| Lender's distance from consensus | 0.60 | +0.002 [+0.001, +0.002] | |
| C1 lender families | 0.47 | +0.000 | More lenders, fewer markdowns, explained by size |
| C2 lender HHI | 0.53 | -0.000 | |
| C4 change in families | 0.42 | +0.003 [+0.001, +0.005] | Reversed: lenders leaving precedes markdowns |

Same pattern at h = 1 and 4, and with the within-sector outcome. Out of sample (1H26), the
gains keep their sign for PIK, V2, the distance from consensus and C4 at h = 1 and 2; the
signed lender-minus-consensus measure, the largest in-sample gain, turns negative at h = 2
(-0.002), so its in-sample gain is partly overfit to its U shape.

### Reading

- **The baseline is hard to beat.** Fair values are smoothed, so a loan already marked down
  keeps being marked down. Every signal adds little AUC on top: gains of 0.002 to 0.008.
- **What survives:** PIK share and toggles, the low mark (V2), the lender's distance from
  consensus, and a falling lender count. Their intervals exclude zero at every horizon and
  the out-of-sample gains keep their sign.
- **What does not:** the lender count and lender HHI. Their raw pattern (more lenders, fewer
  markdowns) disappears once borrower size is controlled, as the guideline expected.
- Rung 1 is a mark outcome. The guideline warns that a signal that predicts markdowns but not
  non-accruals is about marks, not credit; rung 3 below is the test of that.

### Rung 3: non-accrual onset

`na_onset`: a loan flagged non-accrual this quarter that was accruing in its chain's previous
quarter, both known. `Y_na_h`: an onset within h quarters, for loans accruing at t (only
filings with known status count). Base rates: 0.6% at h = 1, 1.1% at h = 2, 2.2% at h = 4.
Outputs: `evaluation_na_summary.csv`, `evaluation_na_buckets.csv`, `evaluation_na_h{2,4}.png`.

| Signal | AUC alone, h = 2 | Gain over baseline, h = 2 [90%] | h = 4 | Out of sample, h = 1 / 2 |
|---|---|---|---|---|
| Price (baseline) | 0.84 | | | |
| P1 PIK share | 0.65 | +0.017 [+0.012, +0.022] | +0.018 | +0.017 / +0.014 |
| P1 acquired PIK share | 0.60 | +0.010 [+0.007, +0.011] | +0.010 | +0.011 / +0.010 |
| P2 PIK toggle | 0.57 | +0.006 [+0.004, +0.007] | +0.006 | +0.005 / +0.005 |
| V1 dispersion | 0.82 | +0.004 [+0.002, +0.008] | +0.011 | -0.007 / -0.007 |
| V2 low mark | 0.78 | +0.003 [-0.000, +0.007] | +0.010 | -0.010 / -0.010 |
| C4 change in families | 0.36 (reversed) | +0.004 [+0.002, +0.006] | +0.007 | +0.005 / +0.008 |
| C1 lender families, C2 lender HHI | 0.43, 0.56 | about 0 | about 0 | about 0 |

- **Price carries most of it:** a loan heading for non-accrual is usually marked down first
  (AUC 0.84 to 0.86).
- **PIK is the credit signal.** Its gain is four times larger for non-accrual than for
  markdowns, and holds out of sample. This is the guideline's distinction: PIK predicts
  credit, not just marks.
- **Lenders leaving (C4)** predicts both outcomes and holds out of sample.
- **Dispersion** helps at four quarters in sample but not in 1H26; treat it as unproven for
  credit outcomes.
- **Lender count and HHI** add nothing once size is controlled, for either outcome.

## Limitations

- Rows are BDC loan-quarters; the same loan held by several BDCs is counted several times and
  the quarter bootstrap does not account for that, so intervals are somewhat too narrow.
- 14 quarters; h = 4 has no out-of-sample window.
- The bootstrap holds the fitted models fixed; it measures ranking uncertainty, not fitting
  uncertainty.
- Non-accrual status is unknown for filings that mark it only in HTML (14% of fair value; 30%
  for private BDCs), so rung 3 covers mostly listed and non-traded BDCs.
- Fig 25 (fund flows) and the NAV denominator for Fig 37 are not in FINAL_DATASET.
- BDC type is not yet used as a split in the evaluation (the guideline's second confounder).
