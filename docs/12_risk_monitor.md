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
python scripts/run_risk_monitor.py --skip-eval      # market and signal layers only (midterm), a few minutes
```

Code: `src/8_risk_monitor/` (`monitor_panel.py`, `benchmarks.py`, `evaluation.py`,
`charts.py`). Reads only `FINAL_DATASET.csv`. Writes to `outputs/risk_monitor/`.

| File | Contents |
|---|---|
| `benchmark_table.csv`, `market_series.csv`, `market_layer.png` | Market layer |
| `outcome_rung1_by_quarter.csv` | Eligible loans, events, decile cutoff and share marked down per quarter |
| `signal_tail_by_bdc.csv`, `signal_layer.png` | Signal layer, all signals combined |
| `signals/signal_tail_<signal>.png`, `signal_tail_summary.csv` | Signal layer, one chart per signal (concentration, crowding, dispersion, pik_total, pik_amendment) |
| `signals/bdc_signals.csv`, `signals/borrower_signals.csv` | The signal values per BDC-quarter and borrower-quarter |
| `signals/pairwise_overlap*.csv`, `signals/pairwise_overlap.png` | Portfolio overlap between manager families (C5) |
| `signals/confounder_size.*`, `signals/confounder_bdc_type.*` | Preliminary confounder splits |
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
| Fig 10 | Top 20% of managers deployed 88% of 2025 capital | 79% (top 20% of 80 BDC families) |
| Fig 40 | Share of loans with a significant amendment, mean 10.6% | 18.3% stress-type, mean across 151 BDCs (PIK toggle 4.1%, spread up 7.6%, maturity moved 9.8%); 29.7% incl. spread cuts |
| Fig 18 | Business services 25%, software 21%, industrials 17% | Software 16.2%, Business Services 10.3%, Industrials (rest) 10.9%; excl. the 10% unclassified: 18.1%, 11.5%, 12.1% |
| Fig 11 | Deals above $1bn ~12% of transactions | 0.4% of borrowers, 11% of par (different unit) |
| Fig 39 | Non-accruals 4.2% public (listed) vs 1.1% private BDCs, June 2026 | Listed: mean across BDCs 4.1%. Non-listed (non-traded + private): cost-weighted 1.2%; non-traded mean 1.0%; private-placement BDCs mean 4.3%, median 1.2% |
| Fig 37 | 79% of funds <= 2.5% non-accrual, 17% above 5% | 67% of 111 BDCs <= 2.5% of NAV, 18% above 5%, 15% in between |
| Fig 38 | Default rate by industry, healthcare highest at 2.7%, software 0.2% | Non-accrual at cost: Transportation 5.2%, Real Estate 2.7%, Software 2.6%, Comm. & Media 2.4%, Healthcare 2.1%, Business Services 0.9% |
| Fig 27 | BDC assets by type | Listed $153bn (48 BDCs), non-traded $233bn (24), private $142bn (99) |
| Fig 25 | Non-listed BDC flows | Pending: needs fund-level issuance and repurchases |

The return and size benchmarks match closely, which supports the panel's fair values. The
public-vs-private non-accrual split matches Fig 39, which the guideline names as the test that
the status field is read correctly: listed BDCs 4.1% (report 4.2%) and non-listed BDCs 1.2%
cost-weighted (report 1.1%). The mean across private-placement BDCs alone (4.3%) is pulled up by
a few small funds; their median is 1.2%.

**Sectors (Fig 18).** `sector_report` (FINAL_DATASET, step 7 of `6_final_dataset.md`) splits
Software out of Technology and Business Services out of Industrials, using the industry text the
BDCs report, so the weights can be set against the BDC Monitor. Software comes close (18% of
classified FV vs 21%). Business services does not (12% vs 25%): the report's bucket is probably
broader (IT services, about 1.5 points, stay in Technology here), and 10% of panel FV has no
sector. Fig 38 is the less comparable one: panel software non-accrual is 2.6% against a 0.2%
software default rate. Non-accrual is not default, and a few large software non-accruals
(2026Q1 re-marks) drive it. Report-aligned sub-index returns against MSCI are in
`5_subindex_construction.md`. Fig 37 divides non-accrual fair value by NAV, as the report does; NAV
comes from each filing's balance sheet (`StockholdersEquity` / `AssetsNet` in
`data/parsed/nonaccrual/filing_totals.csv`, `benchmarks.nav_by_bdc`). The above-5% share matches
(18% against 17%), but the panel has no empty middle (15% at 2.5-5% against the report's 3%).
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

## Signal layer, one signal at a time (`signal_tails.py`)

The guideline asks for each signal's cross-section by quarter, with the right tail marked and
whether the tail is growing. A within-quarter percentile always puts 10% in its top decile, so
the tail threshold is fixed instead: **the 90th percentile of the pooled 2023Q2-2024Q4
cross-section**. Each chart shows the 10-25-50-75-90 percentiles by quarter with the tail units
as points, and the share of units above the threshold (dotted line: the baseline share).

| Signal (unit) | Definition | Tail threshold | Units in the tail | Reading |
|---|---|---|---|---|
| Concentration (BDC) | HHI of the BDC's book across borrowers, cost weights; effective N = 1/HHI | HHI 0.046 (effective N 22) | 6-13% a quarter, no trend | Books get more diversified: median effective N 59 to 71 borrowers |
| Crowding (borrower) | Lender manager families holding the borrower | more than 3 families | 4.1% (2023Q1) to 7.2% (2026Q2) | Growing, partly because the panel covers more BDCs over time |
| Dispersion (BDC) | Par-weighted mean of the BDC's distance from co-lender consensus, loans held by 2+ families | 0.0146 (1.5 points of par) | 4.5-10% in 2024-25, 8.6% in 2026Q1, 11.4% in 2026Q2 (highest since 2023) | The 1H26 stress shows here: median distance up from 0.42 to 0.68 points. 2023 is noisy (few matched loans) |
| PIK, total (BDC) | Share of debt FV paying any PIK | 20.4% | 7-12%, flat | Market-level PIK is steady |
| PIK, amendment-driven (BDC) | Share of debt FV on PIK acquired after origination (cash-pay first, PIK now; includes P2 toggles) | 9.2% | 18% (2024Q4) to 20% (2026Q2) | Rising slowly. Before 2024Q4 the rise is mechanical: acquired PIK needs the panel to have seen the loan pay cash first |

Dispersion is the lender-versus-consensus measure the outline asks for: it ranks which BDC
marks away from its co-lenders, not just whether lenders disagree. PIK is split into total
(the market series the report and Houlihan Lokey publish) and amendment-driven (the sharper
signal, Hunt p.57), using the origination classification in `9_pik_metrics.md`.

### Pairwise portfolio overlap (C5)

For two manager families, **dollar-weighted overlap** = sum over shared borrowers of
min(w_A, w_B), with w the borrower's cost share of the family's book (0 = no common borrower,
1 = identical books); Jaccard = shared borrowers / borrowers in either. Families with 20+
borrowers: 84 in 2026Q2, 3,486 pairs. Families as in `11_tranche_matching.md`, after the 2026-10-24
affiliate merges.

- Most books barely overlap: half the pairs share no borrower, and the median pair that does
  shares about 1%. About 0.9-1.3% of pairs overlap by more than 10%, with no clear trend.
- Among the 15 largest families the highest overlaps are 10-16% (Blackstone - Blue Owl 16%,
  Morgan Stanley - New Mountain 15%, Morgan Stanley - Golub 14%): an allocator holding both owns
  the same borrowers for about a seventh of each book.
- **Overlap found affiliates the family map missed.** Before the merge, Jefferies Credit
  Partners BDC - Senior Credit Investments (88%), LGAM Private Credit - Morgan Stanley (57%) and
  EPT 16 - Trinity Capital (53%) topped the list. The first two (and four more found from the
  filings) are now merged (`11_tranche_matching.md`). EPT 16 - Trinity (53%) is the only pair
  above 20% left: Trinity sources its loans, but Eagle Point values them.

### Confounders (preliminary)

The guideline lists eight confounders (borrower size, BDC type, sector, vintage, market-wide
re-marks, portfolio growth, fund leverage, retail flows). The peer-relative outcome already
handles market-wide re-marks. The two cheapest and most consequential splits are done here,
descriptively (no controls yet).

**Borrower size: are crowded borrowers just big borrowers? Largely yes.** Spearman correlation of
BDC-held par and lender families: 0.47 in the median quarter. By size quintile (pooled):

| Size quintile (median BDC-held par) | Mean lender families | In the crowding tail |
|---|---|---|
| Q1 ($1.8M) | 1.03 | 0.0% |
| Q2 ($6.9M) | 1.13 | 0.3% |
| Q3 ($16M) | 1.27 | 1.4% |
| Q4 ($37M) | 1.57 | 5.6% |
| Q5 ($116M) | 2.66 | 23.8% |

Almost every crowded borrower is in the top size quintile, so crowding has to be read within size
buckets. This matches the evaluation result below (lender count adds nothing once size is
controlled).

**BDC type: do the signals differ for listed vs non-traded BDCs? Yes, on every signal** (2026Q2,
median BDC):

| Signal | Listed (43) | Non-traded (23) | Private (83) |
|---|---|---|---|
| Concentration (HHI) | 0.013 | 0.010 | 0.017 |
| Dispersion (points of par) | 0.82 | 0.59 | 0.65 |
| PIK, total | 13.8% | 6.6% | 7.7% |
| PIK, amendment-driven | 6.9% | 4.5% | 3.9% |
| Share of BDCs in the PIK-total tail | 21% | 4% | 2% |

Listed BDCs carry about twice the PIK and sit further from consensus, in line with their higher
non-accrual rate (Fig 39). Non-traded BDCs are the most diversified and almost never in a tail.
A signal pooled across types partly measures BDC type, so the final evaluation should test the
signals within type.

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

Rerun on 2026-10-24 with the merged manager families (`11_tranche_matching.md`); gains moved by
0.002 or less.

| Signal | AUC alone | Gain over baseline [90%] | Reading |
|---|---|---|---|
| Price (baseline) | 0.68 | | Lowest price quintile: 39% event rate. Marks are persistent |
| Two-quarter price change (baseline) | 0.63 | | Baseline AUC with both and controls: 0.69 |
| P1 PIK share | 0.55 | +0.003 [+0.002, +0.004] | Top quintile 22% |
| P1 acquired PIK share | 0.55 | +0.003 [+0.002, +0.004] | |
| P2 PIK toggle | 0.52 | +0.002 [+0.001, +0.003] | 37% event rate after a toggle vs 16%; rare (2.4% of rows) |
| V1 dispersion | 0.62 | +0.003 [+0.001, +0.006] | Top quintile 25% |
| V2 low mark | 0.61 | +0.004 [+0.002, +0.007] | |
| Lender's mark minus consensus | 0.51 | +0.008 [+0.006, +0.010] | U-shaped: both outliers mark down more later |
| Lender's distance from consensus | 0.60 | +0.002 [+0.001, +0.003] | |
| C1 lender families | 0.48 | +0.000 | More lenders, fewer markdowns, explained by size |
| C2 lender HHI | 0.53 | -0.000 | |
| C4 change in families | 0.42 | +0.003 [+0.001, +0.005] | Reversed: lenders leaving precedes markdowns |

Same pattern at h = 1 and 4, and with the within-sector outcome. Out of sample (1H26), the
gains keep their sign for PIK, V2, the distance from consensus and C4 at h = 1 and 2; the
signed lender-minus-consensus measure, the largest in-sample gain, turns negative at h = 2
(-0.001), so its in-sample gain is partly overfit to its U shape.

### Reading

- **The baseline is hard to beat.** Fair values are smoothed, so a loan already marked down
  keeps being marked down. Every signal adds little AUC on top: gains of 0.002 to 0.009.
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
| P1 PIK share | 0.64 | +0.017 [+0.013, +0.022] | +0.017 | +0.018 / +0.013 |
| P1 acquired PIK share | 0.60 | +0.010 [+0.008, +0.012] | +0.010 | +0.012 / +0.009 |
| P2 PIK toggle | 0.56 | +0.006 [+0.004, +0.007] | +0.006 | +0.005 / +0.004 |
| V1 dispersion | 0.82 | +0.006 [+0.003, +0.010] | +0.018 | -0.003 / -0.005 |
| V2 low mark | 0.79 | +0.005 [+0.001, +0.009] | +0.015 | -0.006 / -0.007 |
| C4 change in families | 0.36 (reversed) | +0.004 [+0.003, +0.006] | +0.008 | +0.004 / +0.008 |
| C1 lender families, C2 lender HHI | 0.44, 0.55 | about 0 | about 0 | about 0 |

- **Price carries most of it:** a loan heading for non-accrual is usually marked down first
  (AUC 0.84 to 0.86).
- **PIK is the credit signal.** Its gain is five to six times larger for non-accrual than for
  markdowns, and holds out of sample. This is the guideline's distinction: PIK predicts
  credit, not just marks.
- **Lenders leaving (C4)** predicts both outcomes and holds out of sample.
- **Dispersion** helps at four quarters in sample but not in 1H26; treat it as unproven for
  credit outcomes.
- **Lender count and HHI** add nothing once size is controlled, for either outcome.

## Limitations

The midterm presentation lists four:

1. **Quarterly, lagged filings.** A quarter's holdings are public 45-90 days after quarter end.
2. **Manager-discretionary marks.** Fair values are the BDC's own appraisal and are smoothed;
   that is why the price baseline is hard to beat.
3. **Growth dilution** (Leger, p.41). BDC fair value more than doubled over the panel ($241bn in
   2023Q1 to $528bn in 2026Q2), so any rate (non-accrual, PIK, markdown frequency) is diluted by new, unseasoned loans,
   and signals that need history (C4, acquired PIK, dispersion on matched loans) rise
   mechanically in 2023. Not yet normalised by cohort; compare quarters from 2024Q4, or restrict
   to loans seen for four or more quarters (`age_q`).
4. **No borrower fundamentals.** Filings do not disclose borrower LTV, leverage or interest
   coverage, the basis of the report's credit chapter (Figs 30, 33, 34). This is what makes it an
   outside-in monitor: it reads what managers disclose about their own holdings. The partial
   proxy the guideline names (a loan's spread against the quarter's new-issue spread, about
   150 bp over BSL, p.19) is not built.

Further limitations of this analysis:

- Rows are BDC loan-quarters; the same loan held by several BDCs is counted several times and
  the quarter bootstrap does not account for that, so intervals are somewhat too narrow.
- 14 quarters; h = 4 has no out-of-sample window.
- The bootstrap holds the fitted models fixed; it measures ranking uncertainty, not fitting
  uncertainty.
- Non-accrual status is unknown for filings that mark it only in HTML (14% of fair value; 30%
  for private BDCs), so rung 3 covers mostly listed and non-traded BDCs.
- Fig 25 (fund flows) is not in FINAL_DATASET.
- BDC type is not yet used as a split in the evaluation (the guideline's second confounder).
