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
| `signals/borrower_warning_flags.csv`, `signals/warning_tail_summary.csv` | Warning signs per borrower-quarter (0/1 per sign, plus the rank behind the two ranked signs) and the share of borrowers flagged per quarter |
| `signals/warning_crowding.png`, `crowding_distribution.csv` | Crowding as a distribution: share of borrowers and of BDC-held debt by manager count (1, 2, 3, 4-5, 6+), each quarter |
| `signals/warning_disagreement.png`, `disagreement_percentiles.csv` | Disagreement as a distribution: median, 90th and 95th percentile of the gap, by borrowers and by dollars, each quarter |
| `signals/warning_pik.png` | PIK: borrowers paying more than half their interest in kind; borrowers switching to PIK each quarter |
| `signals/bdc_exposure.csv`, `bdc_exposure_by_quarter.csv`, `bdc_exposure.png` | Each BDC's exposure to flagged borrowers: share of portfolio, largest single flagged borrower |
| `signals/bdc_exposure_by_type.*` | Exposure by BDC type, latest quarter |
| `signals/borrower_signals.csv`, `signals/confounder_size.*` | Lender families and size per borrower-quarter; crowding by borrower size |
| `signals/pairwise_overlap*.csv`, `signals/pairwise_overlap.png` | Portfolio overlap between manager families (C5) |
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

## Signal layer: warning signs per borrower, exposure per BDC (`signal_tails.py`)

Borrowers default; a BDC is hurt when a borrower it holds defaults, in proportion to how much
of its portfolio that borrower is. So the warning signs are measured on the **borrower**, and
BDC risk is measured second, as **exposure** to the flagged borrowers. (Changed 2026-10-26: the
signal layer used to score concentration, dispersion and PIK per BDC. Concentration is a BDC
measure and stays one, inside exposure; dispersion and PIK are now borrower measures.)

The borrower-quarter panel is `distress_event_study.borrower_quarter_panel` (also used for the
stress events).

**Distributions, not fixed cutoffs** (changed 2026-10-26, on the sponsor's advice). Each sign is
shown as its whole distribution each quarter, counted by number of borrowers and by dollars of
BDC-held debt; that shows whether the tail is growing. Individual borrowers are flagged by rank:
in the top 10% of the quarter (crowding: of its size group, a fifth of BDC-held par). A rank
always flags about 10%, so it says who is in the tail, not whether the tail grows. Ties are never
split (a borrower is in the top 10% when at most 10% of its group have a value at least as high),
so whole-number manager counts flag a little under 10%. Before this change crowding and
disagreement were flagged above a fixed line (the 2023-24 90th percentile: more than 3 manager
families; a gap above 3.9 points of par); the manager-count line fell on a large tie (5.6% of
borrowers above it, 10.7% at or above).

**Crowding** (`C1_families`), share of borrowers / of BDC-held par by manager count:

| Managers | Borrowers, 2023Q1 | Borrowers, 2026Q2 | Par, 2023Q1 | Par, 2026Q2 |
|---|---|---|---|---|
| 1 | 78.9% | 72.5% | 51.0% | 39.4% |
| 2 | 12.7% | 14.6% | 19.4% | 17.3% |
| 3 | 4.3% | 5.7% | 8.7% | 11.9% |
| 4-5 | 3.0% | 4.5% | 12.1% | 12.6% |
| 6+ | 1.0% | 2.7% | 8.7% | 18.8% |

The far tail moved most. Flag: more manager families than 9 in 10 borrowers of its size fifth
that quarter, 4.6-6.8% of borrowers. In 2026Q2 4 managers is about the 74th percentile of the
largest fifth (BVI Medical, WebPT: not flagged); Medallia's 7 in 2025Q1 was in its top 10%.

**Disagreement on value** (`V1_range`: par-weighted range of the lenders' marks over the
borrower's loans held by 2+ families; only those borrowers, 25% of all), cents per dollar owed:

| Quarter | Median, borrowers | 90th pct, borrowers | Median, dollars | 90th pct, dollars | 95th pct, dollars |
|---|---|---|---|---|---|
| 2023Q1 | 1.1 | 5.6 | 2.0 | 5.6 | 7.0 |
| 2024Q4 | 0.5 | 2.8 | 0.9 | 3.9 | 6.1 |
| 2025Q4 | 0.5 | 3.0 | 0.7 | 3.6 | 5.8 |
| 2026Q2 | 0.6 | 4.0 | 1.2 | 6.3 | 9.3 |

"Dollars" weights each borrower by its BDC-held par: the 90th percentile by dollars is the gap
that 10% of shared dollars exceed. By dollars the 2026Q2 tail is the widest in the panel; by
borrowers it is up from the 2024 low but below early 2023. Flag: in the top 10% of the quarter.

| Sign (per borrower) | Column | Flagged when | Share of borrowers flagged | Reading |
|---|---|---|---|---|
| Crowding | `C1_families` | top 10% of its size fifth that quarter | 4.6-6.8% | See the distribution above |
| Disagreement on value | `V1_range` | top 10% of the quarter (borrowers with 2+ families) | 10% of those borrowers by construction | See the percentiles above |
| PIK, more than half | `pik_share` | more than half of its interest is paid in kind (par-weighted PIK share of the coupon, e.g. 6% or more of a 12% rate) | 3.1% (2023Q1) to 4.5% (2026Q2) | Rising steadily. About 12% of borrowers pay some PIK, most of them well under half in kind (median about a third) |
| PIK switch this quarter | `toggle_par_share` | one of its loans switched to PIK this quarter: PIK starts, or its share of the coupon rises, on a loan that paid cash the quarter before (the P2 toggle) | 1.6% (2023Q2), then 2.0-2.6% a quarter, no trend | Starts in 2023Q2: a switch is a change from the quarter before |
| PIK switched in the past year | `toggle_par_share` | a switch this quarter or in the three before | 6.2-6.8% from 2024Q1, flat | The distress sign used for exposure; starts in 2024Q1, the first quarter whose year lies inside the data |

A running stock of "PIK now, cash at origination" loans (`acquired_par_share`, used before
2026-10-26) is no longer a sign: it fills up from zero as the panel ages (5.3% of borrowers in
2024Q1 to about 7% by 2025) while new switches stay near 2.2% a quarter, so its rise was mostly
mechanical.

**Distress sign** = disagreement in the top 10% of the quarter, or a PIK switch in the past year:
8.0-8.6% of borrowers each quarter from 2024Q1 (8.2% in 2026Q2). Crowding is not one (it says how
many BDCs a default reaches, not how likely it is), nor is paying more than half in kind (some
loans are written that way).

**BDC exposure** (BDC-quarter, FINAL_DATASET rows, all instruments, fair value > 0; BDCs with
20+ borrowers): `exposure_<sign>` = share of the BDC's portfolio fair value in borrowers flagged
on that sign; `exposure_distress` for any distress sign; `top_flagged_weight` / `top_flagged` =
its largest distress-flagged borrower; `concentration` = HHI of its book (cost weights). From
2024Q1:

| Quarter | BDCs | Median exposure | BDCs above 20% of portfolio | BDCs with one flagged borrower above 5% |
|---|---|---|---|---|
| 2024Q1 | 119 | 12.5% | 16 (13%) | 11 |
| 2024Q4 | 132 | 13.7% | 33 (25%) | 12 |
| 2025Q4 | 150 | 11.3% | 20 (13%) | 16 |
| 2026Q1 | 155 | 12.2% | 32 (21%) | 16 |
| 2026Q2 | 158 | 12.4% | 36 (23%) | 17 |

The single-name count is the clear trend (9-12 a quarter in 2024, 14-17 since 2025Q1); the share
above 20% moves between 13% and 25% with no clear trend.

Example (Medallia, 2025Q1, flagged on both distress signs while marked at 90): 5.0% of Onex
Direct Lending's portfolio, 1.4% of Blackstone Private Credit Fund's ($1.0bn). The largest single
flagged positions in 2026Q2 are in small funds and sometimes a fund vehicle rather than an
operating company (Varagon Capital: Senior Direct Lending Program, 21%), so check names before
quoting them.

By BDC type (2026Q2): above 20% of portfolio, listed 23%, non-traded 17%, private 24%; one flagged
borrower above 5%, listed 16%, non-traded 13%, private 8%.

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

| Size quintile (median BDC-held par) | Mean lender families | 90th percentile | 2+ families | 6+ families |
|---|---|---|---|---|
| Q1 ($1.8M) | 1.03 | 1 | 3.0% | 0.0% |
| Q2 ($6.9M) | 1.13 | 2 | 10.6% | 0.0% |
| Q3 ($16M) | 1.27 | 2 | 19.3% | 0.1% |
| Q4 ($37M) | 1.57 | 3 | 34.3% | 0.8% |
| Q5 ($116M) | 2.66 | 6 | 58.9% | 10.3% |

Almost every heavily crowded borrower is in the top size quintile, so the crowding flag ranks each
borrower within its own size quintile and quarter (`warning_flags`). This matches the evaluation
result below (lender count adds nothing once size is controlled).

**BDC type.** Listed BDCs carry the most single-name exposure to flagged borrowers (by-type
numbers above), in line with their higher non-accrual rate (Fig 39). Exposure pooled across types partly measures
BDC type, so the final evaluation should test it within type.

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
