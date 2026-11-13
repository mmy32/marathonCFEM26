# Risk Monitor: Warning Signs and BDC Exposure

Borrowers default; a BDC is hurt when a borrower it holds defaults, in proportion to how much of
its portfolio that borrower is. So the warning signs are measured on the **borrower**, and BDC
risk is measured second, as **exposure** to the flagged borrowers (midterm slides 11-15).

```
python scripts/run_risk_monitor.py      # about 3 minutes
```

Code: `src/8_risk_monitor/` (`monitor_panel.py` builds the loan-quarter panel on the tranche
chains of `11_tranche_matching.md`; `distress_event_study.borrower_quarter_panel` pools it to
borrowers, see `13_distress_event_study.md`; `signal_tails.py` flags and charts). Reads only
`FINAL_DATASET.csv` and writes `outputs/risk_monitor/signals/`:

| File | Contents |
|---|---|
| `borrower_warning_flags.csv`, `warning_tail_summary.csv` | Each sign per borrower-quarter (0/1, plus the rank behind the ranked signs) and the share of borrowers flagged per quarter |
| `warning_crowding.png`, `crowding_distribution.csv` | Share of borrowers and of BDC-held debt by number of lender families, each quarter |
| `warning_disagreement.png`, `disagreement_percentiles.csv` | Median, 90th and 95th percentile of the gap between lenders' marks, by borrowers and by dollars |
| `warning_pik.png` | Borrowers paying more than half their interest in kind; borrowers switching to PIK each quarter |
| `bdc_exposure.csv`, `bdc_exposure_by_quarter.csv`, `bdc_exposure.png` | Each BDC's exposure to flagged borrowers: share of portfolio, largest single flagged borrower |

## How a sign is shown and flagged

Each sign is shown as its whole distribution each quarter, counted by borrowers and by dollars of
BDC-held debt, which shows whether the tail is growing. A borrower is flagged by rank: in the top
10% of the quarter (crowding: of its size fifth of BDC-held par, since big borrowers have more
lenders). Ties are never split, so whole-number counts flag a little under 10%. A rank always flags
about 10%: it says who is in the tail, not whether the tail grows.

**Crowding** (`C1_families`), share of borrowers / of BDC-held par by number of manager families:

| Families | Borrowers, 2023Q1 | Borrowers, 2026Q2 | Par, 2023Q1 | Par, 2026Q2 |
|---|---|---|---|---|
| 1 | 78.9% | 72.5% | 51.0% | 39.4% |
| 2 | 12.7% | 14.6% | 19.4% | 17.3% |
| 3 | 4.3% | 5.7% | 8.7% | 11.9% |
| 4-5 | 3.0% | 4.5% | 12.1% | 12.6% |
| 6+ | 1.0% | 2.7% | 8.7% | 18.8% |

The far tail moved most: borrowers with six or more lender families hold 19% of BDC-held par,
up from 9%.

**Disagreement on value** (`V1_range`: the par-weighted range of the families' marks on loans held
by 2+ families; only those borrowers, 25% of all), cents per dollar owed:

| Quarter | Median, borrowers | 90th pct, borrowers | Median, dollars | 90th pct, dollars | 95th pct, dollars |
|---|---|---|---|---|---|
| 2023Q1 | 1.1 | 5.6 | 2.0 | 5.6 | 7.0 |
| 2024Q4 | 0.5 | 2.8 | 0.9 | 3.9 | 6.1 |
| 2025Q4 | 0.5 | 3.0 | 0.7 | 3.6 | 5.8 |
| 2026Q2 | 0.6 | 4.0 | 1.2 | 6.3 | 9.3 |

By dollars the 2026Q2 tail is the widest in the panel; by borrowers it is up from the 2024 low.

| Sign (per borrower) | Column | Flagged when | Share of borrowers flagged |
|---|---|---|---|
| Crowding | `C1_families` | top 10% of its size fifth | 4.6-6.8% |
| Disagreement on value | `V1_range` | top 10% of the quarter (borrowers with 2+ families) | 10% of those borrowers |
| PIK, more than half | `pik_share` | more than half the coupon paid in kind | 3.1% (2023Q1) rising to 4.5% (2026Q2) |
| PIK switch this quarter | `toggle_par_share` | a loan that paid cash the quarter before starts PIK, or its PIK share rises | 2.0-2.6% a quarter, no trend |
| PIK switch in the past year | `toggle_par_share` | a switch this quarter or in the three before | 6.2-6.8% from 2024Q1 |

**Distress sign** = disagreement in the top 10%, or a PIK switch in the past year: 8.0-8.6% of
borrowers each quarter from 2024Q1. Crowding is not a distress sign (it says how many BDCs a
default reaches, not how likely it is), nor is paying more than half in kind (some loans are
written that way).

## BDC exposure

Per BDC-quarter (all instruments, fair value > 0, BDCs with 20+ borrowers): the share of its
portfolio in borrowers flagged on each sign (`exposure_<sign>`, `exposure_distress` for any
distress sign), its largest distress-flagged borrower (`top_flagged`, `top_flagged_weight`) and
the HHI of its book (`concentration`).

| Quarter | BDCs | Median exposure | BDCs above 20% of portfolio | BDCs with one flagged borrower above 5% |
|---|---|---|---|---|
| 2024Q1 | 119 | 12.5% | 16 (13%) | 11 |
| 2024Q4 | 132 | 13.7% | 33 (25%) | 12 |
| 2025Q4 | 150 | 11.3% | 20 (13%) | 16 |
| 2026Q2 | 158 | 12.4% | 36 (23%) | 17 |

The clear trend is single-name exposure: 9-12 BDCs a quarter in 2024 had one flagged borrower
above 5% of the portfolio, and 14-17 since 2025Q1. Example: Medallia in 2025Q1, flagged on both
distress signs while marked at 90, was 5.0% of Onex Direct Lending's portfolio. The largest flagged
positions are sometimes in fund vehicles rather than operating companies, so check names before
quoting them.

## Limitations

- **Quarterly, lagged filings:** a quarter's holdings are public 45-90 days after quarter end.
- **Marks are the managers' own appraisals,** and they are smoothed.
- **Growth dilution:** BDC fair value more than doubled over the panel ($241bn to $528bn), so rates
  are diluted by new loans. Signals that need history rise mechanically in 2023; compare quarters
  from 2024Q1.
- **No borrower fundamentals:** filings do not disclose leverage or interest coverage. This is an
  outside-in monitor of what managers disclose about their holdings.
- **Non-accrual status is unknown** for filings that report it only in HTML (14% of fair value).

Which of these signs predict non-accrual, and by how much, is in `14_early_warning_model.md`.
The market benchmarks and the evaluation layer this monitor used to run are in
`ORIE 5220/miscellaneous/` with the old version of this doc.
