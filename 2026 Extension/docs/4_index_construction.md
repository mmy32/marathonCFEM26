# Flow-Based BDC Loan Index

`src/2_index_construction/index_construction.py` builds a quarterly return index of the loans
BDCs hold and compares it with the Cliffwater Direct Lending Index (CDLI), and its senior-loan
version with CDLI-S (midterm slide 8).

```
python src/2_index_construction/index_construction.py
```

Reads `data/final_dataset/FINAL_DATASET.csv` and `data/reference/cdli.csv`, `cdli-s.csv`. Writes
`outputs/index_construction_plots/index_vs_cdli_panels.png` (all loans vs CDLI, senior loans vs
CDLI-S) and `index_vs_cdli_stats.csv`, and prints the quarterly return decomposition and the
statistics below.

## How it is built

1. **Load** (`load_and_prepare_investment_data`): rows with a fair value; numeric aliases `FV`,
   `COST`, `PAR`, `rate_cash`, `rate_pik`, `interest_rate`.
2. **Rates** (`compute_final_interest_rates`): cash (`PIC_Final`), PIK (`PIK_Final`) and total
   (`IR_Final`) rates, inferring a missing component from the other two.
3. **Positions** (`add_position_key`, `aggregate_positions`): a position is a BDC's holding in one
   borrower at one seniority, keyed on the validated borrower name plus `instrument_seniority`.
   A BDC's rows with the same key in a quarter are combined: values summed, rates weighted by
   principal.
4. **Flows** (`compute_position_level_flows`): each position is lagged to its previous quarter,
   and its return is the change in fair value, net of the change in cost, plus cash and PIK interest,
   over last quarter's fair value (`ret_pos_flow`).
5. **Index** (`compute_market_weights`, `compute_flow_based_market_index`): positions are weighted
   by last quarter's fair value, extreme returns are clipped, and returns are compounded from
   100. `aggregate_return_decomposition` splits each quarter into price, cash-interest and
   PIK-interest parts.
6. **Comparison** (`benchmark_stats`, `compute_tracking_error`,
   `compute_annualized_return_and_vol`): correlation, tracking error and yearly returns against
   the benchmark. `lag1_autocorrelation` and `compute_unsmoothed_return_and_vol` show how much
   appraisal smoothing hides volatility.

**The first quarter (2023Q1) is left out of every statistic** (`start_idx`): no position has a
prior quarter, so its return is undefined.

## Results (2023Q2-2026Q2, 13 quarters)

| | All loans vs CDLI | Senior loans vs CDLI-S |
|---|---|---|
| Correlation | 94.6% | 94.7% |
| Tracking error, annualized | 0.41% | 0.37% |
| 2023 (three quarters), index / benchmark | 9.5% / 9.2% | 9.5% / 9.3% |
| 2024 | 11.0% / 11.3% | 11.2% / 11.3% |
| 2025 | 9.6% / 9.3% | 9.7% / 9.2% |
| 2026 H1 | 2.7% / 3.0% | 3.0% / 2.9% |

The all-loan index returns 10.1% a year with 1.25% volatility. Returns are strongly
autocorrelated (0.66), as appraised marks are; de-smoothed, the volatility is 2.8%.

## Why positions match on borrower name, not `investment_identifier`

The identifier is free text, and BDCs change it between filings: most switched format in their
FY2025 10-Ks (`Active Day, Inc., One stop 1` became `Active Day, Inc. | One stop 1 |
Non-Affiliated Issuer`), and some embed the quarter's rate. Matched on identifier text, positions
linked to a prior quarter carried only 58-73% of fair value (17% in 2025Q4). Matched on borrower
+ seniority they carry 86-94% every quarter, and the index tracks CDLI slightly better
(correlation 94.6% against 94.3%).

## Known issues fixed

- **Values in the wrong unit scale.** A few filings tag fair value or cost in thousands instead of
  units; because the index weights by fair value, even one such row can dominate a quarter.
  `ixbrl_utils.normalize_value_scale()` corrects them upstream (`2_ixbrl_cleaning.md`).
- **Annualizing.** `compute_annualized_return_and_vol()` derives the number of years from the
  returns it is given (after `start_idx`). Do not pass `num_years` unless you mean to override it.

## Limitations

- Two tranches of one borrower at the same seniority are one position.
- When a BDC relabels a position's seniority between quarters, the link breaks for that quarter.
- The index uses BDC marks, which are smoothed (see the autocorrelation above).

The return-decomposition, SOFR-comparison and entry/exit charts were retired on 2026-11-08; the
old version of this doc is in `ORIE 5220/miscellaneous/`.
