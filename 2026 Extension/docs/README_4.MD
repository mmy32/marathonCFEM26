# Flow-Based Market Investment Return Index and Decomposition Toolkit

## Overview

This codebase provides a complete end-to-end pipeline for analyzing investment-level quarterly data and constructing a flow-based market return index. The framework is designed to support return attribution, benchmark comparison, and interest rate contextual analysis.

Key capabilities include:

- Loading and cleaning investment-level quarterly data
- Constructing cash (PIC) and PIK interest rates from incomplete disclosures
- Computing position-level cash flows, valuation changes, and returns
- Aggregating investment-weighted market returns
- Decomposing returns into price, cash interest, and PIK interest components
- Building a flow-based market index
- Comparing index returns to the CDLI benchmark
- Analyzing correlations, tracking error, and interest rate context (SOFR / EFFR)
- Producing publication-ready visualizations

The implementation uses **Polars** for high-performance data processing, **Pandas** for interoperability, and **Matplotlib** for visualization.

---

## Dependencies

### Required Python Packages

- `polars`
- `pandas`
- `numpy`
- `matplotlib`
- `typing`

### Optional Packages

- `openpyxl` (required for reading Excel-based interest rate data)

> **Python 3.9 or later is recommended.**

---

## Data Expectations

### Investment-Level CSV (Primary Input)

The primary investment dataset must contain the following columns.

#### Identifiers

- `cik`
- `investment_identifier`
- `cal_q`  
  Format: `"YYYYQ#"` (e.g., `"2021Q3"`)

#### Valuation Fields

- `InvestmentOwnedAtFairValue_normalized`
- `InvestmentOwnedAtCost_normalized`
- `InvestmentOwnedBalancePrincipalAmount_normalized`

#### Interest Rate Fields (May Be Partially Missing)

- `InvestmentInterestRatePaidInCash_normalized`
- `InvestmentInterestRatePaidInKind_normalized`
- `InvestmentInterestRate_normalized`

---

### Benchmark Data

- CSV file containing:
  - `Quarter`
  - `CDLI`

---

### Interest Rate Data

- Excel file containing:
  - `Date`
  - `RateName`
  - `Value`

`RateName` must include:
- `EFFR`
- `SOFR`

---

## Pipeline Summary

### 1. Load and Prepare Data

**Function**
- load_and_prepare_investment_data()
**Description** 

- Loads the investment CSV into Polars
- Filters observations with valid fair values
- Casts numeric fields to Float64
- Creates standardized aliases:
    - FV    
    - COST
    - PAR
    - rate_cash
    - rate_pik
    - interest_rate

### 2. Data Quality Diagnostics

**Functions**

missing_percentage_summary()  
interest_rate_null_combinations()

**Description**

- Computes the percentage of missing values for selected columns  
- Analyzes all null and non-null combinations across interest rate fields  

---

### 3. Interest Rate Construction

**Function**

compute_final_interest_rates()

**Logic**

- Combines disclosed cash, PIK, and total interest rates  
- Infers missing components when sufficient information is available  
- Drops observations where all interest rate fields are missing  

**Outputs**

- PIC_Final  
- PIK_Final  
- IR_Final  

---

### 4. Quarter Sorting and Flow Calculations

**Helper Functions**

qsort_expr()  
safe_div()

**Main Function**

compute_position_level_flows()

**Description**

- Orders observations by calendar quarter  
- Computes lagged fair value and cost  
- Calculates:  
  - Fair value changes  
  - Cost changes  
  - Cash interest income  
  - PIK interest income  
- Computes flow-based position returns:  
  - ret_pos_flow  

**Known limitation**: the quarter-over-quarter lag is grouped by `(cik, investment_identifier)`
only, with no tiebreaker for two distinct positions that share identical description text within
one quarter — see "Known Limitations — Open" below.

---

### 5. Panel Structure Diagnostics

**Function**

quarter_count_distribution()

**Description**

- Reports the number of observed quarters per (CIK, investment_identifier)  
- Used to assess panel coverage and survivorship  

---

### 6. Entry and Exit AUM Analysis

**Function**

quarterly_cik_entry_exit_aum()

**Description**

- Aggregates assets under management (AUM) by CIK and quarter  
- Identifies first and last appearances of each CIK  
- Computes:  
  - AUM entering the market  
  - AUM exiting the market  
  - Number of entering and exiting CIKs  

---

### 7. Market Weight Construction

**Function**

compute_market_weights()

**Description**

- Flags valid return observations  
- Computes total lagged market fair value per quarter  
- Constructs investment-level market weights (w_mkt)  

---

### 8. Market-Level Return Decomposition

**Function**

aggregate_return_decomposition()

**Produces**

- Price return contribution  
- Cash interest contribution  
- PIK interest contribution  
- Average cash and PIK interest rates  
- Number of active investments  

---

### 9. Return Decomposition Visualization

**Function**

plot_quarterly_return_decomposition()

**Description**

- Stacked quarterly bar chart  
- Separates:  
  - Price return (positive and negative)  
  - Cash interest return  
  - PIK interest return  
- Annotated percentage contributions  
- Publication-ready formatting  

---

### 10. Flow-Based Market Index

**Function**

compute_flow_based_market_index()

**Description**

- Aggregates investment-weighted returns by quarter  
- Clips extreme returns to stabilize the index  
- Builds a cumulative index level  
- Default base level: 100  

---

### 11. Benchmark Comparison

**Function**

plot_index_vs_cdli_returns()

**Description**

- Plots quarterly index returns against CDLI benchmark returns  
- Ensures consistent quarter alignment  
- Return y-axes start at zero (`ylims=(0.0, 4.0)`, and `ylim=(0.0, 4.0)` in
  `plot_index_cdli_sofr`), so bar and line heights are read against a true zero baseline rather
  than an offset one that visually exaggerates differences  
- The default title is `"Index Returns QoQ(%)"`. It is deliberately not "Sector Index ...": this
  function plots the *market-level* index. The sector and senior-loan variants pass their own
  title explicitly  

---

### 12. Performance Metrics

**Functions**

compute_annualized_return_and_vol()  
compute_tracking_error()

**Metrics Computed**

- Geometric annualized return  
- Annualized volatility  
- Correlation with CDLI  
- Quarterly and annualized tracking error  

**The first index quarter is excluded from all four.**
The index's first quarter has no prior-quarter fair value to difference against, so its return is
structurally undefined; any non-zero value there comes from same-quarter duplicate rows standing in
for a prior period. Every consumer of the return series drops it, via a `start_idx` argument:

| function | `start_idx` | drops |
|---|---|---|
| `compute_annualized_return_and_vol` | 1 | first index quarter |
| `plot_index_vs_cdli_returns` | 1 | first index quarter |
| `compute_tracking_error` | 2 | leading rates-only row + first index quarter |
| `plot_index_cdli_sofr` | 2 | leading rates-only row + first index quarter |

`compute_tracking_error` and `plot_index_cdli_sofr` operate on `plot_df`, which carries one extra
leading row from the quarterly-rates file, so `start_idx=2` there is the same cut as `start_idx=1`
on the raw index series. Pass `start_idx=0` to keep the full series.

On the current 14-quarter panel (2023Q1-2026Q2), excluding 2023Q1 moves the headline figures from
10.84% / 1.54% to **10.29% annualized return / 1.25% annualized volatility**. Volatility falls
more than the return does because that quarter's 4.28% was a large outlier against a series that
otherwise sits in the 0.9-3.3% band. Correlation with CDLI (96.2%) and tracking error (0.36%) are
unchanged, since those already excluded it.

---

### 13. Interest Rate Context

**Functions**

load_quarterly_rates()  
build_quarterly_comparison_df()  
plot_index_cdli_sofr()

**Description**

- Loads daily SOFR and EFFR data  
- Computes quarterly average rates  
- Plots index returns, CDLI returns, and SOFR together  

---

### Typical Usage Order

load_and_prepare_investment_data  
compute_final_interest_rates  
compute_position_level_flows  
compute_market_weights  
aggregate_return_decomposition  
plot_quarterly_return_decomposition  
compute_flow_based_market_index  
plot_index_vs_cdli_returns  
load_quarterly_rates  
build_quarterly_comparison_df  
compute_tracking_error  
plot_index_cdli_sofr  

---

### Known Issues Fixed

**Upstream dollar-value scale errors (fixed in the cleaning pipeline, see `README_2.md`).**
A handful of filings tag `InvestmentOwnedAtFairValue` / `InvestmentOwnedAtCost` in the wrong unit
scale (e.g. thousands instead of units), inflating a position's fair value by ~1000x. Because
`compute_market_weights()` weights by prior-quarter fair value, even a few such rows can dominate
an entire quarter's aggregate return — this was traced back as the primary cause of an unexplained
divergence between the index and CDLI in a specific quarter. `ixbrl_utils.normalize_value_scale()`
now catches and corrects this upstream, before the data ever reaches this pipeline.

**`compute_annualized_return_and_vol()` — `num_years` is derived, not hardcoded.** It used to
default to a fixed `10`, which silently produced the wrong annualized return whenever that did not
match how many periods were actually in `returns` — passing `10` for a 10-*quarter* (2.5-year)
series annualizes over 10 years instead, compressing an 11% return down to ~3%. It now defaults to
`None` and is computed from the returns actually used, i.e. **after `start_idx` is applied**. Do
not pass it explicitly unless you mean to override that: computing `len(returns) / 4` at the call
site and passing it in re-introduces the bug in a subtler form, annualizing a 13-quarter series
over a 14-quarter span.

**Selecting `cols` vs `cols_llm` when `flag_llm=True`.** When loading the LLM-enriched dataset
(`OUTPUT_PATH_LLM`), select `cols_llm` (which includes `sector` and `instrument_seniority`), not
the base `cols` list — otherwise those columns get dropped immediately after loading, and the
sector/seniority sub-index sections fail with a `ColumnNotFoundError` despite the enrichment step
having run successfully.

---

### Known Limitations — Open (not fixed)

**`compute_position_level_flows()` — same-quarter duplicate `(cik, investment_identifier)` keys
make `FV_prev` order-dependent.** `FV_prev = pl.col("FV").shift(1).over(["cik",
"investment_identifier"])` lags each position within its own group, sorted by quarter. The group
key doesn't disambiguate two *different* positions that happen to share identical description text
within the same filer and the same quarter — a real, observed pattern (e.g. two tranches of the
same facility disclosed with byte-identical `investment_identifier` text). When that happens,
`.sort()`'s tie-break for same-quarter rows is not guaranteed stable across runs whose upstream row
order differs — a new, entirely unrelated filing landing anywhere earlier in the combined wide
table can reshuffle it — so one of the two rows can get the *other's* fair value treated as its
"prior quarter," even though both are the same quarter.

For any quarter after a position's true first appearance, this only misattributes a small amount of
return between the two duplicate siblings; the aggregate quarterly return is essentially unaffected.
Confirmed by re-running the full pipeline after adding 74 unrelated new 2026Q2 filings: total fair
value per quarter matched to floating-point precision for every quarter except the one that actually
gained new rows, and position-level differences elsewhere were on the order of 1–3bp.

**2023Q1 is fully exposed to this.** It's the index's first quarter, so no position in it has a
*genuine* prior-quarter fair value at all — the entire market-level "return" reported for 2023Q1 is
built up from this same duplicate-tie mechanism, applied across every affected group, with nothing
real behind it. Confirmed empirically: the same 2026Q2-only re-run, with zero change to any 2023Q1
row's own data, moved 2023Q1's reported return from +4.28% to +6.35%. This is why every stats-facing
consumer of the return series (`compute_tracking_error`, `compute_annualized_return_and_vol`,
`plot_index_vs_cdli_returns`, `plot_index_cdli_sofr`) already excludes 2023Q1 via `start_idx` — that
convention predates this finding, but the finding is concrete proof it was the right call, not just
a theoretical one.

**Scope, measured on the 493,364-row cleaned dataset (2026-09-07):** 4,957 `(cik,
investment_identifier, cal_q)` triples share a key with at least one other row — 10,787 rows, 2.19%
of the panel.

**The fix, not yet applied:** add a stable secondary key to `group_cols` in
`compute_position_level_flows()` — `context_id` or `accession` — so genuinely distinct positions
never share a lag group even when their description text collides. Left open since it changes index
construction output, which was out of scope for the pass that found it.

---

### Design Notes

- Polars is used for performance on large investment panels  
- All calculations are aligned at the quarterly frequency  
- Returns are flow-based rather than price-only  
- Safe division prevents spurious infinities  
- Missing interest rates are handled explicitly and transparently  

---

### Outputs

- Clean investment-level panel with computed returns  
- Quarterly market return decomposition  
- Flow-based market index and index levels  
- Benchmark comparison plots  
- Tracking error and correlation statistics  
- Interest rate context plots  
