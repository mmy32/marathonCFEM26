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
