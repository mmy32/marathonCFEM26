# IXBRL Cleaning Pipeline

This repository contains a **rule-based data cleaning pipeline** for IXBRL investment data, focused on **interest rates, spreads, currency normalization, and SOFR-based estimation**.

The logic is split into two parts:
- `ixbrl_utils.py`: reusable cleaning utilities
- `run_ixbrl_pipeline.py`: orchestration script that runs the full pipeline

---

## High-Level Flow

1. Load raw IXBRL data
2. Normalize interest rates and spreads
3. Normalize currencies and convert amounts to USD
4. Classify fixed vs floating rate investments
5. Apply multiple deterministic cleaning passes
6. Resolve partially-missing rate configurations
7. Estimate missing rates using SOFR (USD only)
8. Output a cleaned, reusable loan-level dataset for index construction and further analysis

---

## `ixbrl_utils.py`

This module contains **reusable cleaning functions**. Each function takes a DataFrame and returns a modified copy.

### Core Functions

**Interest rate normalization**
- `normalize_interest_columns` — rescales rates and spreads into consistent decimal form
- `_normalize_series` — applies column-level scaling logic

**Currency handling**
- `determine_currency` — infers currency from IXBRL unit references
- `prepare_fx_data` — prepares quarterly FX rates
- `convert_currencies` — converts monetary values to USD

**Initialization & classification**
- `initialize_clean_dataframe` — adds required audit and tracking columns
- `classify_rate_types` — classifies fixed vs floating rate investments
- `add_check_2` — records which rate fields are present in each row

**Core rate cleaning**
- `perform_initial_swap` — fixes common spread / rate misplacements
- `clean_fixed_rate_data` — enforces `IR = PIC + PIK` for fixed-rate loans
- `perform_additional_corrections` — applies high-confidence cleanup rules
- `clean_additional_rate_issues` — handles secondary inconsistencies
- `fix_pic_from_ir_minus_pik` — recomputes PIC when IR and PIK are known
- `fix_ir_lt_pic` — fixes invalid rate relationships

**Unresolved-case handlers**
- `unresolved_rate_pic`
- `unresolved_rate_pik_pic`
- `unresolved_spread_rate`
- `unresolved_spread_pik`
- `unresolved_spread_pic`
- `unresolved2_spread_rate_pik`

These functions resolve rows based on their `check_2` configuration using deterministic rules. When PIC and PIK are present, they are enforced to sum to IR. If either PIC or PIK is missing, the missing value is computed using the difference from IR. Additional sanity checks detect and correct column misuse (e.g., cases where Spread is mistakenly populated in the IR field). If only Spread is present, 3M SOFR is used to estimate IR and derive subsequent values.

**SOFR-based estimation**
- `add_estimate_with_sofr` — estimates missing rates using SOFR for USD loans

**Row dropping**
- `drop_fully_missing_rate_rows` — drops rows with no usable rate information

---

## `run_ixbrl_pipeline.py`

This script **orchestrates the full workflow**.

### What It Does

- Loads raw IXBRL data and reference files (FX, SOFR)
- Calls utility functions in a fixed, explicit order
- Applies multiple cleaning and resolution passes
- Produces a final cleaned dataset

### Entry Point

```python
run_pipeline(
    data_path="ixbrl_clean.csv",
    fx_path="FX.csv",
    sofr_path="SOFR.csv",
)
```

The final output is written to:

```text
ixbrl_cleaned_out.csv
```

---

## Key Columns Added

- `*_normalized` — cleaned numeric versions of raw fields
- `change_tracker` — pipe-delimited tags describing applied fixes
- `check_1` — resolved / unresolved status
- `check_2` — which rate components are present
- `check_3` — SOFR estimation quality flags
- `estimate` — SOFR-based estimated rate (when applicable)

---

## Design Principles

- Rule-based (no ML, no black-box imputation)
- Conservative: prefer tagging over aggressive fixing
- Fully auditable: every change is explicitly recorded
- Modular: utilities can be reused or reordered if needed

---

## Assumptions

- Missing currency ⇒ assumed USD
- SOFR estimation applies only to USD-denominated rows
- Floating-point tolerance: `1e-6`

---

## Goal

The goal of this pipeline is to produce **clean, consistent, and auditable interest rate data** suitable for downstream financial analysis and research of private credit markets.

