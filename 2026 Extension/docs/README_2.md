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
4. Correct dollar-value scale errors in fair value / cost (e.g. thousands-vs-units tagging mistakes)
5. Classify fixed vs floating rate investments
6. Apply multiple deterministic cleaning passes
7. Resolve partially-missing rate configurations
8. Estimate missing rates using SOFR (USD only)
9. Output a cleaned, reusable loan-level dataset for index construction and further analysis

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

**Dollar-value scale correction**
- `normalize_value_scale` — detects and corrects fair value / cost fields whose scale is
  inconsistent with the position's own principal amount
- `_choose_value_scale` — picks the correcting power-of-ten scale (1, 1,000, or 1,000,000),
  mirroring the same approach used for rate normalization

Some filings tag `InvestmentOwnedAtFairValue` / `InvestmentOwnedAtCost` in a different unit
scale than `InvestmentOwnedBalancePrincipalAmount` for the same investment (e.g. thousands vs.
units), producing a fair value that is ~1000x too large. Since a single loan is never realistically
worth billions of dollars, `normalize_value_scale` compares `FV / PrincipalAmount` and
`Cost / PrincipalAmount` against a plausible range and rescales using the position's own
principal amount as the anchor — the same "compare against a known-good field" principle
already used for spread/rate normalization, just applied to dollar amounts. Rows without a usable
principal amount (e.g. equity positions reported via share counts) are left unchanged rather than
guessed at. Uncorrected, this error was large enough to visibly distort quarterly index returns
(see `README_4.MD`): a handful of ~1000x-inflated positions could single-handedly swing an entire
quarter's aggregate return, since they entered the market-value-weighted average with a wildly
overstated weight.

**Two bands.** The check uses two named ranges:

- `VALUE_RATIO_RANGE = (0.0, 3.0)` — the *pass-through* band. A ratio in here is plausible as
  reported and the row is left alone. The floor is 0 because deeply marked-down positions
  legitimately sit near zero.
- `VALUE_RESCALE_TARGET_RANGE = (0.0, 3.0)` — the band a *rescaled* value must land in for the
  correction to be accepted. **Currently set equal to the pass-through band.**

### Known limitation: the rescale floor

Because `VALUE_RESCALE_TARGET_RANGE` has a floor of 0, dividing by 1,000 lands *any* out-of-band
ratio back inside the band, so `_choose_value_scale` always finds a "fix". A ratio of 5 — which is
not a units-tagging error — is silently rescaled to 0.005, shrinking a real position by ~1000x.
No row is ever flagged `unresolved`.

Raising the floor to 0.3 restricts rescaling to genuine ~1000x / ~1e6x mistags. Measured on the
current dataset, that moves 415 fair-value rows out of `div1000` and into `unresolved`
(1,430 -> 1,015 corrected, 438 flagged) and shifts total fair value by +0.014%.

It is **deliberately left at 0.0** for now. Raising it costs ~0.30pp of correlation against CDLI
(0.9617 -> 0.9587) and ~0.9bp of tracking error, because the 438 newly-`unresolved` rows then enter
the fair-value-weighted index at face value instead of being shrunk to near-zero weight. Those rows
are genuinely bad data — median FV/principal of 10x, p90 of 194x, max of 286,048x — so the floor of 0
is currently doing the right thing for the wrong reason: it suppresses them by accident rather than
by rule. The principled fix is to raise the floor *and* drop or winsorize `unresolved` rows at the
index stage; until that is decided, the floor stays at 0.

**A gap the ratio check structurally cannot close — `flag_filer_quarter_outliers`.**
`normalize_value_scale` catches a fair value or cost that is wrong *relative to its own row's
principal*. It cannot catch a filing where fair value, cost, **and** principal are all wrong
*together*, by the same non-round factor — every row's internal ratio still looks plausible, so
nothing gets flagged. Found this way: TCW Direct Lending VIII LLC's 2023Q1 filing (CIK 1825265)
reported individual loan positions at **$30–46B each** — larger than the entire BDC industry's
typical quarterly total — while the same filer's other 13 quarters, on the very same three fields,
sit in the tens-of-millions range per position. `normalize_value_scale` correctly saw a plausible
FV/principal ratio on every one of those rows and left them alone.

No rescale is possible here (unlike the thousands-vs-units case above): the ratio between the
2023Q1 values and the filer's own normal scale was ~664x — not a round power of ten — so there is no
formula that recovers what the filer actually meant. `flag_filer_quarter_outliers` (in
`ixbrl_utils.py`, called from `run_ixbrl_pipeline.run_pipeline` right after `normalize_value_scale`)
instead flags and drops the affected rows: for every filer with 3+ quarters on file, it compares
each quarter's *total* reported fair value to the median of that same filer's *other* quarters, and
flags the whole quarter when the ratio exceeds 20x. Validated against the full panel: exactly 5
(filer, quarter) pairs cross that line, at 39x–431x — a clean separation, with the next-highest
ratio for any other filer nowhere close — and it does not catch positions that are genuinely large
and simply persist (a real large position recurs at a consistent scale every quarter for that filer,
so it never produces one isolated quarter wildly bigger than its own history). 182 rows removed out
of 493,364 (0.037%), concentrated in 5 filer-quarters. 2023Q1's reported total fair value alone drops
from $558.6B to $234.1B once removed, restoring a smooth, monotonically-plausible growth trajectory
across the panel's full history — and the index's correlation to CDLI *improves* (95.84% → 95.99%)
once these rows stop feeding `FV_prev` linkages for later quarters.

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
- `InvestmentOwnedAtFairValue_normalized_scale_flag` / `InvestmentOwnedAtCost_normalized_scale_flag`
  — `unchanged` (ratio inside the pass-through band), `div1000` / `div1000000` (scale error
  corrected), `unresolved` (ratio implausible and no scale lands it back in the target band — the
  value is left as reported; not produced while the rescale floor is 0, see above), or `na`
  (no principal amount to check against)
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

