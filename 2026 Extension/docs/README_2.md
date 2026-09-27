# IXBRL Cleaning Pipeline

This repository contains a **rule-based data cleaning pipeline** for IXBRL investment data, focused on **interest rates, spreads, currency normalization, and SOFR-based estimation**.

The logic is split into two parts:
- `ixbrl_utils.py`: reusable cleaning utilities
- `run_ixbrl_pipeline.py`: orchestration script that runs the full pipeline

---

## High-Level Flow

1. Load IXBRL data (`ixbrl_clean.csv`) and drop share-count / amounts-only rows
2. Normalize interest rates and spreads to decimals
3. Normalize currencies and convert amounts to USD
4. Correct dollar-value scale errors in fair value / cost (e.g. thousands-vs-units tagging mistakes)
5. Drop filer-quarters whose fair value, cost and principal are wrong together
6. Drop subtotal / heading rows that would double-count fair value
7. Resolve rates with one ordered rule table: rate-type repairs → role repairs → fills → flags
8. Output a cleaned, reusable loan-level dataset for index construction and further analysis

---

## `ixbrl_utils.py`

This module contains **reusable cleaning functions**. Each function takes a DataFrame and returns
a modified copy. All of them are vectorised (no row-wise `apply`).

### Core Functions

**Interest rate normalization**
- `normalize_interest_columns` — rescales rates and spreads into consistent decimal form
  (divide by 100 or 10,000 to fit the valid range) and forces PIC/PIK/Floor non-negative
  (flagged `|sign_error`)
- `rescale_to_range` — shared power-of-ten rescaling used for both rates and dollar values

**Currency handling**
- `normalize_currency` — extracts the ISO code from an IXBRL unit reference (matched as a
  standalone token, so random hashes in unit IDs do not match a currency code)
- `determine_currency` — row currency, in priority order Principal > FairValue > Cost; USD if all missing
- `prepare_fx_data` — prepares quarterly FX rates
- `convert_currencies` — converts monetary values to USD and sets `currency`

**Dollar-value scale correction**
- `normalize_value_scale` — detects and corrects fair value / cost fields whose scale is
  inconsistent with the position's own principal amount, rescaling by 1,000 or 1,000,000

**When the principal is the mis-scaled field.** If fair value *and* cost both need the same
rescale, the principal may be the field that is off (e.g. principal tagged in thousands, or a
unit count). The check then compares the raw fair value with the same position's fair value in
its other rows (same `cik` + `investment_identifier`). If they agree within 3x, and multiplying the
principal by that scale brings FV / principal to within 3x of 1, the principal is multiplied
instead (`InvestmentOwnedBalancePrincipalAmount_normalized_scale_flag = mul1000`) and fair value /
cost are left as reported. The second condition stops a 10x mismatch (e.g. a principal converted
from a foreign-currency unit) from being "fixed" with a 1000x multiplier. On the current data this
applies to 111 rows, restoring $1.4B of fair value that was previously shrunk 1000x. Rows with no
history keep the old behaviour.

**When the principal is too large.** The fair-value check allows any FV / principal between 0 and 3,
so a principal that is 1000x too large (FV / principal near 0) passes it. That matters because the
index computes income as rate x principal. A second check therefore compares the principal with
**cost**: if principal / cost is about 1,000 or 1,000,000 and dividing lands it within 0.5-2x of cost
(`PRINCIPAL_COST_RANGE`), the principal is divided (`..._scale_flag = div1000`). Fair value and cost
are never changed, and the check is anchored on cost rather than fair value, so real drawdowns are
left alone: First Brands' fair value fell to 0.03x cost by 2026Q2 while its principal stayed at
1.0-1.1x cost, and none of its rows are touched. Positions with cost under $1M
(`PRINCIPAL_CHECK_MIN_COST`) are skipped, because a written-down cost basis (e.g. CLO equity at
$20k cost on $39.6M face) or an unfunded commitment can make a large ratio real. On the current
data it fixes 35 rows at 9 filers (Treasury bills, money-market funds, preferred equity reported in
units, and term loans with principal tagged 1000x). Two of them (Linxup and Digicert, CIK 1653384)
alone added about +0.25pp of income to the 2026Q1 index return; with both principal checks the
2026Q1 return is 0.86% instead of 1.24%, and the correlation with CDLI is 0.954 instead of 0.941.

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
ratio back inside the band, so a "fix" is always found. A ratio of 5 — which is not a units-tagging
error — is silently rescaled to 0.005, shrinking a real position by ~1000x. No row is ever flagged
`unresolved`.

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
formula that recovers what the filer actually meant. `flag_filer_quarter_outliers` (called from
`run_ixbrl_pipeline.run_pipeline` right after `normalize_value_scale`) instead flags and drops
the affected rows: for every filer with 3+ quarters on file, it compares each quarter's *total*
reported fair value to the median of that same filer's *other* quarters, and flags the whole quarter
when the ratio exceeds 20x. Validated against the full panel: exactly 5 (filer, quarter) pairs cross
that line, at 39x–431x — a clean separation, with the next-highest ratio for any other filer nowhere
close — and it does not catch positions that are genuinely large and simply persist (a real large
position recurs at a consistent scale every quarter for that filer, so it never produces one isolated
quarter wildly bigger than its own history). 182 rows removed out of 493,364 (0.037%), concentrated
in 5 filer-quarters. 2023Q1's reported total fair value alone drops from $558.6B to $234.1B once
removed, restoring a smooth, monotonically-plausible growth trajectory across the panel's full
history — and the index's correlation to CDLI *improves* (95.84% → 95.99%) once these rows stop
feeding `FV_prev` linkages for later quarters.

**Subtotal rows — `drop_subtotal_rows`.** Some filings tag a section heading as if it were a
position: CIK 1786108 reports "Portfolio Company Debt Securities" next to 200+ rows named
"Portfolio Company Debt Securities- United States … Torus Inc. …", with a fair value equal to
their sum. Left in, it doubles that filer's fair value for the quarter. A row is dropped when its
identifier is extended (after a separator such as `-`, `|`, `,`) by 20+ other identifiers in the
same filing and its fair value is within 5% of their sum: 5 rows, $5.8B. Issuer names that head
only 2–3 rows ("Vision Solutions, Inc." + "Vision Solutions, Inc., Emerald JV LP") are separate
holdings with different terms whose values happen to be close, so they are kept.

---

### Rate resolution: one rule table, two identities

All rate cleaning rests on two identities:

```text
(I1)  IR = PIC + PIK              coupon identity (all loans)
(I2)  IR ≈ max(Base_q + Spread, Floor)   floating-rate identity (floating or untagged loans)
```

`Base_q` is the quarterly average of the loan's own currency's base rate, read from
`SOFR_augmented.csv` by `load_base_rates`: SOFR (USD), 3-month EURIBOR (EUR), SONIA (GBP),
CORRA (CAD) and 3-month BBSW (AUD). `augment_rates.py` builds that file from `SOFR.csv` by
downloading the other series from the ECB, Bank of England (via FRED), Bank of Canada and RBA.
USD Prime-based loans use SOFR + 3.2% (`PRIME_OVER_SOFR`); Prime loans are recognised from the
rate-type tag or from "Prime + x%" / "Prime − x%" in the identifier, since most are untagged.
Rows in other currencies (SEK, JPY, CHF, …) get no base rate, and a base rate is never applied
to loans tagged fixed-rate. `Floor` is the reported interest
rate floor: a base-rate floor (0.5–2%) never binds against base + spread, so the same formula
covers base-rate floors and all-in floors ("Floor rate 9.85%").

Every change is one of three things:

1. **Role repair** — a field holds a value that belongs in another field, detected because moving
   it makes I1 or I2 hold.
2. **Fill** — one term of I1 / I2 is missing and the others are present.
3. **Flag** — the row contradicts I1 / I2 and no move fixes it. The value is left alone and the
   flag goes to `rate_flags`.

A row is only flagged when no identity pins down the right value. Anything that is *known* wrong
is repaired, because `rate_flags` is not read by the index code — a flagged value still enters the
index as-is.

The rules live in one ordered list, `RATE_RULES`. Each `Rule` is a tag, a row condition, and an
assignment; `apply_rules` runs them in order and records every rule that fires in
`change_tracker` (repairs and fills) or `rate_flags` (flags). Each rule sees the result of the
rules before it. Rows with an interest rate of 0 and no PIC/PIK coupon (unfunded revolvers, "—%"
rows) are left at 0 and flagged `zero_rate_unfunded`.

Counts below are from a full run on `ixbrl_clean.csv` (502,094 output rows).

**Rate-type repairs** (use `is_fixed`, from the XBRL rate-type tag, or the value's size when untagged)

| Rule | Condition | Change | Rows |
|---|---|---|---|
| `clear_spread_fixed_rate` | fixed loan, spread = IR | clear spread (the value is the coupon, median 13.6%) | 38 |
| `fixed_spread_is_rate` | fixed loan, only a spread | IR = spread, clear spread | 3 |
| `rate_is_spread_floating` | floating loan, spread = IR | IR = base + spread (the value is the spread, median 6.25%) | 242 |
| `rate_is_spread_untagged` | untagged loan, spread = IR, value 2–9% | IR = base + spread (same split as tagged loans: floating ≤ 7.5%) | 430 |
| `clear_spread_untagged_coupon` | untagged loan, spread = IR, value > 9% | clear spread (the value is the coupon; tagged fixed ≥ 10%) | 445 |

**Role repairs**

| Rule | Condition | Change | Rows |
|---|---|---|---|
| `spread_is_allin_rate` | IR missing, spread = PIC + PIK | IR = spread, clear spread | 1,163 |
| `rate_is_spread` | IR < PIC | spread = old IR (if no spread), IR = PIC + PIK | 4,066 |
| `pic_is_spread` | IR − PIC ≈ base rate, no spread/PIK | spread = old PIC, PIC = IR | 1,513 |
| `rate_is_spread_components_exceed` | PIC + PIK > IR, no spread | spread = old IR, IR = PIC + PIK | 18 |
| `rate_is_spread_pik_is_allin` | PIK > IR, no PIC, PIK ≈ base + IR | spread = old IR, IR = PIK | 92 |
| `rate_is_cash_component` | PIK > IR, no PIC (otherwise) | PIC = old IR, IR = old IR + PIK | 1,133 |
| `rate_below_spread_replaced` | IR < spread, no PIC/PIK | IR = base + spread (old IR was a floor or base rate, median 0.5–1%) | 494 |

**Fills**

| Rule | Condition | Change | Rows |
|---|---|---|---|
| `rate_from_base_plus_spread_plus_pik` | IR missing, spread + PIK, spread < PIK < base rate | PIK cannot be the all-in coupon, so it is paid on top: PIC = base + spread, IR = PIC + PIK ("Prime + 1.35%, Floor 9.85%, PIK 2.50%") | 672 |
| `rate_from_components` | IR missing, PIC/PIK present and above the spread (or no spread) | IR = PIC + PIK | 32,133 |
| `rate_from_base_plus_spread` | IR missing, spread present | IR = base + spread (`rate_source = estimated`) | 84,759 |
| `pic_from_rate_minus_pik` | IR and PIK present, PIC missing | PIC = IR − PIK | 47,516 |
| `pic_from_rate_minus_pik_reconcile` | all three present, I1 off | PIC = IR − PIK | 1,541 |

**Flags only** (`rate_flags`, values unchanged)

| Flag | Meaning | Rows |
|---|---|---|
| `rate_vs_base_plus_spread_gt_2pct` | IR ≥ spread but more than 2% away from base + spread (stale reset, spread field holding only part of the margin, etc.) | 12,211 |
| `zero_rate_unfunded` | IR = 0 with no PIC/PIK coupon | 4,964 |
| `spread_equals_rate` | spread = IR ≤ 2% on an untagged loan (ambiguous) | 477 |
| `rate_below_spread` | IR < spread although PIC + PIK = IR (which field is wrong is ambiguous) | 557 |
| `spread_out_of_range` | spread outside 0–20% (mostly real "Prime − x%" loans) | 85 |
| `pic_below_rate_unexplained` | PIC < IR with no PIK and no SOFR explanation | 51 |
| `coupon_identity_broken` | PIC + PIK ≠ IR after all repairs | 1 |

**Other helpers**
- `classify_rate_type` — adds `RateType` and `is_fixed` from
  `InvestmentVariableInterestRateTypeExtensibleEnumeration`, and `is_prime` from the tag or the
  identifier text. Only ~24% of rows carry the tag, so `is_fixed` is `<NA>` for the rest and those
  rows are treated as floating for base-rate purposes.
- `rate_config` — which of spread / rate / pik / pic are present on input (`check_2`)
- `resolve_rates` — runs the whole rate stage; rows with no rate field at all are dropped

### Known limitations

- **PIK reported as a share, not a rate.** 7 rows carry a PIK of 50% (0.50) — most likely "50% of
  interest paid in kind". This passes the 0–50% range check, so these rows still have PIK > IR.
- **`spread, pic` rows with PIC below the spread** (239 rows) get IR = SOFR + spread while PIC keeps
  its small value; the downstream `compute_final_interest_rates` then books the gap as PIK.
- **Preferred equity with a unit count as principal** that is not a clean 1000x (e.g. 483x cost)
  is left as reported; it adds about +0.25pp of income to the 2024Q4 index return.
- **Prime is approximated** as SOFR + 3.2% (the 2023–2026 quarterly gap is 3.2–3.4%), not read
  from a Prime series.
- **Loans in currencies without a base rate** (SEK, JPY, CHF, NOK, KRW, DKK, CNY) that report only
  a spread get no IR: 130 rows, 0.06% of fair value.
- **National Property REIT** (Prospect's affiliate, $6.6B) reports IR 4.25% (2.25% cash + 2% PIK)
  with a 0.25% spread — below SOFR, but plausibly a real affiliate rate, so it is left as reported.

---

## `run_ixbrl_pipeline.py`

This script **orchestrates the full workflow**.

### What It Does

- Loads IXBRL data and reference files (FX, base rates from `SOFR_augmented.csv`)
- Filters out share-count rows (except `context_type == "mixed"`, see below) and amounts-only / empty rows
- Runs normalization, currency conversion, value-scale correction, the outlier drop and the
  subtotal-row drop
- Calls `resolve_rates` and returns the cleaned dataset

Shares are normally an equity signal, but HPS Corporate Lending Fund's 2026Q1 10-Q tagged shares on
63.7% of its positions (mostly term loans with real rate/FV data), so `mixed` rows are kept even when
shares are present — the same fix as in `preprocessor.py`.

### Entry Point

```python
run_pipeline(
    data_path="ixbrl_clean.csv",
    fx_path="FX.csv",
    sofr_path="SOFR_augmented.csv",
)
```

Running the script directly writes the output to:

```text
ixbrl_cleaned_out.csv
```

A full run on the current dataset takes about 1.5–2.5 minutes.

---

## Key Columns Added

- `*_normalized` — cleaned numeric versions of raw fields
- `*_scale_flag` — `unchanged`, `div100` / `div10000` (rates) or `div1000` / `div1000000` (values),
  `unresolved`, or `na`. For `InvestmentOwnedAtFairValue_normalized_scale_flag` /
  `InvestmentOwnedAtCost_normalized_scale_flag`, `unresolved` is not produced while the rescale
  floor is 0 (see above), and `na` means there was no principal amount to check against
- `currency` — row currency after `determine_currency`
- `RateType` — base-rate member from the XBRL rate-type tag (e.g. `SecuredOvernightFinancingRateSofrMember`, `FixedRateMember`)
- `is_fixed` — `True` / `False` from `RateType`, `<NA>` when untagged
- `is_prime` — Prime-based loan (from `RateType` or the identifier text)
- `InvestmentOwnedBalancePrincipalAmount_normalized_scale_flag` — `mul1000` / `mul1000000` when the
  principal was too small, `div1000` / `div1000000` when it was too large relative to cost, else
  `unchanged` / `na`
- `rate_config` / `check_2` — which rate fields were present on input (e.g. `spread, rate, pik`)
- `rate_source` — where the final IR came from: `reported`, `derived` (from other fields or a
  role repair), `estimated` (base + spread), `zero`, or `none`
- `check_1` — `resolved` if the row ends with a usable non-zero IR, else `unresolved`
- `estimate` — max(base + spread, floor) for USD floating/untagged rows with a spread
- `change_tracker` — pipe-delimited tags of every repair / fill applied
- `rate_flags` — pipe-delimited data-quality flags (values left unchanged)

---

## Design Principles

- Rule-based (no ML, no black-box imputation)
- Repair only when an identity pins down the value; flag when it is ambiguous
- Fully auditable: every change is recorded in `change_tracker`, every doubt in `rate_flags`
- One rule table: adding a rule is one `Rule(...)` entry, not a new function

---

## Assumptions

- Missing currency ⇒ assumed USD
- Base-rate estimation uses the loan's currency (SOFR — or SOFR + 3.2% for Prime —, EURIBOR, SONIA,
  CORRA, BBSW) and never applies to rows tagged fixed-rate
- Floating-point tolerance: `1e-6`; "differs by the base rate" tolerance: 50bp

---

## Goal

The goal of this pipeline is to produce **clean, consistent, and auditable interest rate data** suitable for downstream financial analysis and research of private credit markets.
