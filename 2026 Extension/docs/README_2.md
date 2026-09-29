# IXBRL Cleaning Pipeline

This repository contains a **rule-based data cleaning pipeline** for IXBRL investment data, focused on **interest rates, spreads, currency normalization, and base-rate estimation**.

The logic is split into two parts:
- `ixbrl_utils.py`: reusable cleaning utilities
- `run_ixbrl_pipeline.py`: orchestration script that runs the full pipeline

---

## High-Level Flow

1. Load IXBRL data (`ixbrl_clean.csv`) and drop share-count / amounts-only rows
2. Normalize interest rates and spreads to decimals
3. Normalize currencies and convert amounts to USD
4. Correct dollar-value scale errors in fair value / cost (e.g. thousands-vs-units tagging mistakes)
5. Rescale positions whose fair value, cost and principal are all 1000x too large together
6. Drop subtotal / heading rows that would double-count fair value
7. Resolve rates with one ordered rule table: rate-type repairs → role repairs → fills
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
- `fix_component_scale` — row-level check on PIC / PIK: a component more than 10pp above the row's
  coupon that fits under it once divided by 100 was tagged in percent (a raw 0.50 meaning 0.50%)
  and is divided by 100 (13 PIK values, `|row_div100`); PIC = PIK = 0.5 is read as a 50/50 split of
  IR (1 row, `|split_50_50`). Without it these rows feed negative cash coupons to the index.
- `rescale_to_range` — shared power-of-ten rescaling used for both rates and dollar values

**Currency handling**
- `normalize_currency` — extracts the ISO code from an IXBRL unit reference (matched as a
  standalone token, so random hashes in unit IDs do not match a currency code)
- `determine_currency` — row currency, in priority order Principal > FairValue > Cost; USD if all missing
- `prepare_fx_data` — prepares quarterly FX rates
- `convert_currencies` — converts monetary values to USD and sets `currency`. A principal tagged in a
  different unit from fair value and cost is normal (a EUR loan reported in USD); it is treated as a
  filer unit error only when converting it is what breaks it, i.e. read in the fair-value currency it
  sits within 3x of fair value and converted it does not, and the two currencies are at least 2x
  apart. Those principals are read in the fair-value currency (`unit_mismatch_fixed`): 17 rows at 5
  filers, e.g. CIK 1976336 tagging USD principals as EGP. Left alone, the loan would be labelled EGP
  and the value-scale check would divide its fair value and cost by 1,000.

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
applies to 111 rows, and keeps $1.4B of fair value that dividing it by 1,000 would have removed.
Rows with no other quarter to compare against are rescaled on fair value / cost as usual.

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
alone added about +0.25pp of income to the 2026Q1 index return. With both principal checks the
2026Q1 return is 0.86% and the correlation with CDLI 0.954; without them, 1.24% and 0.941.

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

It is **left at 0.0**. Raising it costs ~0.30pp of correlation against CDLI
(0.9617 -> 0.9587) and ~0.9bp of tracking error, because the 438 newly-`unresolved` rows then enter
the fair-value-weighted index at face value instead of being shrunk to near-zero weight. Those rows
are genuinely bad data — median FV/principal of 10x, p90 of 194x, max of 286,048x — so the floor of 0
is currently doing the right thing for the wrong reason: it suppresses them by accident rather than
by rule. The principled fix is to raise the floor *and* drop or winsorize `unresolved` rows at the
index stage; until that is decided, the floor stays at 0.

**A gap the ratio check structurally cannot close — `rescale_filer_quarter_outliers`.**
`normalize_value_scale` catches a fair value or cost that is wrong *relative to its own row's
principal*. It cannot catch positions where fair value, cost **and** principal are all 1000x too
large together: every row's internal ratio still looks plausible. Found this way: TCW Direct
Lending VIII LLC's 2023Q1 filing (CIK 1825265) reported individual loan positions at **$30–46B
each**, larger than the entire BDC industry's typical quarterly total, while the same filer's other
13 quarters sit in the tens of millions per position.

The fix works at two levels. `flag_filer_quarter_outliers` compares each quarter's *total* fair
value with the median of the same filer's *other* quarters (filers with 3+ quarters) and flags the
quarter when the ratio exceeds 20x. On this data exactly 6 (filer, quarter) pairs cross that line,
with a clean gap to every other filer. Inside a flagged quarter only a few positions (2–14) are
wrong, so `rescale_filer_quarter_outliers` divides fair value, cost and principal by 1,000 on each
position larger than 25% of the fund's usual quarterly total (`OUTLIER_POSITION_SHARE`), tagging
`|filer_quarter_div1000`, and keeps the correctly scaled rows. `flag_filer_quarter_outliers` then
runs again as a fallback and drops any quarter that is still an outlier. On this data 39 positions
are rescaled and no quarter remains an outlier, so nothing is dropped. 2023Q1's total fair value
falls from $565.2B to $241.9B. A genuinely large position recurs at a consistent scale every
quarter, so it never produces one isolated quarter far above its own history.

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
(I2)  IR ≈ Base_q + Spread, with floors   floating-rate identity (floating or untagged loans)
```

`Base_q` is the quarterly average of the loan's own currency's base rate, read from
`SOFR_augmented.csv` by `load_base_rates`: SOFR (USD), 3-month EURIBOR (EUR), SONIA (GBP),
CORRA (CAD), 3-month BBSW (AUD), 3-month STIBOR (SEK), TONA (JPY), SARON (CHF), 3-month NIBOR
(NOK), 91-day CD (KRW), 3-month CIBOR (DKK), 3-month SHIBOR (CNY), 3-month BKBM (NZD) and
3-month compounded SORA (SGD). `augment_rates.py` builds
that file from `SOFR.csv` from the official publishers (see its docstring). Negative SARON / TONA
are kept in the file but floored at 0% for loans, which floor the benchmark at zero.
USD Prime-based loans use SOFR + 3.2% (`PRIME_OVER_SOFR`); Prime loans are recognised from the
rate-type tag or from "Prime + x%" / "Prime − x%" in the identifier, since most are untagged.
Rows in other currencies (e.g. EGP) get no base rate, and a base rate is never applied
to loans tagged fixed-rate. `Floor` is the reported interest rate floor. A floor below the
spread is a base-rate floor, so the estimate is max(Base_q, Floor) + Spread; a floor at or above
the spread is an all-in floor ("Floor rate 9.85%"), so the estimate is max(Base_q + Spread,
Floor). Base-rate floors bind when the benchmark is near or below them: CHF loans with a 0.75%
floor report exactly spread + 0.75% while SARON is negative, and 2023-vintage USD loans with 4–5%
SOFR floors report floor + spread once SOFR fell below it (88% match exactly).

Every change is one of two things:

1. **Role repair** — a field holds a value that belongs in another field, detected because moving
   it makes I1 or I2 hold.
2. **Fill** — one term of I1 / I2 is missing and the others are present.

A row that contradicts I1 / I2 with no move that fixes it is left as reported.

The rules live in one ordered list, `RATE_RULES`. Each `Rule` is a tag, a row condition, and an
assignment; `apply_rules` runs them in order and records every rule that fires in
`change_tracker`. Each rule sees the result of the rules before it. Rows with an interest rate of
0 and no PIC/PIK coupon (unfunded revolvers, "—%" rows) are left at 0 with `rate_source = zero`.

**Why each rule is kept.** Each rule was switched off one at a time and the output rescored on the
consistency check below and on the index. Every rule in the table breaks I1, I2 or a sign on a
measurable share of rows when switched off; the "Without it" column shows what breaks.

Counts below are from a full run on `ixbrl_clean.csv` (502,299 output rows).

**Rate-type repairs** (use `is_fixed`, from the XBRL rate-type tag, or the value's size when untagged)

| Rule | Condition | Change | Rows | Without it |
|---|---|---|---|---|
| `fixed_spread_is_rate` | fixed loan, only a spread | IR = spread, clear spread | 3 | 3 loans with no rate |
| `rate_is_spread_floating` | floating loan, spread = IR | IR = base + spread (the value is the spread, median 6.25%) | 259 | +234 I2 failures |
| `rate_is_spread_untagged` | untagged loan, spread = IR, value 2–9% | IR = base + spread (same split as tagged loans: floating ≤ 7.5%) | 498 | +463 I2 failures |
| `clear_spread_untagged_coupon` | untagged loan, spread = IR, value > 9% | clear spread (the value is the coupon; tagged fixed ≥ 10%) | 445 | +320 I2 failures (0.11% of FV) |

**Role repairs**

| Rule | Condition | Change | Rows | Without it |
|---|---|---|---|---|
| `spread_is_allin_rate` | IR missing, spread = PIC + PIK | IR = spread, clear spread | 1,164 | +80 I1 failures, 10 loans with no rate |
| `rate_is_spread` | IR < PIC | spread = old IR (if no spread), IR = PIC + PIK | 4,066 | +3,979 I1 failures, 3,918 negative coupons in the index |
| `pic_is_spread` | IR − PIC ≈ base rate, no spread/PIK | spread = old PIC, PIC = IR | 1,514 | +1,514 I1 failures |
| `rate_is_spread_components_exceed` | PIC + PIK > IR, no spread | spread = old IR, IR = PIC + PIK ("1M SOFR + 16.00% (0.00% Cash + 20.65% PIK)") | 18 | +11 rows with IR < PIK |
| `rate_is_cash_component` | PIK > IR, no PIC | PIC = old IR, IR = old IR + PIK | 1,223 | +1,004 I1 failures, 1,096 negative coupons |
| `rate_below_spread_replaced` | IR < spread, no PIC/PIK | IR = base + spread (old IR was a floor or base rate, median 0.5–1%) | 498 | +205 I2 failures |
| `rate_is_cash_part_of_base_plus_spread` | IR, PIK and spread, no PIC; IR + PIK = base + spread within 50bp while IR alone is not | PIC = old IR, IR = old IR + PIK (PIK carved out of the margin: "SOFR + 6.00% (3.25% PIK)" tagged IR 6.42%, PIK 3.25%) | 3,973 | coupons understated by their PIK; index coupon −1.3bp; +3,042 rows where IR is more than 2pp from base + spread |

**Fills**

| Rule | Condition | Change | Rows | Without it |
|---|---|---|---|---|
| `rate_from_base_plus_spread_plus_pik` | IR missing, spread + PIK, spread < PIK < base rate | PIK cannot be the all-in coupon, so it is paid on top: PIC = base + spread, IR = PIC + PIK ("Prime + 1.35%, Floor 9.85%, PIK 2.50%") | 674 | coupon −3.6bp, returns move up to 1.6bp |
| `rate_from_components` | IR missing, PIC/PIK present and above the spread (or no spread) | IR = PIC + PIK | 32,129 | 14,988 loans with no rate (2.8% of FV), 8,358 negative coupons |
| `rate_from_base_plus_spread` | IR missing, spread present | IR = base + spread with floors (`rate_source = estimated`) | 87,556 | 87,347 loans with no rate (23% of FV); correlation with CDLI 0.954 → 0.932 |
| `pic_from_rate_minus_pik` | IR and PIK present, PIC missing | PIC = IR − PIK | 43,913 | +24,299 I1 failures (8.6% of FV) |
| `pic_from_rate_minus_pik_reconcile` | all three present, I1 off | PIC = IR − PIK | 1,541 | +1,541 I1 failures |

`fix_component_scale` (in `normalize_interest_columns`) stays for the same reason: without it, 7
PIK values tagged in percent (0.50 meaning 0.50%) pass through as 50% and feed 6 negative cash
coupons to the index.

**Other helpers**
- `classify_rate_type` — adds `RateType` and `is_fixed` from
  `InvestmentVariableInterestRateTypeExtensibleEnumeration`, and `is_prime` from the tag or the
  identifier text. Only ~24% of rows carry the tag, so `is_fixed` is `<NA>` for the rest and those
  rows are treated as floating for base-rate purposes.
- `rate_config` — which of spread / rate / pik / pic are present on input (`check_2`)
- `resolve_rates` — runs the whole rate stage; rows with no rate field at all are dropped

### Consistency check

The rate stage succeeds when every row with a usable rate satisfies I1, or I2 where I1 cannot be
checked, with every component nonnegative:

- **I1** (IR = PIC + PIK within 1bp) is checked wherever PIC or PIK is reported, and takes precedence.
- **I2** (IR ≈ base + spread with floors, within 2%) is checked only where I1 cannot be. Below about
  1%, gaps are benchmark noise: reset dates, 1M vs 3M term SOFR, and 10–26bp credit spread
  adjustments on the quarterly-average base.
- **Sign**: IR, PIC, PIK and spread ≥ 0, except "Prime − x%" loans, whose negative spread is real.

On the current output (496,502 rows with a usable rate, excluding 5,797 zero-rate rows):

| Check | Rows checked | Fails | FV share of fails |
|---|---|---|---|
| No IR | all | 0 | 0.000% |
| Negative IR, PIC or PIK | all | 0 | 0.000% |
| Negative spread, not Prime-minus | all | 9 | 0.000% |
| I1 off by more than 1bp | 74,663 | 339 | 0.006% |
| I2 off by more than 2% | 397,800 | 4,787 | 0.340% |
| **Passes every rule** | 496,502 | — | 99.57% of all FV pass |

24,039 rows (2.65% of FV) report IR only (fixed-rate or no spread), so neither identity applies.
The I2 failures are reported coupons left as filed: most sit above base + spread (an untagged PIK
or a spread field holding part of the margin), the rest below it (mostly foreign loans reported in
USD units and priced off SOFR). The I1 failures are mostly undrawn or partly drawn delayed-draw
loans whose PIC holds a 0.25–1% commitment fee.

Where both identities can be checked (a PIC or PIK and a spread), I1 is kept and I2 is not
enforced. IR differs from base + spread by more than 2pp on 5,339 such rows (1.09% of FV); most
carry a PIK paid on top of base + spread (IR = base + spread + PIK), which is correct as filed.

### Known limitations

- **`spread, pic` rows with PIC below the spread** get IR = base + spread while PIC keeps
  its small value (about 1%, a commitment fee or a floor in the wrong field); the downstream
  `compute_final_interest_rates` then books the gap as PIK. These are the only rows where the
  estimate overrides I1; setting IR = PIC would give ~2% coupons on loans priced near 10%.
- **Preferred equity with a unit count as principal** that is not a clean 1000x (e.g. 483x cost)
  is left as reported; it adds about +0.25pp of income to the 2024Q4 index return.
- **Prime is approximated** as SOFR + 3.2% (the 2023–2026 quarterly gap is 3.2–3.4%), not read
  from a Prime series.
- **Currency comes from the unit tag.** A EUR loan that the filer reports in USD is treated as USD,
  so it is estimated off SOFR rather than EURIBOR (e.g. Pineapple German Bidco). About 279 estimated
  coupons on EUR / GBP / CAD loans are affected (0.05% of FV); reading the benchmark from the
  rate-type tag or identifier would move 99 I2 failures and the index coupon by 0.04bp, so the
  unit is kept as the single source of currency.
- **Commitment fees on undrawn loans.** Some delayed-draw loans report their 0.25–1% unused fee as
  the coupon or as PIC; they account for most of the 339 I1 failures but have fair value at or
  below zero, so they carry no index weight.
- **STIBOR, NIBOR and CIBOR are OECD monthly averages** (via FRED), because the administrators
  license the daily fixings; the quarterly value is the mean of three monthly averages. SHIBOR 3M
  for CNY is a choice, since no filing tags a CNY benchmark.
- **High floors on estimated loans.** 28 USD rows with no reported coupon and a 6–10% floor below
  the spread (e.g. Verano, Dreamfields, likely Prime floors) are estimated at floor + spread; no
  reported coupon confirms it.
- **National Property REIT** (Prospect's affiliate, $6.6B) reports IR 4.25% (2.25% cash + 2% PIK)
  with a 0.25% spread — below SOFR, but plausibly a real affiliate rate, so it is left as reported.

---

## `run_ixbrl_pipeline.py`

This script **orchestrates the full workflow**.

### What It Does

- Loads IXBRL data and reference files (FX, base rates from `SOFR_augmented.csv`)
- Filters out share-count rows (except `context_type == "mixed"`, see below) and amounts-only / empty rows
- Runs normalization, currency conversion, value-scale correction, the filer-quarter outlier
  rescale (with the drop as a fallback) and the subtotal-row drop
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

A full run on the current dataset takes about 3–4 minutes.

---

## Key Columns Added

- `*_normalized` — cleaned numeric versions of raw fields
- `*_scale_flag` — `unchanged`, `div100` / `div10000` (rates) or `div1000` / `div1000000` (values),
  `unresolved`, or `na`. For `InvestmentOwnedAtFairValue_normalized_scale_flag` /
  `InvestmentOwnedAtCost_normalized_scale_flag`, `unresolved` is not produced while the rescale
  floor is 0 (see above), and `na` means there was no principal amount to check against
- `currency` — row currency after `determine_currency`; it also selects the loan's base rate
- `unit_mismatch_fixed` — principal read in the fair-value currency (see `convert_currencies`)
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
- `estimate` — base + spread with floors (see I2) for floating/untagged rows with a spread and a
  base rate
- `change_tracker` — pipe-delimited tags of every repair / fill applied

---

## Design Principles

- Rule-based (no ML, no black-box imputation)
- Repair only when an identity pins down the value; otherwise leave the reported value
- Fully auditable: every change is recorded in `change_tracker`
- Every rule earns its place: removing it must break I1, I2 or a sign on a measurable share of rows
- One rule table: adding a rule is one `Rule(...)` entry, not a new function

---

## Assumptions

- Missing currency ⇒ assumed USD
- Base-rate estimation uses the loan's currency (SOFR — or SOFR + 3.2% for Prime —, EURIBOR, SONIA,
  CORRA, BBSW, STIBOR, TONA, SARON, NIBOR, CD 91-day, CIBOR, SHIBOR, BKBM, compounded SORA), taken
  from the value unit, and never applies to rows tagged fixed-rate
- A loan's base rate is floored at 0% (negative SARON / TONA do not pass through), and a reported
  floor below the spread is a base-rate floor
- Floating-point tolerance: `1e-6`; "differs by the base rate" tolerance: 50bp

---

## Goal

The goal of this pipeline is to produce **clean, consistent, and auditable interest rate data** suitable for downstream financial analysis and research of private credit markets.
