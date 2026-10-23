# IXBRL Cleaning Pipeline

This repository contains a **rule-based data cleaning pipeline** for IXBRL investment data, focused on **interest rates, spreads, currency normalization, and base-rate estimation**.

The logic is split into two parts:
- `ixbrl_utils.py`: reusable cleaning utilities
- `run_ixbrl_pipeline.py`: orchestration script that runs the full pipeline

---

## High-Level Flow

1. Load IXBRL data (`ixbrl_clean.csv`); drop share-count rows unless `context_type == "mixed"`, and drop amounts-only and empty rows
2. Normalize interest rates and spreads to decimals
3. Normalize currencies and convert amounts to USD
4. Correct dollar-value scale errors in fair value / cost (e.g. thousands-vs-units tagging mistakes)
5. Rescale positions whose fair value, cost and principal are all 1000x too large together
6. Drop subtotal / heading rows that would double-count fair value
7. Resolve rates with one ordered rule table: rate-type repairs → role repairs → fills; rows with no rate field at all (spread, rate, PIK, PIC) are dropped
8. Output a cleaned, reusable loan-level dataset for index construction and further analysis

### Row counts (`ixbrl_clean.csv` → `ixbrl_cleaned_out.csv`)

Only three steps remove rows on the current data. Every other step changes values only, and the
filer-quarter outlier drop (the fallback after step 5) removes nothing.

| Step | Rows dropped | Rows left |
|---|---|---|
| Input (`ixbrl_clean.csv`) | | 608,680 |
| 1. Share-count / amounts-only / empty rows | 91,610 (all `empty`: no amount, rate or term; the other two filters already ran in `preprocessor.py`) | 517,070 |
| 6. Subtotal / heading rows | 5 ($5.8B of fair value) | 517,065 |
| 7. Rows with no rate field (`rate_config == "none"`) | 14,766 (110 BDCs; mostly warrants, equity and undrawn revolvers, including 4,089 share-count rows kept as `mixed` in step 1) | **502,299** |

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
- The reverse case is also handled there: a principal in a high-denomination currency tagged as
  something else. Spruce Bidco's yen tranche is tagged `jpy` by three filers but `usd` / `cny` by
  CIK 1872371, 1974793 and 2052152, so it read ~157x cost. When raw principal / cost (USD) is within
  10% of a currency's rate for the quarter, the principal is re-read in that currency
  (`unit_mismatch_fixed`): 12 rows, about $10B of principal. Only currencies at 50-500 per USD that
  some filer actually tags a principal in (JPY, ISK) are candidates. At 20-35 per USD, CLO equity and
  mortgage notionals match a rate by coincidence; at ~1,400 per USD (KRW) a rate match cannot be told
  from a 1000x scale error. Positions with cost under $1M are skipped, because below $1M
  mortgage-trust notionals at 160-176x cost match the yen rate too; the one Spruce Bidco row this
  leaves out (CIK 2052153, 2025Q1, $705k cost) is in `MANUAL_CORRECTIONS` (see below).
- `COUNTRY_TO_CURRENCY` includes Iceland (ISK) and "Korea" (KRW, the FX file's name for South Korea).
  Without them Advania Sverige's `isk` principal (8 rows, $11.7B) was read as USD and Delivery Hero's
  `U_KRW` principal (8 rows, ~$135M each) converted to NaN.

**Dollar-value scale correction**
- Three thresholds are shared by every check on fair value, cost and principal: two amounts *agree*
  when their ratio is within 0.5-2x (`AGREE_RANGE`), a ratio is *exactly* 1,000x or a currency rate
  when it is within 5% (`EXACT_BAND`; 10% for an FX rate, which drifts), and two amounts are *out of
  line* when they are more than 3x apart (`OUT_OF_LINE`). Several rules used their own bands
  (0.8-1.25x, 0.9-1.1, 20x, 12%, 30%); moving them onto these shared values left
  `ixbrl_cleaned_out.csv` byte-for-byte identical.
- `normalize_value_scale` — detects and corrects fair value / cost fields whose scale is
  inconsistent with the position's own principal amount, rescaling by 1,000 or 1,000,000
- `cap_equity_unit_principal` — an equity position (identifier names preferred, equity, units,
  warrants, common, shares or membership) whose principal exceeds 100x cost is reporting a unit
  count, not a face amount: principal is set to cost (`|equity_units_capped`). 24 rows on the
  current data (17 more that sit at exactly 1000x cost are now divided by 1,000 upstream, with the
  same result). Two preferred stakes (Veronica Holdings / Vapotherm at 457x cost,
  VardimanBlack Holdings at 2,449x) otherwise added ~0.9pp of cash and ~0.5pp of PIK income to the
  2024Q4 Healthcare sub-index and 0.19pp to the 2024Q4 market index. When the cap was introduced,
  correlation with CDLI was 94.49% against 95.38% without it: the fictitious income had happened to
  move 2024Q4 toward CDLI.
- The same function handles the error the other way round: a share or unit count far *below* the
  dollar amounts (PG&E: 230,000 shares against $10.7M of cost; preferred units; ABS certificates).
  When cost and fair value agree with each other (0.5-2x) and cost exceeds 3x principal
  (`UNITS_VALUE_RATIO`), the principal is not a dollar amount and is set to cost
  (`|units_as_principal`): 146 rows at 27 filers, $280M of cost. There is no text test, because a
  debt principal is never a third of its own cost with fair value still at cost. Fair value and cost
  on these rows had been flagged `unresolved` against the old principal and are reset to `unchanged`.
- `fix_value_sign` — a funded position tagged with a negative cost, or a negative cost and fair
  value. A real negative cost is the unamortized fee on an undrawn commitment: small against the
  principal, with fair value at or below zero. Two cases are flipped to the absolute value
  (`|sign_error` on each changed field), 79 rows at 18 filers, $203.2M of cost:
  - *cost only* (30 rows, $145.5M): principal and fair value are both positive and |cost| agrees
    with each of them (0.5-2x). 24 rows are at CIK 1895316 in 2023Q2-2024Q3 (TouchFuse term loan at
    cost -$20.9M against $21.4M of principal and fair value; +$21.0M the quarter before and +$20.9M
    after).
  - *cost and fair value* (49 rows, $57.6M of fair value): both are negative and each agrees in size
    with a positive principal. 11 rows at CIK 1895316 and CIK 1916099 carry $57.3M of it and were
    checked against the filing: the 10-K prints the amounts without parentheses (Action Elevator
    delayed-draw loan, CIK 1916099 2025Q4: cost 26,471, fair value 25,957) while the XBRL fact has
    `sign="-"`; genuine negatives in the same filings are printed in parentheses. The other 38
    rows are under $0.2M each ($0.4M in total) and were not checked against their filings.
  With both cases flipped, position costs sum to the reported total within $0.1M in all six
  CIK 1895316 filings and exactly at CIK 1916099.
- `MANUAL_CORRECTIONS` — four rows that are wrong in a way no general rule separates from real
  positions, keyed by `(accession, context_id)`: the Spruce Bidco yen tranche under the $1M minimum
  (principal re-read in JPY), and three loans whose cost and fair value are tagged 1000x too small
  against a plausible principal (Chartwell Cumming term loan and delayed-draw loan, CIK 1959568
  2023Q4; Hellers delayed-draw loan, CIK 1655050 2024Q4; flag `manual_mul1000`). Written-down CLO
  equity has the same shape (cost a thousandth of face), which is why this is a list and not a rule.
- One more named correction in `MANUAL_CORRECTIONS` (`units_as_principal`): Equitrans Midstream preferred
  equity, CIK 1675033 2023Q1. The filing prints 250,000, a share count, in the principal column against
  $5.3M of cost; at 47x cost it is under the 100x equity cap, so the principal is set to cost by name
  (`|units_as_principal`). Left alone it added about 0.4pp to the Energy & Utilities yield that quarter.
- `manual_principal_currency.csv` — 171 more named corrections, loaded into `MANUAL_CORRECTIONS` when
  `ixbrl_utils` is imported: principals the filing reports in a local currency next to USD cost and
  fair value, but tags in another currency (USD; GBP or EUR for the krona loans). 139 rows print the
  currency on the row ("£ 129,900", "C$ 99,924", "kr 319,308"); 32 carry a footnote on the row
  ("Principal amount is denominated in Canadian dollars", "The Par value is denoted in CAD"). The raw
  principal is re-read in that currency: 86 GBP, 64 CAD, 12 AUD, 6 NOK and 3 SEK rows, after which
  principal / cost has a median of 0.98-1.11 per currency (before: 0.78 GBP, 1.40 CAD, 1.48 AUD, 11-12
  NOK / SEK). Each row was checked against its filing and the evidence is kept in the file. No rule
  does this safely: near parity the ratio looks like a premium or a part-drawn loan, and lowering
  `MISTAG_FX_MIN` to reach SEK / NOK re-reads 62-212 rows that are mostly CLO equity and part-drawn
  USD loans. The row's `currency` becomes the reported one, so its base-rate estimate changes too
  (171 `estimate` values, 69 final rates). Not corrected: 7 rows ($78M of principal) with the same
  ratio and a CDOR / CORRA / SONIA / BBSY rate in the identifier, where neither the row nor its
  footnotes state a currency (one filing says par is USD unless noted), and USD-tagged EUR loans,
  where the rate is too close to 1.

**When a whole filing tags the principal in the wrong unit — `fix_filing_principal_scale`.** Five
filings tag *every* principal in thousands while fair value and cost are in dollars (CIK 1784700
2023Q1 / Q3 / Q4, 1948368 2026Q1, 2012139 2026Q2). Row by row that looks like fair value and cost
1000x too large, and the row-level check used to divide them: 1784700's 2023Q4 fair value went from
$683M to $0.7M, about $3.5B across the five filings. Position-history matching (next paragraph)
cannot catch it, because these filers' identifiers embed text that changes every quarter ("Debt
Investments - 168.5%"). The decision is therefore made per filing: when at least 50% of its rows
have value / principal at ~1,000 (`FILING_SCALE_SHARE`) and the filing's fair-value total is within
3x of the filer's median in its other quarters (`FILING_FV_HISTORY`), fair value is right and the
principal is rescaled (`..._scale_flag = filing_mul1000`), including partly funded loans whose ratio
is not near 1,000. Rows already at ~1x are left alone. 959 rows; principal / cost on them has a
median of 1.007-1.011.

**When the principal is the mis-scaled field.** If fair value *and* cost both need the same
rescale, the principal may be the field that is off (e.g. principal tagged in thousands, or a
unit count). The check then compares the raw fair value with the same position's fair value in
its other rows (same `cik` + `investment_identifier`). If they agree within 3x, and multiplying the
principal by that scale brings FV / principal to within 3x of 1, the principal is multiplied
instead (`InvestmentOwnedBalancePrincipalAmount_normalized_scale_flag = mul1000`) and fair value /
cost are left as reported. The second condition stops a 10x mismatch (e.g. a principal converted
from a foreign-currency unit) from being "fixed" with a 1000x multiplier. On the current data this
applies to 39 rows (it was 111 before the filing-level check above took the whole-filing cases).
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
alone added about +0.25pp of income to the 2026Q1 index return. When these principal checks were
introduced the 2026Q1 return fell from 1.24% to 0.86%.

Under $1M of cost the ratio alone proves nothing, so the same division is applied only when three
more things hold: principal / cost is 1,000 within 5% (`PRINCIPAL_SMALL_COST_BAND`), fair value is
within 0.5-2x of cost (an unfunded commitment has fair value at or below zero), and the principal is
more than 5x the filer's median principal that quarter (`PRINCIPAL_SMALL_COST_SIZE`). Halcyon CLO
equity at 990x cost is an ordinary-sized principal for its filer (1.1x the median) and is left alone;
a Northern Institutional money-market fund at $514M principal on $514k of cost (168x the median), a
Coreweave delayed-draw loan and two Rapid Displays loans are divided. 21 rows: those 4 ($1.6B of
principal) plus 17 equity rows that `cap_equity_unit_principal` used to set to cost.

**When the fair value or cost is too small.** A small or negative fair value is normally a real
mark, so the ratio check never touches it. The exception is a fair value of exactly cost / 1,000 on
a position whose principal agrees with cost (0.5-2x): fair value x 1,000 / cost within 0.95-1.05
(`FV_SMALL_BAND`) is multiplied by 1,000
(`InvestmentOwnedAtFairValue_normalized_scale_flag = mul1000`). On the current data this is 16 rows
at 4 filers, $178M of fair value (CIK 1655050 2024Q1 alone: nine loans, $131M of cost carried at
$0.13M). The band separates cleanly: the mistags sit at 0.955-1.037, while genuine write-offs
(First Brands, Interface Security) are at 0.73 or below, or 1.11 or above. The same test with fair
value and cost swapped multiplies a cost of exactly fair value / 1,000
(`InvestmentOwnedAtCost_normalized_scale_flag = mul1000`): 4 small rows at CIK 1784700.

Some filings tag `InvestmentOwnedAtFairValue` / `InvestmentOwnedAtCost` in a different unit
scale than `InvestmentOwnedBalancePrincipalAmount` for the same investment (e.g. thousands vs.
units), producing a fair value that is ~1000x too large. Since a single loan is never realistically
worth billions of dollars, `normalize_value_scale` compares `FV / PrincipalAmount` and
`Cost / PrincipalAmount` against a plausible range and rescales using the position's own
principal amount as the anchor — the same "compare against a known-good field" principle
already used for spread/rate normalization, just applied to dollar amounts. Rows without a usable
principal amount (e.g. equity positions reported via share counts) are left unchanged rather than
guessed at. Uncorrected, this error was large enough to visibly distort quarterly index returns
(see `4_index_construction.md`): a handful of ~1000x-inflated positions could single-handedly swing an entire
quarter's aggregate return, since they entered the market-value-weighted average with a wildly
overstated weight.

**Two bands.** The check uses two named ranges:

- `VALUE_RATIO_RANGE = (0.0, 3.0)` — the *pass-through* band. A ratio in here is plausible as
  reported and the row is left alone. The floor is 0 because deeply marked-down positions
  legitimately sit near zero.
- `VALUE_RESCALE_TARGET_RANGE = (1/3, 3.0)` — the band a *rescaled* value must land in for the
  correction to be accepted: a 1e3 / 1e6 fix has to bring the ratio back to about 1.

### The rescale floor and `unresolved` rows

The target band used to have a floor of 0. Dividing by 1,000 then landed *any* out-of-band ratio
back inside it, so a "fix" was always found: a cost 5x principal, which is not a units-tagging
error, was silently rescaled to 0.005x, shrinking a real position by 1000x (298 rows, $454M of
cost). The floor is now 1/3, so only genuine ~1000x / ~1e6x mistags are rescaled (105 + 6
fair-value rows, 121 cost rows). Anything else out of band is flagged `unresolved` and **left as
reported**. That was 408 rows; 146 of them were unit counts tagged as principal and are now repaired
by `cap_equity_unit_principal` (`|units_as_principal`), leaving 217 fair-value and 70 cost flags on
262 rows, $236M of fair value.

What the remaining `unresolved` rows are (12 further rows fit none of these):

| Pattern | Rows | Fair value | Reading |
|---|---|---|---|
| Fair value > 3x principal, principal ≈ cost | 100 | $108M | Mixed: equity-like upside on small debt positions, and large negative marks on small principals. Not a round scale error. |
| Cost > 3x principal, fair value differs from cost | 58 | $34M | Cost basis includes more than the tagged principal (Eating Recovery Center), or warrants with a nominal principal. |
| Fair value > 3x principal, other | 92 | $97M | Preferred equity and warrants with a face amount unrelated to value. |

The cleaning goal is a consistent dataset, so these rows are kept at face value rather than shrunk
to make the index fit. Moving the floor took the correlation with CDLI from 0.9449 to 0.9430 with
tracking error unchanged at 0.44%.

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

### Consistency after cleaning, and what is left

Of the 407,653 rows with a positive principal and a positive cost, 98.09% (98.99% by principal) have
principal within 1/3-3x of cost and fair value no more than 3x cost. A small or negative fair value
counts as consistent: it is a mark, not an error. The median principal / cost is 1.009.

| Change | Rows | Index effect |
|---|---|---|
| Rescale floor 0 -> 1/3, filing-level principal fix | 959 + 408 | +6.8bp in 2025Q4, under 0.2bp elsewhere |
| ISK / mis-tagged yen principals | 20 | -2 to -5bp per quarter 2023Q2-2024Q2 (phantom income removed) |
| Fair value 1000x too small | 16 | +2.3bp 2024Q1, -1.5bp 2024Q2 |
| KRW mapping, small-cost principal, cost 1000x too small | 33 | under 0.3bp |
| Unit count as principal, four manual corrections | 150 | under 0.03bp |

Correlation with CDLI over the whole sequence went 0.9449 -> 0.9408; tracking error stayed at 0.44%.

Known inconsistencies that are **not** fixed, because no rule separates them from real positions:

- **Principal > 3x cost: 7,368 rows, $51.6B of principal.** Mostly real: revolvers and delayed-draw
  loans carried at the funded amount or at a small fee balance, CLO equity and mortgage-trust
  certificates with a notional principal. 68 rows ($5.3B) sit above 20x cost on a position with cost
  of $1M or more and fair value near cost; most are structured-finance notionals, and none matches
  a currency rate or a round 1e3 / 1e6 scale.
- **The 262 `unresolved` rows** above: real but unusual positions, kept as reported.
- **A principal with a rate but no fair value and no cost**: 2,597 rows, $22.9B of principal, mostly
  at CIK 1572694, 1920145 and 1772704. These look like unfunded commitments or facts split across
  XBRL contexts. They do not enter the index (no fair value), but they are not three-way checkable.
- **Position history is not a reliable check.** A trailing number in the identifier ("Integrity
  Marketing Group LLC, Insurance 1" / "2") is the row's order in the filing, and it swaps between
  quarters, so the same identifier can be a $0.2M revolver one quarter and a $98M term loan the
  next. Comparing a value with "the same position's" other quarters therefore flags hundreds of
  false 1000x jumps and is used only as a secondary condition, never on its own.
- Rows with no principal (equity and warrants) have nothing to anchor fair value or cost on. The
  filer-quarter outlier check is their only protection, and it needs the quarter's total to be 20x
  the filer's norm.

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
| `rate_is_cash_part_of_base_plus_spread` | IR, PIK and spread, no PIC; IR + PIK = base + spread within 50bp while IR alone is not | PIC = old IR, IR = old IR + PIK (PIK carved out of the margin: "SOFR + 6.00% (3.25% PIK)" tagged IR 6.42%, PIK 3.25%) | 3,973 | +3,032 rows with IR more than 2pp below base + spread (I1 still passes: the PIK is counted in place of the cash coupon, not on top of it) |

**Fills**

| Rule | Condition | Change | Rows | Without it |
|---|---|---|---|---|
| `rate_from_base_plus_spread_plus_pik` | IR missing, spread + PIK, spread < PIK < base rate | PIK cannot be the all-in coupon, so it is paid on top: PIC = base + spread, IR = PIC + PIK ("Prime + 1.35%, Floor 9.85%, PIK 2.50%") | 674 | +445 rows with IR more than 2pp below base + spread (I1 still passes with IR = PIK alone) |
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
- **I2** (IR ≈ base + spread with floors, within 2pp) is checked only where I1 cannot be. Below about
  1%, gaps are benchmark noise: reset dates, 1M vs 3M term SOFR, and 10–26bp credit spread
  adjustments on the quarterly-average base.
- **Sign**: IR, PIC, PIK and spread ≥ 0, except "Prime − x%" loans, whose negative spread is real.

On the current output (496,502 rows with a usable rate, excluding 5,797 zero-rate rows):

| Check | Rows checked | Fails | Share of rows checked |
|---|---|---|---|
| No IR | 496,502 | 0 | 0.00% |
| Negative IR, PIC or PIK | 496,502 | 0 | 0.00% |
| Negative spread, not Prime-minus | 496,502 | 9 | 0.002% |
| I1 off by more than 1bp | 74,663 | 339 | 0.45% |
| I2 off by more than 2pp | 397,800 | 4,787 | 1.20% |
| **Passes every check** | 496,502 | 5,127 | **491,375 rows pass (98.97%)** |

24,039 rows (4.84% of rows with a usable rate) report IR only (fixed-rate or no spread), so neither identity applies.
The I2 failures are reported coupons left as filed: most sit above base + spread (an untagged PIK
or a spread field holding part of the margin), the rest below it (mostly foreign loans reported in
USD units and priced off SOFR). The I1 failures are mostly undrawn or partly drawn delayed-draw
loans whose PIC holds a 0.25–1% commitment fee.

**Where the two identities disagree.** Both can be checked on 50,005 rows (a PIC or PIK and a
spread). I1 is kept there and I2 is not enforced. They disagree (I1 holds, IR more than 2pp from
base + spread) on 5,339 of these rows (10.7%): 4,602 sit above base + spread, of which 4,118 equal
base + spread + PIK within 2pp, i.e. PIK paid on top of the margin, which is correct as filed; 737
sit below it.

### Known limitations

- **`spread, pic` rows with PIC below the spread** get IR = base + spread while PIC keeps
  its small value (about 1%, a commitment fee or a floor in the wrong field); the downstream
  `compute_final_interest_rates` then books the gap as PIK. These are the only rows where the
  estimate overrides I1; setting IR = PIC would give ~2% coupons on loans priced near 10%.
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
- Runs normalization, currency conversion, value-scale correction, the cost / fair value sign fix, the filer-quarter outlier
  rescale (with the drop as a fallback) and the subtotal-row drop
- Calls `resolve_rates` and returns the cleaned dataset

Shares are normally an equity signal, but HPS Corporate Lending Fund's 2026Q1 10-Q tagged shares on
63.7% of its positions (mostly term loans with real rate/FV data), so `mixed` rows are kept even when
shares are present — the same fix as in `preprocessor.py`.

### Entry Point

```bash
python scripts/run_ixbrl_pipeline.py
```

This calls

```python
run_pipeline(
    data_path="data/parsed/ixbrl_clean.csv",
    fx_path="data/reference/FX.csv",
    sofr_path="data/reference/SOFR_augmented.csv",
)
```

and writes the output to:

```text
data/cleaned/ixbrl_cleaned_out.csv
```

A full run on the current dataset takes about 3–4 minutes.

---

## Key Columns Added

- `*_normalized` — cleaned numeric versions of raw fields
- `*_scale_flag` — `unchanged`, `div100` / `div10000` (rates) or `div1000` / `div1000000` (values),
  `unresolved`, or `na`. For `InvestmentOwnedAtFairValue_normalized_scale_flag` /
  `InvestmentOwnedAtCost_normalized_scale_flag`, `unresolved` means the value is more than 3x the
  principal but not by a round 1e3 / 1e6 and was left as reported, `mul1000` means it was exactly
  1000x too small (`manual_mul1000` when set by `MANUAL_CORRECTIONS`), and `na` means there was no
  principal amount to check against
- `currency` — row currency after `determine_currency`; it also selects the loan's base rate
- `unit_mismatch_fixed` — principal read in a currency other than the one tagged: the fair-value
  currency, or the high-denomination currency its ratio to cost matches (see `convert_currencies`)
- `RateType` — base-rate member from the XBRL rate-type tag (e.g. `SecuredOvernightFinancingRateSofrMember`, `FixedRateMember`)
- `is_fixed` — `True` / `False` from `RateType`, `<NA>` when untagged
- `is_prime` — Prime-based loan (from `RateType` or the identifier text)
- `InvestmentOwnedBalancePrincipalAmount_normalized_scale_flag` — `mul1000` / `mul1000000` when the
  principal was too small, `filing_mul1000` when the whole filing's principals were, `div1000` /
  `div1000000` when it was too large relative to cost, else `unchanged` / `na`; suffix
  `|equity_units_capped` or `|units_as_principal` when a unit count was replaced by cost
- `InvestmentOwnedAtCost_normalized_scale_flag`, `InvestmentOwnedAtFairValue_normalized_scale_flag` —
  suffix `|sign_error` when a negative amount on a funded position was set to its absolute value
- `rate_config` / `check_2` — which rate fields were present on input (e.g. `spread, rate, pik`)
- `rate_source` — where the final IR came from: `reported`, `derived` (from other fields or a
  role repair), `estimated` (base + spread), `zero`, or `none`
- `check_1` — `resolved` if the row ends with a usable non-zero IR, else `unresolved`
- `principal_cost_check` — a label only; no value is changed. `consistent` (400,272 rows) when principal
  and cost are within 3x and fair value is not above both; `na` (94,567) when cost or principal is
  missing, zero or negative; `principal_high|fv_cost` (6,443 rows, $43.1B of principal) when principal
  exceeds 3x cost and fair value agrees with cost, with `|fv_principal` (186), `|fv_neither` (685) and
  `|fv_na` (45) for the other cases; `cost_high` (50) when cost exceeds 3x principal; `fv_high` (51)
  when fair value exceeds 3x both. The `principal_high` rows are mostly real: commitments carried at
  the drawn amount, mortgage-trust and CLO notes with a notional principal, volumetric production
  payments. They are flagged because income is booked as rate x principal: they add about 5bp to the
  index yield, but up to 2.3pp to the Real Estate sector (2025Q4, 26 mortgage-trust rows at CIK 1930679
  and 1975736) and 0.8pp to subordinated / mezzanine. Filter on this column, or use cost as the income
  base on these rows, in any sector or seniority analysis.
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
