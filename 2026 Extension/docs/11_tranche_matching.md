# Tranche Matching and Manager Families

## Overview

The risk signals in the Risk Dimensions doc are defined per loan (tranche), not per borrower,
and count each manager family as one holder. Two shared modules in `src/1_data_processing/`
supply that:

| Module | Gives |
|---|---|
| `manager_family.py` | `family_map()` / `add_family(df)`: CIK -> manager family |
| `tranches.py` | `build_tranche_panel(df)`: funded debt tranches, chains over time (`chain_id`), matches across BDCs (`xb_id`, `xb_tier`), PIK at origination (P3) |
| `src/4_val_dispersion/val_dispersion.py` | `tranche_dispersion(tr)`: V1, V2, V4 on those tranches |

They are used by the warning signs and the borrower-quarter panel in the risk monitor
(`12_risk_monitor.md`, `13_distress_event_study.md`: crowding, disagreement on value, PIK) and
by crowding (`7_concentration_crowding.md`) and the early-warning model
(`14_early_warning_model.md`). The return index (`4_index_construction.md`) uses its own
borrower + seniority positions.

Tests: `python tests/test_tranches.py`.

---

## Manager families

Funds run by one manager share a valuation process, so they count as one holder.
`manager_family.py` maps each CIK to a family by regex on all of its names in
`BDC_filings_2001_present.csv`, plus `CIK_FAMILIES` for BDCs whose filing names hide the
manager (OCSL, BBDC, TCPC). A BDC matching no rule is its own family.

**Affiliates merged 2026-10-24.** Pairwise portfolio overlap (`12_risk_monitor.md`, C5) showed
family pairs sharing 30-100% of their books, and reading the adviser named in every unmapped
BDC's own 10-K/10-Q found more. Six merges, each confirmed from the BDC's filing (adviser and
valuation designee):

| BDC | Adviser in its filing | Family |
|---|---|---|
| Senior Credit Investments, LLC | Jefferies Credit Management | Jefferies (with Jefferies Credit Partners BDC) |
| LGAM Private Credit LLC | MS Capital Partners Adviser | Morgan Stanley |
| West Bay BDC LLC | Goldman Sachs Asset Management | Goldman Sachs |
| APS BDC, LLC | OHA Private Credit Advisors II | OHA (T. Rowe Price) |
| CCS IX Portfolio Holdings, LLC | Crescent Capital Group | Crescent |
| Commonwealth Credit Partners BDC I | Commonwealth Credit Advisors, a Comvest affiliate | Comvest (with AMG Comvest Senior Lending Fund) |

Not merged: **EPT 16 LLC** shares 56% of its book with Trinity Capital, whose adviser
sub-advises it and sources its loans, but Eagle Point Credit Management is its adviser and
valuation designee, so its marks are a separate opinion. BDCs with no filing in `cache/` could
not be checked this way; their names show no affiliation.

The 196 BDCs in FINAL_DATASET form 96 families (102 before the merge). The largest: Golub (8
BDCs), Blue Owl, Goldman Sachs and Morgan Stanley (7 each), New Mountain and TCW (6 each). The map is shared with the
data-center charts (`data_center_analysis.manager()`).

---

## Tranche matching

Tranches are told apart by their terms, never by `investment_identifier` text:

| Field | Source | Coverage (debt rows) |
|---|---|---|
| Lien | `instrument_seniority`, grouped | 100% (24% `OTHER_UNKNOWN`) |
| Spread | `InvestmentBasisSpreadVariableRate_normalized` | 92% |
| Maturity | `InvestmentMaturityDate` | 99.4% (`maturity_source`: 38% tagged in XBRL, the rest read from the HTML schedule) |

76 of 190 BDCs, including the largest, never tag maturity in XBRL but show it in their HTML
schedule of investments. `maturity_html.py` reads it from there (see `6_final_dataset.md`,
step 6); before that, maturity covered 38% of debt rows and most matches rested on spread
alone. The matching still works without maturity and records how confident each match is.
Hiding maturity for the BDCs that tag it showed what spread-only matching costs: precision
0.90 when collapsing a BDC's pieces, 0.94 for chain links, 0.94 (recall 0.93) across BDCs.

**Lien groups.** `SENIOR_SECURED_1L` and `UNITRANCHE` -> 1L; then 2L, SUB, UNSEC, CLO. Two
known groups never match. `SENIOR_SECURED_UNSPEC` and `OTHER_UNKNOWN` are ANY: they match any
group, but only when maturity or spread confirms it. One BDC often labels a loan 1L while
another leaves it unknown; matching on the exact label split First Brands' first-lien
syndicate into three.

### Steps

1. **Debt rows** (`standardize_debt_rows`): rows with par > 0. Two flags:
   - `funded`: FV > 0, cost > 0, cost >= 5% of par. Undrawn commitments are dropped.
   - `price_ok`: funded and cost >= 80% of par. Below that, par is not what was lent: a partly
     drawn revolver or delayed-draw line reports the whole commitment as par, and a cost
     write-down or a deep-discount purchase also puts cost far below par. These rows (about 4%)
     stay in the chains but are left out of prices. Before this rule, the top dispersion hits
     were partly drawn lines (e.g. a $60M Spotless Brands line with $8M drawn read as a price
     of 12).
   - `is_dip`: DIP loans, flagged from identifier text (classifying text, not matching on it).
     They match only other DIP loans.
2. **Collapse** (`collapse_tranches`): one row per BDC x quarter x borrower x lien x maturity x
   spread. A BDC's pieces with identical terms become one tranche. Rates are par-weighted.
   `price` = FV / par over the `price_ok` rows.
3. **Chains** (`link_tranche_chains`): each tranche links to a lien-compatible tranche of the
   same BDC and borrower in the previous observed quarter (gap <= 2 quarters):

   | Rule | Match | Links, 2023Q2-2026Q2 |
   |---|---|---|
   | 1 | Maturity within 31 days, spread within 25 bp (or missing) | 206,328 |
   | 2 | Maturity within 31 days, spread changed (repricing) | 13,207 |
   | 3 | Maturity missing, spread within 25 bp; or neither side reports either | 1,028 |
   | 4 | What is left: the only remaining pair, or pairs by closest par within 2x | 8,505 |

   Within a rule the closest pair links first; a missing term counts as half a tolerance away,
   and par size breaks ties, so a small new rescue loan starts its own chain instead of taking
   over the term loan's. 282,749 tranche-quarters form 53,681 chains (with HTML maturities,
   2026-10-24; before, 273,769 and 51,231, because loans sharing a spread were merged).
   Across BDCs, 99.2% of tranches now match at tier 1 (was 3% for BDCs without tagged maturity).
4. **Across BDCs** (`match_tranches_across_bdcs`): the same quarter's tranches of one borrower
   are clustered, per known lien group:

   | Tier | Match |
   |---|---|
   | 1 | Maturity within 31 days (spread within 25 bp where both report it) |
   | 2 | Maturity missing on a side; spread within 25 bp |
   | 3 | Neither term reported; matched only if every BDC holds a single tranche of the borrower x lien |
   | NaN | Could not be placed (ambiguous); its own cluster |

   ANY-lien tranches join a known-lien cluster only when maturity (tier 1) or spread (tier 2)
   singles out one cluster. A cluster never takes a second tranche from the same BDC through
   tiers 2-3. Rows with no validated borrower are never matched.
5. **PIK at origination, P3** (`add_pik_origination`): for each chain, the cross-BDC group in
   its first quarter shows when the loan first appeared in any BDC and whether it had PIK
   then. Loans first seen in 2023Q1 are censored (P3 null, 35% of chains).

### Example: Computer Data Source (Main Street, MSIF)

Main Street tags no maturity and labels the loans `SENIOR_SECURED_UNSPEC`. In 2026Q2 the
term loan reprices from SOFR + 8.25% to + 10.25% while a $0.8M rescue loan with no spread is
held alongside it. The term loan keeps its chain (rule 4), the rescue loan has its own chain
from 2026Q1 with P3 = True, and the two are never merged across BDCs.

---

## Limitations

- **Maturity is missing for 76 BDCs**, so most matches rest on lien + spread (tier 2). Two
  loans of one borrower with the same lien and spread (e.g. an original term loan and an
  incremental) can be merged when neither side reports maturity. Restricting to tier 1 is the
  robustness check, at the cost of sample size.
- **Rule-4 chain links** (6% of links) carry an amendment that moved both terms, or a wrong
  link. PIK toggles are about 9% of rule-2 and rule-4 links against about 1% of rule-1 and
  rule-3 links, which fits amendments but should be checked with a rule 1-3 subsample.
- **Cost write-downs are not priced** (`price_ok` needs cost >= 80% of par), so a loan whose
  cost was written down drops out of dispersion. A cost write-down is a T2 outcome in the Risk
  Dimensions doc and should be measured from the raw rows, not from tranche prices.
- **Currency.** About 0.4% of debt rows have FV / par above 1.2, some from par left in local
  currency (e.g. Crescent's AUD loan to Vermont AUS). Family prices above 1.2 are dropped
  (`MAX_PRICE`); the cause sits in the cleaning step.
- **P3 looks one hop back.** A loan whose first holders left before the current ones arrived
  is dated from the earliest holder still present.
