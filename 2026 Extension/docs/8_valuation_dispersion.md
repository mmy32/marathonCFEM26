# Valuation Dispersion Across BDCs

## Overview

When several unrelated lenders hold the same loan, each marks it separately. The gap between
their marks measures valuation uncertainty, and the lowest mark may carry the newest
information. This follows the Risk Dimensions doc, section 3:

- **Same loan:** tranches are matched across BDCs on lien, spread and maturity, never on
  identifier text (`11_tranche_matching.md`). Each match carries a confidence tier.
- **Unrelated holders:** each manager family is one holder; its price is its funded fair value
  over par summed across its BDCs. Main Street and MSIF, for example, are one opinion.
- **Price = fair value / par**, not / cost (two BDCs can pay different prices for one loan).
- **Excluded:** undrawn commitments, partly drawn lines and rows whose cost is under 80% of par
  (par is not what was lent), family prices above 1.2 (data errors), DIP loans matched only to
  DIP loans.

Metrics, per cross-BDC tranche and quarter (`tranches.tranche_dispersion`):

| Code | Column | Definition |
|---|---|---|
| V4 | `V4_consensus` | Median family price (the borrower-level version of the price baseline) |
| V1 | `V1_std`, `V1_iqr` | Std (ddof = 1) and IQR of family prices; `V1_range` = max - min for reference |
| V2 | `V2_low_mark` | Lowest family price minus the median (<= 0) |
| | `V1_pct`, `V2_pct`, `V4_pct` | Within-quarter percentiles (V2: low = most pessimistic) |
| | `n_families`, `n_bdcs`, `tier`, `total_par` | Holder counts, worst match tier, priced par |

V3 (lender optimism) is shown descriptively in `analyze_dispersion_statistics`; its
persistence test is not built yet.

```
python scripts/run_dispersion_signals.py    # writes outputs/risk_signals/dispersion_tranches.csv
                                            #        and tranche_match_coverage.csv
```

Notebook views in `outputs/exploratory_analysis.py` (display charts, return the data):

```python
import sys; sys.path.insert(0, "outputs")
import exploratory_analysis as e

e.screen_valuation_dispersion(quarter="2025Q4")
e.visualize_borrower_valuation_dispersion("FIRST BRANDS")
```

| Function | What it does |
|---|---|
| `compute_tranche_dispersion(min_families=2, max_tier=2)` | The V1/V2/V4 table above |
| `screen_valuation_dispersion(min_families=2, min_total_par=0, min_range_pct=15, quarter=None, max_tier=2, top_n=20)` | Tranches with the widest range of family prices |
| `visualize_borrower_valuation_dispersion(borrower_query, lien_group=None, max_tier=3)` | One borrower: each family's price per quarter on its largest matched tranche |
| `analyze_borrower_universe()` | Families per borrower and funded principal, at each borrower's peak quarter |
| `analyze_dispersion_statistics(max_tier=2)` | Range distribution; dispersion by consensus-price decile; lender optimism by family |
| `analyze_macro_dispersion_trend(max_tier=2)` | Median and 90th-percentile range per quarter |
| `visualize_single_filing(cik, accession)` | One filing's extracted fields (parser QA, not dispersion) |

`max_tier`: 1 = maturity matched for every holder (strict, small); 2 adds spread matches where
maturity is missing (default); 3 adds borrower x lien matches with no terms.

Needs `seaborn` and `ipython`. The tranche panel takes about a minute to build and is cached
per session.

---

## Results on the current data

- **Coverage:** about half of priced par sits in tranches held by 2+ families (50.8% in
  2026Q2 at tier <= 2; 5.4% at tier 1). Most borrowers have one family lender (median 1,
  90th percentile 3, maximum 24).
- **Matched tranche-quarters** (2+ families, tier <= 2): 25,795. Family prices differ by
  5 points or more in 7.1%, by 10 or more in 3.2%.
- **Over time:** the median range stays between 0.45 and 1.14 points; the 90th percentile
  falls from 5.8 points in 2023Q1 to 2.9-4.0 points in 2025-2026.
- **Dispersion and price are linked:** median std of family prices is 4.5 points in the lowest
  consensus decile (consensus 84) and under 1 point from the third decile up. V1 and V2 are
  only informative after V4 is accounted for.
- **Example:** First Brands' first-lien loan, 10 families in 2025Q2 at 91.2-94.8, fell to a
  consensus of 36.4 in 2025Q3 (7 families, 35.2-45.5), around its September 2025 bankruptcy.

### What changed (2026-10-23)

The earlier version compared CIKs at the borrower level, mixed tranches (and first lien with
equity), included unfunded commitments, used fair value / cost in the screen, and sorted loans
into fixed price buckets (< 60, 60-80, 80-95, > 95). Its 90th-percentile spread of 50-76 points
came largely from those mixes. The old file is in
`_archive/outputs_before_2026-10-23/exploratory_analysis.py`.

---

## Limitations

- Matches without maturity (tier 2) can merge two loans of one borrower that share lien and
  spread; tier 1 is the robustness check. See `11_tranche_matching.md`.
- A loan with a written-down cost (cost < 80% of par) is not priced, so it drops out of the
  dispersion sample in exactly the quarters it is distressed. Cost write-downs belong in the
  T2 outcome instead.
- Some currency errors remain in the cleaned data (par in local currency, FV in USD); family
  prices above 1.2 are dropped.
