# Valuation Dispersion Across BDCs

## Overview

When several BDCs lend to the same borrower in the same quarter, each marks its position
separately. The gap between their marks is a measure of valuation uncertainty, and it tends to
widen as a borrower runs into trouble.

`outputs/exploratory_analysis.py` holds notebook-friendly functions that measure this. They
display charts and tables inline and return the data behind them. Nothing is written to disk.

```python
import sys
sys.path.insert(0, "src/1_data_processing")
import paths; paths.add_code_dirs_to_path()
import exploratory_analysis as e

e.screen_valuation_dispersion(quarter="2025Q4")
e.visualize_borrower_valuation_dispersion("FIRST BRANDS")
```

Needs `seaborn` and `ipython` (both in `requirements.txt`).

---

## Data

The dispersion functions read `data/final_dataset/FINAL_DATASET.csv` through
`load_final_panel()`:

| Column used | Source column |
|---|---|
| `borrower_name` | `borrower_canonical`: the validated name, so one borrower has one spelling across BDCs |
| `FV` | `InvestmentOwnedAtFairValue_normalized` |
| `PAR` | `InvestmentOwnedBalancePrincipalAmount_normalized` |
| `cost` (where used) | `InvestmentOwnedAtCost_normalized` |

Rows with no borrower name or no fair value are dropped. All amounts are in USD after the
cleaning step's currency conversion and scale fixes.

A mark is fair value divided by principal (or by cost, where the function offers the choice).

---

## Functions

| Function | What it does |
|---|---|
| `screen_valuation_dispersion(min_bdcs=4, min_total_amount=30e6, min_spread_pct=15, quarter=None, denominator="cost", top_n=20)` | Ranks borrower-quarters by the gap between the highest and lowest BDC mark. Each BDC's positions in the borrower are summed first, so a BDC's first-lien loan and equity stake count as one mark. |
| `visualize_borrower_valuation_dispersion(borrower_query, denominator="principal")` | For one borrower (substring match on the name): each BDC's mark by quarter, and a box plot of the cross-BDC spread |
| `analyze_borrower_universe()` | Distribution of lenders per borrower and of loan sizes, using each borrower's peak quarter. Returns the 90th-percentile lender count and principal, as screening thresholds. |
| `analyze_dispersion_statistics()` | For borrower-quarters with 2+ BDCs: distribution of the mark spread, spread by distress tier (mean mark < 60, 60-80, 80-95, > 95), and each of the 10 largest BDCs' average deviation from the syndicate mean |
| `analyze_macro_dispersion_trend()` | Median and 90th-percentile mark spread per quarter |
| `visualize_single_filing(cik, accession)` | Shows one filing's extracted fields from `data/parsed/ixbrl_clean.csv`. This checks the parsing step and is not part of the dispersion analysis. |

---

## Results on the current data

- **Borrower universe:** 14,832 borrowers with positive principal. Most have a single BDC
  lender (median 1). The 90th percentile is 5 BDCs, and the most widely held has 41. Median peak
  principal is $17.0M; the 90th percentile is $123.8M.
- **Syndicated borrower-quarters** (2+ BDCs, all marks between 0 and 2x principal): 40,977. In
  23.3% of them the marks differ by 5 points or more; in 18.0% by 10 or more.
- **Over time:** the median spread stays below 1 point in every quarter. The 90th percentile
  rises from 16.4 points in 2023Q1 to 50-76 points from 2024Q4 on.
- **Example:** First Brands' marks across BDCs fall from about 94% of principal in 2025Q2 to
  about 36% in 2025Q3, matching its September 2025 bankruptcy.

---

## Limitations

- **`analyze_dispersion_statistics` and `analyze_macro_dispersion_trend` compare single
  positions.** A borrower's first-lien loan and its equity count as one syndicate, so part of
  the spread reflects different instruments rather than disagreement. This likely inflates the
  90th percentile. `screen_valuation_dispersion` avoids it by summing each BDC's positions
  first, which is the better measure of disagreement.
- Positions are matched across BDCs on `borrower_canonical`, so borrowers with no validated
  name (397 rows) are left out.
