# PIK Risk Metrics

## Overview

PIK (payment-in-kind) interest is paid by adding to the loan's principal instead of in cash.
PIK written into a deal at origination is a structuring choice. PIK switched on later is often
a sign of borrower stress, because the lender accepts a promise to pay later instead of cash
now.

`src/5_pik/pik_metrics.py` separates these cases and tracks them over time.
`scripts/run_pik_risk_metrics.py` runs the whole set:

```
python scripts/run_pik_risk_metrics.py
python tests/test_pik_metrics.py        # unit tests
```

It reads `data/final_dataset/FINAL_DATASET.csv` and prepares it in four steps:

1. `load_and_prepare_investment_data()`
2. `prepare_subindex_input()`: merges near-duplicate sector labels and applies the rate ceiling
   (see `5_subindex_construction.md`).
3. `compute_final_interest_rates()`: gives `PIC_Final`, `PIK_Final` and `IR_Final`.
4. `compute_position_level_flows()` and `compute_market_weights()`: give each position's prior
   quarter, income and market weight.

Outputs:

| File | Contents |
|---|---|
| `outputs/pik/pik_provenance_summary.csv` | Per quarter: positions, FV and par by provenance (explicit / inferred / zero PIK) |
| `outputs/pik/pik_provenance_by_cik.csv` | Per BDC: how often its PIK is explicit vs. inferred |
| `outputs/pik/pik_transitions_summary.csv` | Per quarter: counts and FV by transition state, amendment rates |
| `outputs/pik/pik_non_cash_metrics.csv` | Per quarter: the three non-cash shares, weighted and unweighted |
| `outputs/pik/pik_sector_time_series.csv` | The same metrics per sector and quarter |
| `outputs/pik/pik_amendment_event_study.csv` | Markdown rates around a distress amendment, amended vs. never-amended loans |
| `outputs/pik_plots/*.png` | Eight charts of the above |

---

## 1. Provenance: was the PIK rate in the filing?

**`classify_pik_provenance(df)`** labels each position by where its PIK rate came from. It uses
`check_2`, which lists the rate fields the filing tagged, e.g. `spread, rate, pik`.

| Label | Rule |
|---|---|
| `explicit_pik` | PIK > 0, and the filing tagged both the cash and PIK rates (`check_2` is `rate, pik, pic` or `spread, rate, pik, pic`) |
| `inferred_pik` | PIK > 0 otherwise: PIK tagged but cash rate inferred, or PIK derived by the cleaning step |
| `zero_pik` | No PIK |

Without a `check_2` column the function falls back to "any positive tagged PIK is explicit".
`FINAL_DATASET.csv` has carried `check_2` since 2026-10-22, so the fallback is not used.

Under this definition explicit PIK is rare: 136 positions in 2026Q2, against 4,734 inferred.
Most loans with a tagged PIK rate count as inferred because their cash rate was not tagged.
If "explicit" should mean "the PIK rate itself was tagged", change the two `check_2` lists in
`classify_pik_provenance`.

## 2. Transitions: contractual vs. distress PIK

**`classify_loan_pik_transitions(df, min_amendment_bps=0.015)`** follows each position from
quarter to quarter. A position is a BDC's holding in one borrower at one seniority
(`cik`, `position_key`), as built by `compute_position_level_flows()` (see
`4_index_construction.md`, "Position Matching"). Several tranches of one borrower and seniority
are combined, with their PIK rate weighted by principal.

| State | Rule |
|---|---|
| `contractual_pik` | PIK > 0 in the position's first quarter |
| `distress_amendment_pik` | A seasoned position whose PIK goes from 0 to > 0, or rises by at least 150 bp |
| `seasoned_pik_continuation` | Ongoing PIK on a seasoned position, no jump |
| `cured_pik` | PIK goes from > 0 back to 0 |
| `stable_cash` | No PIK |

**`aggregate_pik_transitions()`** gives quarterly counts and FV, and the amendment rates
`pct_amendment_count` (share of seasoned positions) and `pct_amendment_fv` (share of FV).

On the current data, 1.5% to 2.3% of seasoned positions turn to distress PIK each quarter
(168 to 426 positions), with no break in 2025Q4.

## 3. Non-cash shares

**`compute_non_cash_metrics()`** gives three position-level measures:

| Metric | Formula | Measures |
|---|---|---|
| A. Rate share | `PIK_Final / IR_Final` | Contracted non-cash part of the coupon |
| B. Income share | `pik_income / (cash_income + pik_income)` | Non-cash part of interest accrued in the quarter |
| C. Capitalized burden | `(PAR_t - PAR_incept) / PAR_t` | Principal growth since the loan was first seen |

**`aggregate_non_cash_metrics()`** gives market-value-weighted and simple means per quarter.
The weighted income share stays between 4.6% and 5.8% from 2023Q2 to 2026Q2 (4.85% in 2026Q2).
2023Q1 has no prior quarter, so its shares are zero. The rise from 1.3% reported before
2026-10-22 came from identifier matching, which missed most continuing positions early on.

## 4. Sector sweep

**`compute_sector_time_series_pik_metrics()`** reruns steps 1 to 3 within each sector.
It feeds `pik_sector_time_series.csv` and the sector amendment chart.

## 5. Event study: do amendments predict markdowns?

**`compute_amendment_event_study(window=(-2, 1))`** takes loans at their first distress
amendment (quarter t) and compares them with loans that never amend.

- **Comparison loans:** from the same sector and event quarter, reweighted to the amended loans'
  sector-by-quarter mix.
- **Balanced panel:** both groups must be observed in every quarter of the window.
- **Outcomes:** the share marked below 80% of cost, and the share whose FV falls more than 20%
  in one quarter.

On the current data there are 1,941 amended loans and 116,806 comparison loans:

| Quarter relative to amendment | Amended: FV/Cost < 0.8 | Never amended: FV/Cost < 0.8 | Amended: FV drop > 20% | Never amended: FV drop > 20% |
|---|---|---|---|---|
| t-2 | 7.8% | 2.2% | 3.1% | 2.8% |
| t-1 | 11.2% | 2.4% | 5.6% | 3.5% |
| t | 11.6% | 2.9% | 10.2% | 4.1% |
| t+1 | 13.4% | 3.4% | 5.7% | 4.8% |

Amended loans are already marked lower two quarters before the amendment (7.8% vs. 2.2% below
80% of cost). In the amendment quarter, their share with a drop of more than 20% is about 2.5
times the comparison group's (10.2% vs. 4.1%).

---|---|---|---|---|
| t-2 | 7.0% | 2.7% | 5.5% | 4.7% |
| t-1 | 9.7% | 2.9% | 6.9% | 5.3% |
| t | 9.5% | 3.3% | 11.8% | 5.7% |
| t+1 | 11.7% | 3.8% | 7.3% | 6.9% |

Amended loans are already marked lower two quarters before the amendment (7.0% vs. 2.7% below
80% of cost). In the amendment quarter, their share with a drop of more than 20% is about twice
the comparison group's (11.8% vs. 5.7%).

---

## Charts (`outputs/pik_plots/`)

| File | Shows |
|---|---|
| `pik_provenance_positions.png` | Share of positions by provenance; explicit vs. inferred among PIK positions |
| `pik_provenance_dollar_volume.png` | FV and par by provenance ($bn) |
| `bdc_idiosyncratic_tagging.png` | BDCs with the most inferred PIK, by fund name (`CIK_NAME_MAP`) |
| `distress_amendments_vs_markdowns.png` | Amendment volume against the share of FV marked below 80% / 90% of cost |
| `sector_distress_amendments.png` | Amendment rate by sector |
| `pik_distress_vs_contractual_share.png` | PIK payments by state, and the distress share of PIK |
| `pik_non_cash_trends.png` | The three non-cash shares over time |
| `amendment_event_study.png` | The event study above, with ±1.96 SE bands for amended loans |

---

## Limitations

- **Tranches are combined.** A PIK amendment on one tranche of a borrower shows up as a smaller
  rise in the position's principal-weighted PIK rate, and may fall under the 150 bp threshold.
- **A seniority relabel breaks the link.** When a BDC relabels a position's seniority between
  quarters, the position starts again, and its PIK counts as contractual, not as an amendment.
- **The first quarter (2023Q1) has no seasoned positions** by construction, so its amendment rate
  is empty.
- **The saved outputs from GitHub could not be reproduced.** The PIK files committed there came
  from an unpushed version of the code. The current files were regenerated from this code and
  `FINAL_DATASET.csv` on 2026-10-22, with borrower matching. Rerunning gives the same numbers; only the row order of some
  tables can change.
