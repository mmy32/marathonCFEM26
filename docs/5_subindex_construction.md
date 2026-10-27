# Sub-Index Construction Toolkit

## Overview

This module generalizes the ad hoc "Technology sector" / "Senior Loans" cells that used to live
directly in the pipeline notebook into a reusable mechanism: build a named, market-value-weighted
sub-index along any categorical column (today: `sector` or `instrument_seniority`), by filtering to
a bucket and re-running the same weighting/decomposition/index functions `4_index_construction.md`'s
whole-market engine already provides.

**Scope: full treatment, minus benchmarking.** Every sub-index gets market weights (renormalized to
sum to 1 *within* the bucket), a return decomposition, and a full flow-based index level. CDLI /
CDLI-S-style benchmark comparison and tracking error are intentionally out of scope here. The
benchmark files (`data/reference/cdli.csv`, `cdli-s.csv`) are used only by the whole-market index
(`4_index_construction.md`).

**What this module adds beyond the whole-market engine**: explicit, auditable suppression of
sub-index-quarters that are too thin to be reliable, rather than silently reporting a return built
on a handful of positions. A sub-index is not automatically "the same math, just filtered" — some
buckets (notably `instrument_seniority` values like `CLO_STRUCTURED_EQUITY` or `UNITRANCHE`) are
thin enough, in enough quarters, that presenting every quarter as equally reliable would be
misleading.

---

## Dependencies

Same as `4_index_construction.md` — `polars`, `pandas`, `numpy`, `matplotlib`. No new dependency is introduced.

---

## Data-Quality Preparation

**Function**: `prepare_subindex_input()`

Runs once, right after `load_and_prepare_investment_data()` and **before**
`compute_final_interest_rates()` (so a nulled-out bad rate can still be derived from the other two
rate fields by that function's existing fallback logic):

- Nulls (never drops) any of `rate_cash` / `interest_rate` / `rate_pik` whose absolute value exceeds
  `DEFAULT_RATE_CEILING` (1.0, i.e. 100%).
- Merges the two near-duplicate sector labels `"Other / Unknown"` and
  `"Other / Unknown / Unresolved"` into one `"Other / Unclassified"` bucket.
- Returns `(df_clean, report)` — `report` is a plain dict of what was actually affected, meant to
  be printed, not silent.

**On the current dataset, the rate-ceiling clip is a no-op — by design, and this is worth
understanding rather than treating as dead code.** `load_and_prepare_investment_data()` reads the
`*_normalized` rate columns, which `ixbrl_utils.normalize_interest_columns()` (via
`RATE_RANGE = (0.00, 0.50)`) already rescales into `[0, 0.5]` upstream. Measured directly on the
487,715-row panel loaded from `FINAL_DATASET.csv`: **0 rows flagged, for all three rate columns.** The clip stays as an
always-applied defensive backstop: the report prints `n_flagged=0` today, and would immediately
surface a real problem if that upstream guarantee ever stopped holding (e.g. a future change to
`ixbrl_utils.py`, or a re-run against a differently-processed input file).

The sector-label merge is not a no-op: **41,547 rows** (41,429 `"Other / Unknown / Unresolved"` +
118 `"Other / Unknown"`) were combined into `"Other / Unclassified"` on the current data. (Before the
label validation that produced `FINAL_DATASET.csv`, it was 97,384 rows.)

---

## Coverage and Suppression

**Functions**: `compute_subindex_coverage()`, `flag_first_quarter_artifact()`,
`apply_suppression()`, `suppress_return_series()`

For every quarter in a bucket, three independent data-sufficiency checks run against the same
`is_valid` population `w_mkt` and `n_investments` already use (not the raw filtered row count):

| threshold | default | rationale |
|---|---|---|
| `min_positions` | 30 | floor on raw position count |
| `min_distinct_ciks` | 5 | floor on distinct issuers — catches single-issuer concentration a position-count-only rule would miss |
| `min_aum_usd` | $250,000,000 | floor on prior-quarter fair value backing the bucket |

Independently of those three, every bucket's earliest quarter (`start_idx=1` quarter, by the same
`qsort` ordering used everywhere else in this pipeline) is flagged via `is_inception_quarter`,
**unconditionally**. A bucket's first quarter has no prior quarter to compute a return from. Under
the old identifier matching, it could still look populated because of duplicate-identifier
artifacts, so thresholds alone would not reliably catch it.

A quarter failing any of the four checks is `suppressed`, with a comma-joined `suppression_reason`
(e.g. `"min_ciks,first_quarter_artifact"`). Suppression never drops a row: `suppress_return_series()`
preserves the actual computed value as `{col}_raw`, zeroes the public return column, and — for the
index frame — recomputes `IndexLevel` from the suppressed-adjusted return, so a suppressed quarter
holds the index level **flat** instead of compounding a number known to be unreliable.

### What suppression actually catches, measured on `FINAL_DATASET.csv` (2026-10-22)

Positions are matched across quarters on borrower + seniority (see `4_index_construction.md`,
"Position Matching"). The 487,715 loaded rows become 265,833 BDC-position-quarters.

- **Only the first quarter is suppressed.** All 11 sectors and all 8 seniority values clear every
  floor in every quarter after 2023Q1. 2023Q1 is suppressed for every bucket: it is the panel's
  first quarter, so no position has a prior quarter and nothing is valid.
- **Seniority buckets are not thin.** Outside 2023Q1, `CLO_STRUCTURED_EQUITY` has 187-591 valid
  positions at 13-26 BDCs, `UNITRANCHE` 781-1,691 at 12-40, `SUBORDINATED_MEZZ` 141-292 at 30-50,
  and `SENIOR_UNSECURED` 64-101 at 21-31.
- **The custom 5-value "Senior Loans" grouping** is likewise suppressed only in 2023Q1.

**History.** Before 2026-10-22, positions were matched on `investment_identifier` text. Under
that matching, and with the unvalidated labels, `CLO_STRUCTURED_EQUITY` was suppressed in all 14
quarters and `UNITRANCHE` in 8. 2025Q4 was also thin panel-wide, with 7,225 matched positions,
because most BDCs reworded their identifiers in their FY2025 10-Ks. Borrower matching gives
2025Q4 20,228 matched positions, in line with 2025Q3 (19,335) and 2026Q1 (21,997).

## Orchestration

**Functions**: `build_subindex()`, `build_all_subindices_for_dimension()`

`build_subindex(df, dimension_col, bucket_name, bucket_values)` is the one generic primitive: it
covers both "one bucket per raw category value" (`bucket_values=["Technology"]`) and "one bucket =
a custom multi-value union" (`bucket_values=DEFAULT_SENIOR_LOANS_BUCKET`) — no per-dimension
branching. `df` must already be the flowed panel: the output of `compute_final_interest_rates()` →
`compute_position_level_flows()`.

`build_all_subindices_for_dimension(df, dimension_col)` auto-builds one bucket per distinct value —
this is how every sector (including the merged `"Other / Unclassified"` catch-all) and every
seniority value (including `"OTHER_UNKNOWN"`) gets its own reportable sub-index by default, with no
special-casing of catch-all buckets. Pass `bucket_map` explicitly to build custom groupings instead
(e.g. `{"Senior Loans": DEFAULT_SENIOR_LOANS_BUCKET}`).

Both return a dict — `build_subindex` returns `{"positions", "decomp", "index", "coverage", "meta"}`
for one bucket; `build_all_subindices_for_dimension` returns `{bucket_name: that dict}`. This
mirrors the notebook's existing per-bucket-variable mental model (`df_sector`, `df_loans`) rather
than forcing every bucket into one long-format frame up front, since `positions`/`decomp`/`index`/
`coverage` don't share one row shape.

**Cross-bucket helpers**: `stack_subindex_results(results, dimension_col, frame="index")` produces
an opt-in long-format frame (adding `dimension`/`bucket` columns) for faceted plotting or
comparison across buckets. `summarize_suppressed_quarters(results)` scans every bucket's coverage
frame and returns one row per suppressed `(bucket, quarter)` — the audit table to print right after
a full sweep.

### Typical Usage Order

```
load_and_prepare_investment_data
prepare_subindex_input
compute_final_interest_rates
compute_position_level_flows
build_subindex                          # one named bucket
  -- or --
build_all_subindices_for_dimension      # every value of a dimension
stack_subindex_results                  # optional, for faceted comparison
summarize_suppressed_quarters           # audit table
```

---

## Report-aligned sectors vs. MSCI (added 2026-10-24)

`subindex_construction.py` also builds one sub-index per `sector_report` value (Software and
Business Services split out, as in Financing the Economy 2026, Fig 18; see
`6_final_dataset.md`, step 7) and compares each with the MSCI Global Private Credit Closed-End
Fund Index (Fig 15: 9.5% in 2025, 0.9% in Q1 2026). Outputs in
`outputs/sector_subindices_report_plots/`: one figure per sector,
`summary_vs_bdc_market.csv` and `summary_vs_msci.csv` (FV weight at 2026Q2, 2025 and Q1 2026
returns, and the gap to MSCI).

It also draws one **return decomposition** per report sector, `decomp_<sector>.png`, in the same
format as `index_construction_plots/technology_sector_decomposition.png`
(`plot_sector_decompositions`, added 2026-10-26): the sector's positions re-weighted within the
sector, and each quarter's return split into price change, cash interest and PIK interest. One
panel per sector, so it is the simpler chart to present. Numbers in `decomposition_by_sector.csv`.
Software, 2026Q1: price -3.33%, cash interest 2.10%, PIK 0.09%, total -1.14% (the only negative
sector; every sector lost some value that quarter).

| Sector | FV weight | 2025 | Q1 2026 |
|---|---|---|---|
| Energy & Utilities | 1.7% | 7.9% | 1.9% |
| **Software** | 16.2% | 8.9% | **-1.1%** |
| Real Estate | 1.4% | 9.0% | 2.3% |
| Materials | 3.2% | 9.1% | 1.2% |
| Healthcare | 15.9% | 9.1% | 1.2% |
| Industrials (ex-business services) | 10.9% | 9.2% | 1.9% |
| Consumer | 11.6% | 9.2% | 1.8% |
| *MSCI private credit index* | | *9.5%* | *0.9%* |
| *BDC market* | 100% | *9.6%* | *1.0%* |
| Business Services | 10.3% | 9.6% | 1.3% |
| Financials & Insurance | 9.2% | 9.7% | 1.3% |
| Technology (ex-software) | 5.6% | 10.5% | 1.3% |
| Communication & Media | 3.2% | 10.9% | 0.6% |
| Transportation | 0.6% | 12.3% | 0.8% |

Software is the only sector with a negative Q1 2026 return, 2 points below MSCI: the AI-driven
software re-mark the report describes (p.27). Splitting it out matters, since the old Technology
bucket mixed it with IT services and hardware, which returned 1.3%. The sub-indices are
fair-value-weighted total returns on BDC positions, and MSCI is a fund-level index, so the
levels are comparable only roughly.

## Known Limitations

- **Thresholds catch thin buckets, not bad data.** A bucket that passes every floor can still
  carry a mismarked position. Suppression only checks there is enough data, not that it is right.
- **Two tranches of one borrower at the same seniority are one position** (see
  `4_index_construction.md`, "Position Matching"). When a BDC relabels a position's seniority
  between quarters, its link to the prior quarter breaks for that quarter.

---

## Design Notes

- Every function here calls back into `index_construction.py`'s `compute_market_weights`,
  `aggregate_return_decomposition`, and `compute_flow_based_market_index` unchanged — this module
  owns segmentation, data-quality prep, and suppression; `index_construction.py` stays the
  whole-market engine.
- Suppression never drops rows. A suppressed quarter is always present in every output frame, with
  `suppressed=True` and a `suppression_reason`, and its original (unreliable) computed value
  preserved in a `{col}_raw` column — auditable, not silently missing.
- `min_aum_usd` is a fixed dollar floor against a panel whose quarterly AUM has grown from ~$234B
  (2023Q1) to ~$515B (2026Q2). Fine for the current ~3.5-year sample; would need to become
  share-of-panel rather than a fixed dollar amount if this pipeline runs for many more years.
- `compute_subindex_coverage()` only covers quarters that appear at all (valid or not) in the
  filtered bucket. A bucket-quarter with literally zero rows of any kind is not synthesized into a
  suppressed row — not a scenario the current sector/seniority buckets hit (every bucket has at
  least some rows in all 14 panel quarters), but worth knowing before applying this to a sparser
  future dimension.
