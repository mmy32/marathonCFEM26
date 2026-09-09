# Sub-Index Construction Toolkit

## Overview

This module generalizes the ad hoc "Technology sector" / "Senior Loans" cells that used to live
directly in the pipeline notebook into a reusable mechanism: build a named, market-value-weighted
sub-index along any categorical column (today: `sector` or `instrument_seniority`), by filtering to
a bucket and re-running the same weighting/decomposition/index functions `README_4.MD`'s
whole-market engine already provides.

**Scope: full treatment, minus benchmarking.** Every sub-index gets market weights (renormalized to
sum to 1 *within* the bucket), a return decomposition, and a full flow-based index level. CDLI /
CDLI-S-style benchmark comparison and tracking error are intentionally out of scope here — no
benchmark CSVs (`cdli.csv`, `cdli-s.csv`, `SOFR.csv`, `FX.csv`) exist anywhere in this repo yet.

**What this module adds beyond the whole-market engine**: explicit, auditable suppression of
sub-index-quarters that are too thin to be reliable, rather than silently reporting a return built
on a handful of positions. A sub-index is not automatically "the same math, just filtered" — some
buckets (notably `instrument_seniority` values like `CLO_STRUCTURED_EQUITY` or `UNITRANCHE`) are
thin enough, in enough quarters, that presenting every quarter as equally reliable would be
misleading.

---

## Dependencies

Same as `README_4.MD` — `polars`, `pandas`, `numpy`, `matplotlib`. No new dependency is introduced.

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
478,634-row loaded panel: **0 rows flagged, for all three rate columns.** The clip stays as an
always-applied defensive backstop: the report prints `n_flagged=0` today, and would immediately
surface a real problem if that upstream guarantee ever stopped holding (e.g. a future change to
`ixbrl_utils.py`, or a re-run against a differently-processed input file).

The sector-label merge is not a no-op: **97,384 rows** (97,354 `"Other / Unknown / Unresolved"` +
30 `"Other / Unknown"`) were combined into `"Other / Unclassified"` on the current data.

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
**unconditionally** — a bucket can look adequately populated in its first quarter while still being
pure duplicate-tie artifact (see "Known Limitation, Carried Forward" below), so thresholds alone
would not reliably catch it.

A quarter failing any of the four checks is `suppressed`, with a comma-joined `suppression_reason`
(e.g. `"min_ciks,first_quarter_artifact"`). Suppression never drops a row: `suppress_return_series()`
preserves the actual computed value as `{col}_raw`, zeroes the public return column, and — for the
index frame — recomputes `IndexLevel` from the suppressed-adjusted return, so a suppressed quarter
holds the index level **flat** instead of compounding a number known to be unreliable.

### What suppression actually catches, measured on the real 476,403-row flowed panel

- **Sector never trips the three data-sufficiency thresholds.** Every one of the 11 merged sectors
  clears all three floors in every quarter except 2023Q1 (where the whole panel has only 285
  `is_valid` positions total — see below). Across a full sector sweep, all 11 suppressed rows are
  in 2023Q1.
- **Seniority is where it's load-bearing.** `CLO_STRUCTURED_EQUITY` is suppressed in **all 14
  quarters** (as low as 0 valid positions in several quarters, peaking at 11 positions / 1 CIK /
  $23M in 2026Q2 — never close to any floor). `UNITRANCHE`, built as its own single-value bucket,
  is suppressed in 8 of 14 quarters (2023Q1-Q4, 2025Q1, 2025Q3-2026Q1) and clear in the other 6
  (2024Q1-Q3, 2025Q2, 2026Q2) — a real, order-dependent mix of `min_positions`/`min_ciks`/`min_aum`
  failures across the sample, not a one-time inception issue.
- **The custom 5-value "Senior Loans" grouping** (`SENIOR_SECURED_1L`, `SENIOR_SECURED_2L`,
  `SENIOR_UNSECURED`, `SENIOR_SECURED_UNSPEC`, `UNITRANCHE` combined) is thick enough that only its
  2023Q1 inception quarter is suppressed — pooling several seniority values into one bucket is
  itself a way to get past thin-bucket suppression, when that grouping is analytically meaningful.
- A full seniority sweep (one bucket per raw value, `CLO_STRUCTURED_EQUITY` through
  `SENIOR_SECURED_1L`) produces 30 suppressed bucket-quarters in total, across 14 quarters and 8
  buckets.

**2023Q1 is thin panel-wide, not just for narrow buckets**: the whole market has only 285 `is_valid`
positions in 2023Q1 versus 12,000-19,000+ in every later quarter — a ~40-60x drop. This is the same
duplicate-tie/inception-quarter mechanism documented in `README_4.MD`'s "Known Limitations — Open",
confirmed directly on this dataset.

---

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

## Known Limitation, Carried Forward (not fixed here)

`compute_position_level_flows()` lags each position within `(cik, investment_identifier)` groups;
~2.2% of rows panel-wide share a duplicate `(cik, investment_identifier, cal_q)` key (distinct
`context_id`), making `FV_prev` order-dependent for those rows (full detail in `README_4.MD`,
"Known Limitations — Open"). **This module does not fix that upstream issue.**

It is **proportionally worse here than for the whole panel**: a bucket with a handful of valid
positions in a quarter (e.g. `CLO_STRUCTURED_EQUITY` at 0-11 positions, or `UNITRANCHE` at
13-96) is far more sensitive to one or two mis-attributed rows than the ~476K-row panel is to the
same 2.2%. The suppression logic above does not detect or correct this — it only catches buckets
that are thin in the aggregate sense (too few positions/issuers/AUM). A bucket could in principle
pass every suppression threshold while still having its within-threshold return meaningfully
distorted by this duplicate-tie effect. The documented, not-yet-applied fix (adding `context_id` or
`accession` to `compute_position_level_flows()`'s `group_cols`) would address both the whole-market
and every sub-index at once, but changes index construction output and was out of scope for this
pass.

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
