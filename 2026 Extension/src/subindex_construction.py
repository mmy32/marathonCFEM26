"""
Sub-index construction utilities.

Builds a named market-value-weighted sub-index (e.g. one sector, one seniority
bucket, or a custom multi-value grouping like "Senior Loans") on top of the
whole-market engine in `index_construction.py`, by filtering to a bucket and
re-running the same weighting/decomposition/index functions on the subset —
generalizing the ad hoc `pl.col(...).is_in([...])` pattern that used to live
directly in the pipeline notebook.

Scope: full treatment (market weights, return decomposition, flow-based index
level) minus benchmarking. CDLI/CDLI-S-style comparison and tracking error are
intentionally out of scope here -- no benchmark CSVs exist in this repo yet.

Known, un-fixed caveat carried forward from `index_construction.py`
("Known Limitations -- Open" in docs/README_4.MD): `compute_position_level_flows()`
lags each position within (cik, investment_identifier) groups, but ~2.2% of
rows share a duplicate (cik, investment_identifier, cal_q) key (distinct
`context_id`), making `FV_prev` order-dependent for those rows. This is
PROPORTIONALLY WORSE for a narrow sub-index bucket than for the whole panel --
a bucket with a handful of positions in a quarter is far more sensitive to one
or two mis-attributed rows than the full ~490K-row panel is. This module does
not fix that upstream issue; it only adds suppression so a bucket-quarter that
is too thin to be reliable is flagged rather than silently reported.

Related, already-established convention this module extends rather than
reinvents: the panel's first quarter (2023Q1) has no position with a genuine
prior-quarter fair value at all, so its reported return is entirely an
artifact of the same duplicate-tie mechanism. Every stats-facing consumer in
`index_construction.py` already excludes it via a `start_idx` parameter; this
module applies the same convention to every sub-index via
`flag_first_quarter_artifact`, independent of (and in addition to) the
data-sufficiency checks below, because an inception-quarter artifact can look
adequately populated while still being pure artifact.
"""

import polars as pl

from index_construction import (
    safe_div,
    compute_market_weights,
    aggregate_return_decomposition,
    compute_flow_based_market_index,
)

# =====================
# Defaults
# =====================

DEFAULT_SECTOR_LABEL_MERGE_MAP = {
    "Other / Unknown / Unresolved": "Other / Unclassified",
    "Other / Unknown": "Other / Unclassified",
}

# Ported verbatim from the notebook's existing senior_loans_cols
DEFAULT_SENIOR_LOANS_BUCKET = [
    "SENIOR_SECURED_1L",
    "SENIOR_SECURED_2L",
    "SENIOR_UNSECURED",
    "SENIOR_SECURED_UNSPEC",
    "UNITRANCHE",
]

DEFAULT_RATE_CEILING = 1.0  # 100%
DEFAULT_MIN_POSITIONS = 30
DEFAULT_MIN_DISTINCT_CIKS = 5
DEFAULT_MIN_AUM_USD = 250_000_000.0
DEFAULT_START_IDX = 1


# =====================
# Data-quality preparation
# =====================

def prepare_subindex_input(
    df: pl.DataFrame,
    rate_cols: list[str] = ["rate_cash", "interest_rate", "rate_pik"],
    rate_ceiling: float = DEFAULT_RATE_CEILING,
    sector_col: str = "sector",
    sector_label_merge_map: dict[str, str] | None = None,
) -> tuple[pl.DataFrame, dict]:
    """
    Null out implausible rate values (abs > rate_ceiling, never dropping the
    row) and merge near-duplicate sector labels, before any sub-index is
    built. Must run before `compute_final_interest_rates()` so a nulled rate
    can still be derived from the other two rate fields by that function's
    existing fallback logic.

    NOTE: on the dataset this pipeline currently produces, the rate-ceiling
    clip is a no-op. `load_and_prepare_investment_data()` reads the
    `*_normalized` rate columns, which `ixbrl_utils.normalize_interest_columns()`
    (via `RATE_RANGE = (0.00, 0.50)`) already rescales into [0, 0.5] upstream
    -- so zero rows exceed even 1.0 today. It is kept as an always-applied
    defensive backstop, not dead code: the returned report should show
    `n_flagged=0` for every rate column today, and would immediately surface
    a real problem if that upstream guarantee ever stopped holding.

    Returns (df_clean, report) where report is a plain dict summarizing rows
    affected by each change -- meant to be printed, not silent.
    """
    if sector_label_merge_map is None:
        sector_label_merge_map = DEFAULT_SECTOR_LABEL_MERGE_MAP

    report: dict = {
        "rate_clipping": {},
        "sector_merge": {},
        "row_count_before": df.height,
    }

    out = df
    total_abs_fv = out.select(pl.col("FV").abs().sum()).item() or 0.0

    for col in rate_cols:
        if col not in out.columns:
            continue
        flagged_mask = pl.col(col).abs() > rate_ceiling
        n_non_null = out.select(pl.col(col).is_not_null().sum()).item() or 0
        n_flagged = out.select(flagged_mask.sum()).item() or 0
        flagged_fv = out.select(
            pl.col("FV").abs().filter(flagged_mask).sum()
        ).item() or 0.0

        out = out.with_columns(
            pl.when(flagged_mask).then(None).otherwise(pl.col(col)).alias(col)
        )

        report["rate_clipping"][col] = {
            "n_flagged": int(n_flagged),
            "pct_of_nonnull": round(100 * n_flagged / n_non_null, 4) if n_non_null else 0.0,
            "pct_of_aum_estimate": round(100 * flagged_fv / total_abs_fv, 4) if total_abs_fv else 0.0,
        }

    if sector_col in out.columns and sector_label_merge_map:
        for old_label, new_label in sector_label_merge_map.items():
            n_rows = out.select((pl.col(sector_col) == old_label).sum()).item() or 0
            if n_rows:
                report["sector_merge"][f"{old_label} -> {new_label}"] = {"rows": int(n_rows)}

        out = out.with_columns(pl.col(sector_col).replace(sector_label_merge_map))

        for merged_label in sorted(set(sector_label_merge_map.values())):
            n_rows = out.select((pl.col(sector_col) == merged_label).sum()).item() or 0
            report["sector_merge"][f"{merged_label} (combined)"] = {"rows": int(n_rows)}

    report["row_count_after"] = out.height
    return out, report


# =====================
# Coverage / suppression
# =====================

def compute_subindex_coverage(
    df_weighted: pl.DataFrame,
    quarter_col: str = "cal_q",
    cik_col: str = "cik",
    fv_prev_sum_col: str = "FV_prev_sum_q",
    min_positions: int = DEFAULT_MIN_POSITIONS,
    min_distinct_ciks: int = DEFAULT_MIN_DISTINCT_CIKS,
    min_aum_usd: float = DEFAULT_MIN_AUM_USD,
) -> pl.DataFrame:
    """
    One row per quarter present in `df_weighted` (valid or not -- a
    bucket-quarter with zero VALID positions still gets a row here, filled
    with zeros, rather than silently disappearing), with n_positions,
    n_distinct_ciks, aum_prior_q (= FV_prev_sum_q), and boolean
    pass_min_positions / pass_min_ciks / pass_min_aum columns. Counts are
    over the `is_valid` population -- the same population `w_mkt` and
    `aggregate_return_decomposition`'s `n_investments` already use, not the
    raw filtered row count.

    Known simplification: this only covers quarters that appear at all in
    `df_weighted`. A bucket with a quarter entirely absent from the input
    (no rows of any validity) will not get a suppressed row for it -- not a
    scenario the current sector/seniority buckets hit (every bucket has at
    least some rows in every one of the panel's 14 quarters), but worth
    knowing before applying this to a future, sparser dimension.
    """
    all_quarters = df_weighted.select(quarter_col).unique()

    valid_agg = (
        df_weighted.filter(pl.col("is_valid"))
        .group_by(quarter_col)
        .agg([
            pl.len().alias("n_positions"),
            pl.col(cik_col).n_unique().alias("n_distinct_ciks"),
        ])
    )
    aum_agg = (
        df_weighted.group_by(quarter_col)
        .agg(pl.col(fv_prev_sum_col).first().alias("aum_prior_q"))
    )

    coverage = (
        all_quarters
        .join(valid_agg, on=quarter_col, how="left")
        .join(aum_agg, on=quarter_col, how="left")
        .with_columns([
            pl.col("n_positions").fill_null(0),
            pl.col("n_distinct_ciks").fill_null(0),
            pl.col("aum_prior_q").fill_null(0.0),
        ])
        .sort(quarter_col)
        .with_columns([
            (pl.col("n_positions") >= min_positions).alias("pass_min_positions"),
            (pl.col("n_distinct_ciks") >= min_distinct_ciks).alias("pass_min_ciks"),
            (pl.col("aum_prior_q") >= min_aum_usd).alias("pass_min_aum"),
        ])
    )
    return coverage


def flag_first_quarter_artifact(
    df: pl.DataFrame,
    quarter_col: str = "cal_q",
    start_idx: int = DEFAULT_START_IDX,
) -> pl.DataFrame:
    """
    One row per distinct quarter in `df`, with a boolean `is_inception_quarter`
    flag = True for the `start_idx` earliest quarters (by the same
    year*4+quarter ordering `qsort_expr()` uses), applied unconditionally --
    independent of whatever `compute_subindex_coverage` finds, since an
    inception-quarter artifact can look adequately populated.
    """
    ranked = (
        df.select(quarter_col).unique()
        .with_columns(
            (
                pl.col(quarter_col).str.slice(0, 4).cast(pl.Int32) * 4
                + pl.col(quarter_col).str.slice(-1).cast(pl.Int32)
            ).alias("_qsort")
        )
        .sort("_qsort")
        .with_row_index("_rank")
        .with_columns((pl.col("_rank") < start_idx).alias("is_inception_quarter"))
        .drop(["_qsort", "_rank"])
    )
    return ranked


def apply_suppression(
    coverage_df: pl.DataFrame,
    artifact_df: pl.DataFrame,
    quarter_col: str = "cal_q",
) -> pl.DataFrame:
    """
    Join coverage pass/fail with the inception-quarter flag into one verdict
    per quarter: `suppressed` (bool) and `suppression_reason` (comma-joined
    string, or null if not suppressed).
    """
    joined = (
        coverage_df.join(artifact_df, on=quarter_col, how="left")
        .with_columns(pl.col("is_inception_quarter").fill_null(False))
        .with_columns(
            (
                (~pl.col("pass_min_positions"))
                | (~pl.col("pass_min_ciks"))
                | (~pl.col("pass_min_aum"))
                | pl.col("is_inception_quarter")
            ).alias("suppressed")
        )
        .with_columns([
            pl.when(~pl.col("pass_min_positions")).then(pl.lit("min_positions")).alias("_r1"),
            pl.when(~pl.col("pass_min_ciks")).then(pl.lit("min_ciks")).alias("_r2"),
            pl.when(~pl.col("pass_min_aum")).then(pl.lit("min_aum")).alias("_r3"),
            pl.when(pl.col("is_inception_quarter")).then(pl.lit("first_quarter_artifact")).alias("_r4"),
        ])
        .with_columns(
            pl.concat_list(["_r1", "_r2", "_r3", "_r4"]).list.drop_nulls().alias("_reasons")
        )
        .with_columns(
            pl.when(pl.col("_reasons").list.len() == 0)
              .then(None)
              .otherwise(pl.col("_reasons").list.join(","))
              .alias("suppression_reason")
        )
        .drop(["_r1", "_r2", "_r3", "_r4", "_reasons"])
    )
    return joined


def suppress_return_series(
    result_df: pl.DataFrame,
    suppression_df: pl.DataFrame,
    quarter_col: str = "cal_q",
    return_cols: tuple[str, ...] = (),
    level_col: str | None = None,
    base_level: float = 100.0,
) -> pl.DataFrame:
    """
    Left-joins suppression flags onto `result_df`. For every column in
    `return_cols`, preserves the original as `{col}_raw` and zeroes the
    public column where suppressed. If `level_col` is given, recomputes it
    from the suppressed-adjusted first `return_cols` entry, so a suppressed
    quarter holds the index level flat instead of compounding a number known
    to be unreliable. No rows are ever dropped -- suppression is always a
    visible flag, never a silent gap.
    """
    out = (
        result_df.join(
            suppression_df.select([quarter_col, "suppressed", "suppression_reason"]),
            on=quarter_col,
            how="left",
        )
        .with_columns(pl.col("suppressed").fill_null(False))
    )

    for col in return_cols:
        out = out.with_columns(pl.col(col).alias(f"{col}_raw")).with_columns(
            pl.when(pl.col("suppressed")).then(0.0).otherwise(pl.col(col)).alias(col)
        )

    if level_col is not None and return_cols:
        primary_return_col = return_cols[0]
        out = out.sort(quarter_col).with_columns(
            ((1 + pl.col(primary_return_col)).cum_prod() * base_level).alias(level_col)
        )

    return out


# =====================
# Orchestration
# =====================

def build_subindex(
    df: pl.DataFrame,
    dimension_col: str,
    bucket_name: str,
    bucket_values: list,
    *,
    min_positions: int = DEFAULT_MIN_POSITIONS,
    min_distinct_ciks: int = DEFAULT_MIN_DISTINCT_CIKS,
    min_aum_usd: float = DEFAULT_MIN_AUM_USD,
    start_idx: int = DEFAULT_START_IDX,
    quarter_col: str = "cal_q",
    cik_col: str = "cik",
) -> dict:
    """
    Build one named sub-index: filter `df` to `dimension_col` in
    `bucket_values`, then run the same market weights / return decomposition
    / flow-based index functions `index_construction.py` uses for the
    whole market, plus coverage-based suppression of unreliable quarters.

    A single bucket_values list covers both "one bucket per raw category
    value" (e.g. bucket_values=["Technology"]) and "one bucket = a custom
    multi-value union" (e.g. the 5-value Senior Loans grouping) -- no
    per-dimension branching needed.

    `df` must already be the flowed panel: the output of
    `compute_final_interest_rates()` -> `compute_position_level_flows()`
    (same point as right before the whole-market `compute_market_weights()`
    call). Market weights are always (re)computed on the filtered subset
    here, so they renormalize to sum to 1 within the bucket per quarter --
    this is what makes it a genuine sub-index rather than a slice of the
    whole-market weights.

    Returns a dict: {"positions", "decomp", "index", "coverage", "meta"}.
    """
    subset = df.filter(pl.col(dimension_col).is_in(bucket_values))

    positions = compute_market_weights(subset, safe_div, quarter_col=quarter_col)
    decomp = aggregate_return_decomposition(positions, safe_div, quarter_col=quarter_col)
    index = compute_flow_based_market_index(
        positions, quarter_col=quarter_col, index_name=bucket_name
    )

    coverage = compute_subindex_coverage(
        positions,
        quarter_col=quarter_col,
        cik_col=cik_col,
        min_positions=min_positions,
        min_distinct_ciks=min_distinct_ciks,
        min_aum_usd=min_aum_usd,
    )
    artifact = flag_first_quarter_artifact(coverage, quarter_col=quarter_col, start_idx=start_idx)
    suppression = apply_suppression(coverage, artifact, quarter_col=quarter_col)

    decomp = suppress_return_series(
        decomp, suppression, quarter_col=quarter_col,
        return_cols=("contrib_price", "contrib_cash", "contrib_pik"),
    )
    index = suppress_return_series(
        index, suppression, quarter_col=quarter_col,
        return_cols=("IndexReturn",), level_col="IndexLevel",
    )

    return {
        "positions": positions,
        "decomp": decomp,
        "index": index,
        "coverage": suppression,
        "meta": {
            "dimension_col": dimension_col,
            "bucket_name": bucket_name,
            "bucket_values": list(bucket_values),
            "quarter_col": quarter_col,
            "thresholds": {
                "min_positions": min_positions,
                "min_distinct_ciks": min_distinct_ciks,
                "min_aum_usd": min_aum_usd,
            },
            "start_idx": start_idx,
        },
    }


def build_all_subindices_for_dimension(
    df: pl.DataFrame,
    dimension_col: str,
    bucket_map: dict[str, list] | None = None,
    **build_subindex_kwargs,
) -> dict[str, dict]:
    """
    Build one `build_subindex` result per bucket.

    If `bucket_map` is None, auto-builds one bucket per distinct value of
    `dimension_col` -- this is how every sector (including the merged
    "Other / Unclassified" catch-all) and every seniority value (including
    "OTHER_UNKNOWN") gets its own reportable sub-index by default, with no
    special-casing of catch-all buckets.

    If `bucket_map` is given (e.g. {"Senior Loans": DEFAULT_SENIOR_LOANS_BUCKET}),
    builds exactly those named, possibly multi-value, groupings instead.
    """
    if bucket_map is None:
        values = (
            df.select(pl.col(dimension_col).unique().drop_nulls())
            .to_series()
            .sort()
            .to_list()
        )
        bucket_map = {v: [v] for v in values}

    return {
        name: build_subindex(df, dimension_col, name, values, **build_subindex_kwargs)
        for name, values in bucket_map.items()
    }


# =====================
# Cross-bucket helpers
# =====================

def stack_subindex_results(
    results: dict[str, dict],
    dimension_col: str,
    frame: str = "index",
) -> pl.DataFrame:
    """
    Long-format concatenation of one frame kind ("index", "decomp", or
    "coverage") across every bucket in `results`, with `dimension` and
    `bucket` columns added -- for faceted plotting / cross-bucket
    comparison. Kept as an opt-in derived helper rather than the primary
    return shape of `build_subindex`, since each bucket's result bundles
    heterogeneous frames that don't share one row shape.
    """
    frames = []
    for bucket_name, result in results.items():
        f = result[frame].with_columns([
            pl.lit(dimension_col).alias("dimension"),
            pl.lit(bucket_name).alias("bucket"),
        ])
        frames.append(f)
    return pl.concat(frames, how="diagonal_relaxed")


def summarize_suppressed_quarters(
    results: dict[str, dict],
    quarter_col: str = "cal_q",
) -> pl.DataFrame:
    """
    One row per (bucket, quarter) across every bucket in `results` where
    that quarter was suppressed, with n_positions / n_distinct_ciks /
    aum_prior_q / suppression_reason -- the audit table to print right
    after building a full sweep.
    """
    rows = []
    for bucket_name, result in results.items():
        cov = result["coverage"].filter(pl.col("suppressed"))
        if cov.height == 0:
            continue
        rows.append(
            cov.with_columns(pl.lit(bucket_name).alias("bucket")).select(
                ["bucket", quarter_col, "n_positions", "n_distinct_ciks", "aum_prior_q", "suppression_reason"]
            )
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "bucket": pl.Utf8,
                quarter_col: pl.Utf8,
                "n_positions": pl.Int64,
                "n_distinct_ciks": pl.UInt32,
                "aum_prior_q": pl.Float64,
                "suppression_reason": pl.Utf8,
            }
        )

    return pl.concat(rows, how="diagonal_relaxed").sort(["bucket", quarter_col])
