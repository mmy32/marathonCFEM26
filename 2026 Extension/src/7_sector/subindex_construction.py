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
out of scope here; the whole-market index in index_construction.py does that.

Positions are matched across quarters on borrower + seniority (`position_key`,
see index_construction.add_position_key), not on investment_identifier text,
which BDCs reword between filings. compute_position_level_flows() first combines
a BDC's rows for the same position in a quarter, so each (cik, position_key,
quarter) is one row and the lag is deterministic.

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

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py
from paths import add_code_dirs_to_path
add_code_dirs_to_path()  # index_construction lives in 2_index_construction/

import polars as pl

from index_construction import (
    safe_div,
    compute_market_weights,
    aggregate_return_decomposition,
    compute_flow_based_market_index,
    flag_first_quarter_artifact,
    DEFAULT_START_IDX,
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


# =====================
# Data-quality preparation
# =====================

def prepare_subindex_input(
    df: pl.DataFrame,
    rate_cols: tuple[str, ...] = ("rate_cash", "interest_rate", "rate_pik"),
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
        # .first() is safe, not merely convenient: fv_prev_sum_col is built in
        # compute_market_weights as `valid_fv_expr.sum().over(quarter_col)`, a
        # window aggregate that polars broadcasts identically to every row of
        # a quarter group by construction -- it cannot differ across rows of
        # the same quarter, regardless of row order, as long as df_weighted is
        # compute_market_weights's direct output (no reordering/filtering of
        # individual rows in between, which build_subindex's call sequence
        # guarantees). Verified empirically too: 0 of 168 (bucket, quarter)
        # groups across every current sector/seniority bucket have more than
        # one distinct fv_prev_sum_col value.
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


# =====================
# Per-sector plots
# =====================

# Sector labels that mean "we don't know the sector". Excluded from the
# per-sector plots: a catch-all bucket can't tell you which sector lags.
DEFAULT_UNKNOWN_SECTOR_LABELS = (
    "Other / Unclassified",
    "Other / Unknown / Unresolved",
    "Other / Unknown",
)

_SECTOR_COLOR = "#2a78d6"   # sector line
_MARKET_COLOR = "#8a8985"   # BDC-market reference (neutral gray)
_BEAT_COLOR = "#2a78d6"     # quarter where the sector beat the market
_LAG_COLOR = "#eb6834"      # quarter where the sector lagged the market
_INK = "#0b0b0b"
_INK_MUTED = "#52514e"


def build_market_reference_index(
    flows: pl.DataFrame,
    quarter_col: str = "cal_q",
    start_idx: int = DEFAULT_START_IDX,
    base_level: float = 100.0,
) -> pl.DataFrame:
    """
    BDC-market flow-based index (all BDC positions, every sector) to
    compare each sector against. The first
    `start_idx` quarters get a zero return, the same treatment every sub-index
    gets for the inception-quarter artifact, so both lines start from the
    same footing.
    """
    weighted = compute_market_weights(flows, safe_div, quarter_col=quarter_col)
    market = compute_flow_based_market_index(
        weighted, quarter_col=quarter_col, index_name="Market"
    ).sort(quarter_col)
    return (
        market.with_row_index("_i")
        .with_columns(
            pl.when(pl.col("_i") < start_idx).then(0.0)
              .otherwise(pl.col("IndexReturn")).alias("IndexReturn")
        )
        .with_columns(((1 + pl.col("IndexReturn")).cum_prod() * base_level).alias("IndexLevel"))
        .drop("_i")
    )


def _safe_filename(name: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in name]
    return "_".join(filter(None, "".join(keep).split("_")))


def plot_sector_subindices(
    sector_results: dict[str, dict],
    market_index: pl.DataFrame,
    out_dir = _Path(__file__).resolve().parents[2] / "outputs" / "sector_subindices_plots",
    exclude: tuple[str, ...] = DEFAULT_UNKNOWN_SECTOR_LABELS,
    quarter_col: str = "cal_q",
    start_idx: int = DEFAULT_START_IDX,
    periods_per_year: int = 4,
    show: bool = False,
):
    """
    One figure per sector (saved as its own PNG in `out_dir`), skipping the
    unknown / unclassified buckets in `exclude`.

    Each figure has two panels:
      - top: sector index level vs. the BDC-market index (both start at 100)
      - bottom: sector quarterly return minus BDC-market quarterly return, in
        percentage points; orange bars are quarters the sector lagged.
    Suppressed quarters (thin coverage or the inception-quarter artifact) are
    marked with a gray "s" under the axis, since their return is held at 0.

    Every figure uses the same y-limits so sectors can be compared across
    separate images. Returns a pandas summary table ranked worst-first by
    annualized excess return vs. the BDC market.
    """
    import math
    from pathlib import Path

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import pandas as pd

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mkt = market_index.select([quarter_col, "IndexReturn", "IndexLevel"]).rename(
        {"IndexReturn": "mkt_ret", "IndexLevel": "mkt_level"}
    )

    frames = {}
    for name, res in sector_results.items():
        if name in exclude:
            continue
        f = (
            res["index"].select([quarter_col, "IndexReturn", "IndexLevel", "suppressed"])
            .join(mkt, on=quarter_col, how="inner")
            .sort(quarter_col)
            .with_columns((pl.col("IndexReturn") - pl.col("mkt_ret")).alias("excess"))
            .to_pandas()
        )
        frames[name] = f

    if not frames:
        raise ValueError("No sectors left to plot after exclusions.")

    # Shared y-limits across every figure
    lvl_vals = pd.concat([pd.concat([f["IndexLevel"], f["mkt_level"]]) for f in frames.values()])
    exc_vals = pd.concat([f["excess"].iloc[start_idx:] * 100 for f in frames.values()])
    lvl_pad = (lvl_vals.max() - lvl_vals.min()) * 0.08
    lvl_lim = (lvl_vals.min() - lvl_pad, lvl_vals.max() + lvl_pad)
    exc_abs = max(abs(exc_vals.min()), abs(exc_vals.max())) * 1.15 or 1.0
    exc_lim = (-exc_abs, exc_abs)

    summary_rows = []
    for name, f in frames.items():
        rets = f["IndexReturn"].iloc[start_idx:]
        mrets = f["mkt_ret"].iloc[start_idx:]
        n = len(rets)
        ann = (1 + rets).prod() ** (periods_per_year / n) - 1 if n else math.nan
        mann = (1 + mrets).prod() ** (periods_per_year / n) - 1 if n else math.nan
        cum = f["IndexLevel"].iloc[-1] / 100 - 1
        mcum = f["mkt_level"].iloc[-1] / 100 - 1
        n_lag = int((f["excess"].iloc[start_idx:] < 0).sum())
        n_supp = int(f["suppressed"].iloc[start_idx:].sum())
        summary_rows.append({
            "sector": name,
            "cum_return": cum,
            "bdc_market_cum_return": mcum,
            "ann_return": ann,
            "ann_excess_vs_bdc_market": ann - mann,
            "quarters_lagging": n_lag,
            "quarters_total": n,
            "quarters_suppressed": n_supp,
        })

        x = list(range(len(f)))
        labels = f[quarter_col].astype(str).tolist()

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6.5), sharex=True,
            gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
        )

        ax1.plot(x, f["mkt_level"], color=_MARKET_COLOR, lw=2, ls="--", label="BDC market")
        ax1.plot(x, f["IndexLevel"], color=_SECTOR_COLOR, lw=2, label=name)
        # Direct end labels
        ax1.annotate(f"{f['IndexLevel'].iloc[-1]:.1f}", (x[-1], f["IndexLevel"].iloc[-1]),
                     xytext=(6, 0), textcoords="offset points", va="center",
                     color=_INK, fontsize=9)
        ax1.annotate(f"{f['mkt_level'].iloc[-1]:.1f}", (x[-1], f["mkt_level"].iloc[-1]),
                     xytext=(6, 0), textcoords="offset points", va="center",
                     color=_INK_MUTED, fontsize=9)
        ax1.set_ylim(*lvl_lim)
        ax1.set_ylabel("Index level (start = 100)", color=_INK_MUTED)
        ax1.legend(loc="upper left", frameon=False)
        ax1.set_title(
            f"{name} sub-index vs. BDC market\n"
            f"Cumulative {cum:+.1%} vs. BDC market {mcum:+.1%}  |  "
            f"annualized excess {ann - mann:+.2%}  |  lagged in {n_lag} of {n} quarters",
            loc="left", fontsize=11, color=_INK,
        )

        exc = f["excess"] * 100
        colors = [_LAG_COLOR if v < 0 else _BEAT_COLOR for v in exc]
        bars = exc.copy()
        bars.iloc[:start_idx] = 0.0
        ax2.bar(x, bars, color=colors, width=0.6)
        ax2.axhline(0, color=_INK_MUTED, lw=0.8)
        ax2.set_ylim(*exc_lim)
        ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
        ax2.set_ylabel("Return vs. BDC market\n(pct. points, QoQ)", color=_INK_MUTED)

        for i, s in enumerate(f["suppressed"]):
            if s:
                ax2.annotate("s", (i, exc_lim[0]), xytext=(0, 2), textcoords="offset points",
                             ha="center", va="bottom", color=_MARKET_COLOR, fontsize=9)

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        ax2.set_xlabel("Quarter", color=_INK_MUTED)

        for ax in (ax1, ax2):
            ax.grid(axis="y", color="#e4e3df", lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=_INK_MUTED)

        fig.text(0.01, 0.005,
                 "Orange = sector lagged the BDC market that quarter.  "
                 "s = suppressed quarter (thin coverage or first-quarter artifact), return held at 0.  "
                 "Same y-axis scale on every sector figure.",
                 fontsize=7.5, color=_INK_MUTED)
        fig.subplots_adjust(left=0.1, right=0.93, top=0.88, bottom=0.17)
        fig.savefig(out_path / f"subindex_{_safe_filename(name)}.png", dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("ann_excess_vs_bdc_market")
        .reset_index(drop=True)
    )
    return summary


if __name__ == "__main__":
    from paths import FINAL_DIR
    from index_construction import (
        load_and_prepare_investment_data, compute_final_interest_rates,
        qsort_expr, safe_div, compute_position_level_flows,
    )

    CSV_PATH = FINAL_DIR / "FINAL_DATASET.csv"

    print("=== subindex_construction.py: sector / seniority sub-indices ===")
    df = load_and_prepare_investment_data(str(CSV_PATH))
    df, report = prepare_subindex_input(df)
    print("data-quality report:", report)

    df = compute_final_interest_rates(df)
    flows = compute_position_level_flows(df, qsort_expr, safe_div)

    sector_results = build_all_subindices_for_dimension(flows, "sector")
    print(f"\nbuilt {len(sector_results)} sector sub-indices: {sorted(sector_results)}")

    seniority_results = build_all_subindices_for_dimension(
        flows, "instrument_seniority",
        bucket_map={"Senior Loans": DEFAULT_SENIOR_LOANS_BUCKET},
    )
    print(f"built {len(seniority_results)} custom seniority bucket(s): {sorted(seniority_results)}")

    all_results = {**sector_results, **seniority_results}
    suppressed = summarize_suppressed_quarters(all_results)
    print("\n--- suppressed bucket-quarters (thin coverage or inception-quarter artifact) ---")
    print(suppressed if suppressed.height else "none")

    sample_name = next(iter(sector_results))
    print(f"\n--- sample sub-index: {sample_name} ---")
    print(sector_results[sample_name]["index"].select(["cal_q", "IndexReturn", "IndexLevel", "suppressed"]))

    market_index = build_market_reference_index(flows)
    sector_summary = plot_sector_subindices(sector_results, market_index)
    print("\n--- sector sub-indices vs. BDC market (worst first; unknown sectors excluded) ---")
    print(sector_summary.to_string(
        index=False,
        formatters={c: "{:+.2%}".format for c in
                    ["cum_return", "bdc_market_cum_return", "ann_return", "ann_excess_vs_bdc_market"]},
    ))
    print("figures saved to outputs/sector_subindices_plots/")
