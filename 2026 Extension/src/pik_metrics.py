"""
pik_metrics.py

Core analytical library for BDC private credit PIK (Paid-In-Kind) risk detection,
provenance tracking, transition classification, and non-cash trend measurement.

Designed with separation of concerns and DRY principles:
- Reusable transformation and aggregation logic for Polars DataFrames.
- Composable with `index_construction.py` and `subindex_construction.py`.
"""

from typing import Optional
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from index_construction import safe_div

def quarter_sort_expr(quarter_col: str = "cal_q") -> pl.Expr:
    """Integer sorting key for YYYYQ# quarterly strings."""
    y = pl.col(quarter_col).str.slice(0, 4).cast(pl.Int32)
    q = pl.col(quarter_col).str.slice(-1).cast(pl.Int32)
    return (y * 4 + q).alias("qsort")


# ==============================================================================
# 0. Disclosed vs. Inferred PIK Classification
# ==============================================================================

def classify_pik_provenance(
    df: pl.DataFrame,
    raw_pik_col: str = "rate_pik",
    final_pik_col: str = "PIK_Final",
    check2_col: str = "check_2",
) -> pl.DataFrame:
    """
    Classify each row by the provenance of its PIK interest rate:
      - 'explicit_pik': Direct non-null iXBRL tag present (rate_pik > 0).
      - 'inferred_pik': Tag was absent (null), but PIK_Final > 0 was derived
                        via interest_rate - rate_cash or upstream heuristics.
      - 'explicit_pik': Both cash and PIK were tagged in the filing (check_2 in
                        ['rate, pik, pic', 'spread, rate, pik, pic']).
      - 'inferred_pik': PIK was tagged but cash/rate had to be inferred via
                        estimate-pik, rate-pik, or single tag (check_2 in
                        ['spread, rate, pik', 'spread, pik', 'pik', ...]),
                        or raw_pik was null and derived by upstream heuristics.
      - 'zero_pik': No PIK rate present or derived (PIK_Final <= 0 or null).

    Adds:
      - `pik_provenance`: Categorical string label.
      - `is_explicit_pik`: Boolean indicator.
      - `is_inferred_pik`: Boolean indicator.
    """
    has_final_pik = (pl.col(final_pik_col).is_not_null()) & (pl.col(final_pik_col) > 0)
    has_raw_pik = (pl.col(raw_pik_col).is_not_null()) & (pl.col(raw_pik_col) > 0)
    has_final_pik = (pl.col(final_pik_col).fill_null(0) > 0)
    has_final_pik = pl.col(final_pik_col).fill_null(0) > 0

    if check2_col in df.columns:
        is_explicit = pl.col(check2_col).is_in(["rate, pik, pic", "spread, rate, pik, pic"]).fill_null(False)
    else:
        # Fallback for synthetic/minimal frames
        is_explicit = (pl.col(raw_pik_col).is_not_null()) & (pl.col(raw_pik_col) > 0)

    provenance_expr = (
        pl.when(has_final_pik & has_raw_pik)
        pl.when(has_final_pik & is_explicit)
        .then(pl.lit("explicit_pik"))
        .when(has_final_pik & ~has_raw_pik)
        .when(has_final_pik & ~is_explicit)
        .then(pl.lit("inferred_pik"))
        .otherwise(pl.lit("zero_pik"))
        .alias("pik_provenance")
    )

    return df.with_columns([
        provenance_expr,
        (has_final_pik & has_raw_pik).alias("is_explicit_pik"),
        (has_final_pik & ~has_raw_pik).alias("is_inferred_pik"),
        (has_final_pik & is_explicit).alias("is_explicit_pik"),
        (has_final_pik & ~is_explicit).alias("is_inferred_pik"),
    ])


def aggregate_pik_provenance(
    df: pl.DataFrame,
    quarter_col: str = "cal_q",
    cik_col: str = "cik",
    fv_col: str = "FV",
    par_col: str = "PAR",
) -> dict[str, pl.DataFrame]:
    """
    Aggregate PIK provenance metrics across quarters and across BDC filers.

    Returns a dict with:
      - 'quarterly': Position counts, percentages, total FV, and PAR volume
                     for explicit, inferred, and zero PIK by quarter.
      - 'by_cik': Filer-level summary quantifying BDC reporting habits.
    """
    if "pik_provenance" not in df.columns:
        df = classify_pik_provenance(df)

    # Quarterly aggregation
    quarterly = (
        df.group_by([quarter_col, "pik_provenance"])
        .agg([
            pl.len().alias("n_positions"),
            pl.col(fv_col).fill_null(0).sum().alias("total_fv"),
            pl.col(par_col).fill_null(0).sum().alias("total_par"),
        ])
        .with_columns([
            (pl.col("n_positions") / pl.col("n_positions").sum().over(quarter_col)).alias("pct_positions"),
            (pl.col("total_fv") / pl.col("total_fv").sum().over(quarter_col)).alias("pct_fv"),
        ])
        .sort([quarter_col, "pik_provenance"])
    )

    # Cross-tabulation by BDC CIK
    by_cik = (
        df.group_by(cik_col)
        .agg([
            pl.len().alias("total_positions"),
            (pl.col("pik_provenance") == "explicit_pik").sum().alias("n_explicit_pik"),
            (pl.col("pik_provenance") == "inferred_pik").sum().alias("n_inferred_pik"),
            (pl.col("pik_provenance") == "zero_pik").sum().alias("n_zero_pik"),
            pl.col(fv_col).fill_null(0).sum().alias("total_fv"),
        ])
        .with_columns([
            (pl.col("n_explicit_pik") / pl.col("total_positions")).alias("pct_explicit"),
            (pl.col("n_inferred_pik") / pl.col("total_positions")).alias("pct_inferred"),
        ])
        .sort("total_positions", descending=True)
    )

    return {"quarterly": quarterly, "by_cik": by_cik}


# ==============================================================================
# 1. Contractual PIK vs. Distress Amendment Classification
# ==============================================================================

def classify_loan_pik_transitions(
    df: pl.DataFrame,
    quarter_col: str = "cal_q",
    cik_col: str = "cik",
    id_col: str = "investment_identifier",
    final_pik_col: str = "PIK_Final",
    par_col: str = "PAR",
    min_amendment_bps: float = 0.015,
) -> pl.DataFrame:
    """
    Track loan-level interest structure over consecutive quarters within each
    position (cik, investment_identifier) and classify PIK state transitions:

      - 'contractual_pik': Active in position's inception quarter with PIK > 0.
      - 'distress_amendment_pik': Seasoned position converting from PIK=0 -> PIK>0,
                                  or jumping by >= min_amendment_bps (default 150 bps).
      - 'stable_cash': Position maintaining PIK <= 0.
      - 'cured_pik': Position transitioning from PIK > 0 back to PIK <= 0.
      - 'seasoned_pik_continuation': Seasoned position with ongoing contractual PIK
                                     without an amendment jump.

    Adds:
      - `PAR_incept`: Principal at first observation of the loan.
      - `PIK_prev`: Prior-quarter PIK_Final.
      - `delta_pik`: Quarter-over-quarter change in PIK rate.
      - `pik_transition_state`: Categorical transition classification.
      - `is_distress_amendment`: Boolean indicator.
    """
    group_cols = [cik_col, id_col]

    # Chronological sort across quarters
    df_sorted = df.sort([cik_col, id_col, quarter_sort_expr(quarter_col)])

    # Inception and lag tracking
    df_tracked = df_sorted.with_columns([
        pl.col(quarter_col).first().over(group_cols).alias("q_incept"),
        pl.col(par_col).first().over(group_cols).alias("PAR_incept"),
        pl.col(final_pik_col).shift(1).over(group_cols).alias("PIK_prev"),
        pl.col(par_col).shift(1).over(group_cols).alias("PAR_prev"),
    ]).with_columns([
        (pl.col(quarter_col) == pl.col("q_incept")).alias("is_inception"),
        (pl.col(final_pik_col).fill_null(0) - pl.col("PIK_prev").fill_null(0)).alias("delta_pik"),
        (pl.col(final_pik_col).fill_null(0) > 0).alias("curr_has_pik"),
        (pl.col("PIK_prev").fill_null(0) > 0).alias("prev_has_pik"),
    ])

    # Classify state
    is_incept = pl.col("is_inception")
    curr_pik = pl.col("curr_has_pik")
    prev_pik = pl.col("prev_has_pik")
    jump_pik = pl.col("delta_pik") >= min_amendment_bps

    transition_expr = (
        pl.when(is_incept & curr_pik)
        .then(pl.lit("contractual_pik"))
        .when(is_incept & ~curr_pik)
        .then(pl.lit("stable_cash"))
        .when(~is_incept & ~prev_pik & curr_pik)
        .then(pl.lit("distress_amendment_pik"))
        .when(~is_incept & prev_pik & jump_pik)
        .then(pl.lit("distress_amendment_pik"))
        .when(~is_incept & prev_pik & ~curr_pik)
        .then(pl.lit("cured_pik"))
        .when(~is_incept & ~prev_pik & ~curr_pik)
        .then(pl.lit("stable_cash"))
        .otherwise(pl.lit("seasoned_pik_continuation"))
        .alias("pik_transition_state")
    )

    return df_tracked.with_columns([
        transition_expr,
        (
            (~is_incept & ~prev_pik & curr_pik) | (~is_incept & prev_pik & jump_pik)
        ).alias("is_distress_amendment"),
    ])


def aggregate_pik_transitions(
    df: pl.DataFrame,
    quarter_col: str = "cal_q",
    fv_col: str = "FV",
    par_col: str = "PAR",
) -> pl.DataFrame:
    """
    Aggregate loan-level transition classifications into quarterly summary metrics.

    Produces:
      - Counts and Fair Value across all 5 transition states.
      - pct_amendment_count: Share of seasoned positions converting to amendment PIK.
      - pct_amendment_fv: Share of portfolio fair value in distress amendment.
      - total_amendment_fv: Absolute dollar volume of distress amendments.
    """
    if "pik_transition_state" not in df.columns:
        df = classify_loan_pik_transitions(df, quarter_col=quarter_col)

    # Base counts and sums per quarter and state
    state_agg = (
        df.group_by([quarter_col, "pik_transition_state"])
        .agg([
            pl.len().alias("count"),
            pl.col(fv_col).fill_null(0).sum().alias("fv_sum"),
            pl.col(par_col).fill_null(0).sum().alias("par_sum"),
        ])
    )

    # Pivot into wide format for clean quarterly reporting
    pivoted = (
        state_agg.pivot(
            on="pik_transition_state",
            index=quarter_col,
            values=["count", "fv_sum"],
        )
        .sort(quarter_sort_expr(quarter_col))
    )

    # Add portfolio-level relative shares
    q_totals = (
        df.group_by(quarter_col)
        .agg([
            pl.len().alias("total_positions"),
            pl.col(fv_col).fill_null(0).sum().alias("total_fv"),
            (pl.col("is_inception") == False).sum().alias("seasoned_positions"),
            (pl.col("is_distress_amendment") == True).sum().alias("n_amendments"),
            pl.col(fv_col).filter(pl.col("is_distress_amendment") == True).fill_null(0).sum().alias("amendment_fv"),
        ])
        .with_columns([
            (pl.col("n_amendments") / pl.col("seasoned_positions")).alias("pct_amendment_count"),
            (pl.col("amendment_fv") / pl.col("total_fv")).alias("pct_amendment_fv"),
        ])
    )

    return q_totals.join(pivoted, on=quarter_col, how="left").sort(quarter_sort_expr(quarter_col))


# ==============================================================================
# 2. Non-Cash Portion of the Loan (Three Distinct Metrics)
# ==============================================================================

def compute_non_cash_metrics(
    df: pl.DataFrame,
    ir_col: str = "IR_Final",
    pik_col: str = "PIK_Final",
    cash_income_col: str = "cash_income",
    pik_income_col: str = "pik_income",
    par_col: str = "PAR",
    par_incept_col: str = "PAR_incept",
) -> pl.DataFrame:
    """
    Compute three distinct non-cash metrics at the position level:

      - Metric A: Contractual Coupon Share (Rate Spread)
            rate_share = PIK_Final / IR_Final
            Measures contracted non-cash fraction of total interest coupon.

      - Metric B: Periodic Income Accrual Share (Flow Share)
            income_share = pik_income / (cash_income + pik_income)
            Measures accounting accrued non-cash flow credited to BDC.

      - Metric C: Cumulative Capitalized Principal Burden (Stock Share)
            cap_burden = (PAR_t - PAR_incept) / PAR_t
            Measures balance growth from accumulated unpaid interest compounding into principal.
    """
    cols_to_add = []

    # Metric A: Contractual Coupon Share
    cols_to_add.append(
        safe_div(pl.col(pik_col).fill_null(0), pl.col(ir_col)).clip(0.0, 1.0).alias("rate_share")
    )

    # Metric B: Periodic Income Accrual Share
    if cash_income_col in df.columns and pik_income_col in df.columns:
        total_income = pl.col(cash_income_col).fill_null(0) + pl.col(pik_income_col).fill_null(0)
        cols_to_add.append(
            safe_div(pl.col(pik_income_col).fill_null(0), total_income).clip(0.0, 1.0).alias("income_share")
        )
    else:
        # Fallback if flow columns not pre-computed
        cols_to_add.append(pl.lit(None).cast(pl.Float64).alias("income_share"))

    # Metric C: Cumulative Capitalized Principal Burden
    if par_incept_col in df.columns and par_col in df.columns:
        cap_growth = pl.col(par_col) - pl.col(par_incept_col)
        # Cap burden clipped to [-1.0, 1.0] to prevent divide-by-near-zero distortion from liquidated loans
        cols_to_add.extend([
            safe_div(cap_growth, pl.col(par_col)).clip(-1.0, 1.0).alias("cap_burden"),
            pl.when(cap_growth > 0)
            .then(safe_div(cap_growth, pl.col(par_col)).clip(0.0, 2.0))
            .otherwise(0.0)
            .alias("cap_expansion"),
        ])
    else:
        cols_to_add.extend([
            pl.lit(None).cast(pl.Float64).alias("cap_burden"),
            pl.lit(None).cast(pl.Float64).alias("cap_expansion"),
        ])

    return df.with_columns(cols_to_add)


def aggregate_non_cash_metrics(
    df: pl.DataFrame,
    quarter_col: str = "cal_q",
    w_mkt_col: str = "w_mkt",
) -> pl.DataFrame:
    """
    Aggregate position-level non-cash metrics into quarterly market averages.

    Computes market-value-weighted and cross-sectional mean shares for:
      - rate_share (Metric A)
      - income_share (Metric B)
      - cap_burden & cap_expansion (Metric C)
    """
    if "rate_share" not in df.columns:
        df = compute_non_cash_metrics(df)

    has_weights = w_mkt_col in df.columns

    aggs = [
        pl.len().alias("n_positions"),
        # Metric A
        pl.col("rate_share").mean().alias("mean_rate_share"),
        pl.col("rate_share").median().alias("median_rate_share"),
        # Metric B
        pl.col("income_share").mean().alias("mean_income_share"),
        pl.col("income_share").median().alias("median_income_share"),
        # Metric C
        pl.col("cap_burden").filter(pl.col("cap_burden").is_not_null()).mean().alias("mean_cap_burden"),
        pl.col("cap_expansion").filter(pl.col("cap_expansion") > 0).mean().alias("mean_pos_expansion"),
        (pl.col("cap_expansion") > 0).mean().alias("pct_loans_expanded"),
    ]

    if has_weights:
        aggs.extend([
            (pl.col(w_mkt_col) * pl.col("rate_share").fill_null(0)).sum().alias("weighted_rate_share"),
            (pl.col(w_mkt_col) * pl.col("income_share").fill_null(0)).sum().alias("weighted_income_share"),
            (pl.col(w_mkt_col) * pl.col("cap_expansion").fill_null(0)).sum().alias("weighted_cap_expansion"),
        ])

    return (
        df.group_by(quarter_col)
        .agg(aggs)
        .sort(quarter_sort_expr(quarter_col))
    )


# ==============================================================================
# 3. Parametric Multi-Period & Sector Sweeps
# ==============================================================================

def compute_sector_time_series_pik_metrics(
    df: pl.DataFrame,
    sector_col: str = "sector",
    quarter_col: str = "cal_q",
    cik_col: str = "cik",
    id_col: str = "investment_identifier",
    w_mkt_col: str = "w_mkt",
    fv_col: str = "FV",
    par_col: str = "PAR",
) -> pl.DataFrame:
    """
    Execute full parametric PIK metric sweeps across distinct market sectors over time.
    Applies DRY principles by reusing the core transition and non-cash calculations.

    Returns a long-format panel per (sector, quarter) containing:
      - Total positions and Fair Value volume
      - Provenance breakdown (explicit vs. inferred PIK)
      - Transition counts (distress amendment count and FV share)
      - Non-cash metrics (weighted rate share, income share, and cap burden)
    """
    # Ensure transition and non-cash columns are populated
    if "pik_transition_state" not in df.columns:
        df = classify_loan_pik_transitions(df, quarter_col=quarter_col, cik_col=cik_col, id_col=id_col, par_col=par_col)
    if "pik_provenance" not in df.columns:
        df = classify_pik_provenance(df)
    if "rate_share" not in df.columns:
        df = compute_non_cash_metrics(df)

    has_weights = w_mkt_col in df.columns

    # Base aggregations per (sector, quarter)
    aggs = [
        pl.len().alias("n_positions"),
        pl.col(fv_col).fill_null(0).sum().alias("total_fv"),
        pl.col(par_col).fill_null(0).sum().alias("total_par"),
        # Provenance
        (pl.col("pik_provenance") == "explicit_pik").sum().alias("n_explicit_pik"),
        (pl.col("pik_provenance") == "inferred_pik").sum().alias("n_inferred_pik"),
        (pl.col("pik_provenance") == "zero_pik").sum().alias("n_zero_pik"),
        # Transition
        (pl.col("is_distress_amendment") == True).sum().alias("n_distress_amendments"),
        pl.col(fv_col).filter(pl.col("is_distress_amendment") == True).fill_null(0).sum().alias("distress_amendment_fv"),
        (pl.col("pik_transition_state") == "contractual_pik").sum().alias("n_contractual_pik"),
        (pl.col("pik_transition_state") == "cured_pik").sum().alias("n_cured_pik"),
        # Non-cash metrics
        pl.col("rate_share").mean().alias("mean_rate_share"),
        pl.col("income_share").mean().alias("mean_income_share"),
        pl.col("cap_burden").filter(pl.col("cap_burden").is_not_null()).mean().alias("mean_cap_burden"),
        pl.col("cap_expansion").filter(pl.col("cap_expansion") > 0).mean().alias("mean_pos_expansion"),
        (pl.col("cap_expansion") > 0).mean().alias("pct_loans_expanded"),
    ]

    if has_weights:
        aggs.extend([
            (pl.col(w_mkt_col) * pl.col("rate_share").fill_null(0)).sum().alias("weighted_rate_share"),
            (pl.col(w_mkt_col) * pl.col("income_share").fill_null(0)).sum().alias("weighted_income_share"),
            (pl.col(w_mkt_col) * pl.col("cap_expansion").fill_null(0)).sum().alias("weighted_cap_expansion"),
        ])

    return (
        df.filter(pl.col(sector_col).is_not_null())
        .group_by([sector_col, quarter_col])
        .agg(aggs)
        .with_columns([
            (pl.col("n_distress_amendments") / pl.col("n_positions")).alias("pct_amendment_count"),
            (pl.col("distress_amendment_fv") / pl.col("total_fv")).alias("pct_amendment_fv"),
        ])
        .sort([sector_col, quarter_sort_expr(quarter_col)])
    )


# ==============================================================================
# 4. Time-Series Visualization Suite
# ==============================================================================

def plot_non_cash_metrics_time_series(
    non_cash_df: pl.DataFrame,
    quarter_col: str = "cal_q",
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the three distinct non-cash metrics across quarters:
      - Metric A: Contractual Coupon Rate Share (PIK / IR)
      - Metric B: Periodic Income Accrual Share (pik_income / total_income)
      - Metric C: Cumulative Capitalized Principal Expansion Share
    """
    pdf = non_cash_df.to_pandas()
    quarters = pdf[quarter_col].astype(str)

    plt.figure(figsize=figsize)
    plt.plot(quarters, pdf["weighted_rate_share"] * 100, marker="o", label="Metric A: Rate Spread Share (PIK/IR, %)", color="#1f77b4", linewidth=2)
    plt.plot(quarters, pdf["weighted_income_share"] * 100, marker="s", label="Metric B: Income Accrual Share (% Flow)", color="#ff7f0e", linewidth=2)
    plt.plot(quarters, pdf["weighted_cap_expansion"] * 100, marker="^", label="Metric C: Compounded Principal Expansion (% Par)", color="#2ca02c", linewidth=2, linestyle="--")

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    plt.title("Evolution of Non-Cash PIK Channels Across US BDCs (2023Q1–2026Q2)", fontsize=12, fontweight="bold")
    plt.xlabel("Filing Quarter", fontsize=10)
    plt.ylabel("Portfolio Weighted Share (%)", fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_distress_amendments_vs_markdowns(
    transitions_df: pl.DataFrame,
    df: pl.DataFrame,
    quarter_col: str = "cal_q",
    figsize: tuple = (11, 5.5),
    save_path: Optional[str] = None,
) -> None:
    """
    Dual-axis time series comparing Distress Amendment Volume against
    portfolio fair value markdowns (FV / Cost < 0.80 and < 0.90).
    """
    # Compute quarterly valuation markdown metrics
    marks_df = (
        df.filter(pl.col("COST") > 0)
        .group_by(quarter_col)
        .agg([
            ((pl.col("FV") / pl.col("COST")) < 0.90).mean().alias("pct_marked_down_10pct"),
            ((pl.col("FV") / pl.col("COST")) < 0.80).mean().alias("pct_marked_down_20pct"),
        ])
    )

    merged = (
        transitions_df.select([quarter_col, "n_amendments", "amendment_fv", "pct_amendment_count"])
        .join(marks_df, on=quarter_col, how="left")
        .sort(quarter_sort_expr(quarter_col))
        .to_pandas()
    )

    # Exclude 2023Q1 inception quarter where amendment transitions are structurally 0
    merged = merged.iloc[1:].reset_index(drop=True)
    quarters = merged[quarter_col].astype(str)

    fig, ax1 = plt.subplots(figsize=figsize)

    color_bar = "#d62728"
    ax1.set_xlabel("Filing Quarter", fontsize=10)
    ax1.set_ylabel("Distress Amendment Volume ($ Billions FV)", color=color_bar, fontsize=10)
    bars = ax1.bar(quarters, merged["amendment_fv"] / 1e9, color=color_bar, alpha=0.55, width=0.45, label="Distress Amendment FV ($B)")
    ax1.tick_params(axis="y", labelcolor=color_bar)
    ax1.set_ylim(0, max(merged["amendment_fv"] / 1e9) * 1.35)

    ax2 = ax1.twinx()
    color_line1 = "#1f77b4"
    color_line2 = "#7f7f7f"
    ax2.set_ylabel("Portfolio Markdown Rate (%)", color=color_line1, fontsize=10)
    l1 = ax2.plot(quarters, merged["pct_marked_down_10pct"] * 100, color=color_line1, marker="o", linewidth=2, label="Loans Marked Down >10% (FV/Cost < 0.90)")
    l2 = ax2.plot(quarters, merged["pct_marked_down_20pct"] * 100, color=color_line2, marker="x", linewidth=2, linestyle="--", label="Loans Marked Down >20% (FV/Cost < 0.80)")
    ax2.tick_params(axis="y", labelcolor=color_line1)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax2.set_ylim(0, max(merged["pct_marked_down_10pct"] * 100) * 1.4)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    plt.title("Lead-Lag Signal: Distress PIK Amendments vs. Valuation Markdowns", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(len(quarters)))
    ax1.set_xticklabels(quarters, rotation=45)
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_sector_distress_amendments(
    sector_df: pl.DataFrame,
    quarter_col: str = "cal_q",
    sector_col: str = "sector",
    top_sectors: tuple = ("Technology", "Healthcare", "Consumer", "Industrials", "Communication & Media", "Financials & Insurance"),
    figsize: tuple = (11, 5.5),
    save_path: Optional[str] = None,
) -> None:
    """
    Multi-line time-series displaying distress amendment rates (% of seasoned loans)
    across key economic sectors.
    """
    pdf = sector_df.filter(pl.col(sector_col).is_in(list(top_sectors))).to_pandas()
    # Exclude 2023Q1 inception quarter
    pdf = pdf[pdf[quarter_col] != "2023Q1"]

    plt.figure(figsize=figsize)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for sector, color in zip(top_sectors, palette):
        sec_data = pdf[pdf[sector_col] == sector].sort_values(quarter_col)
        if len(sec_data) > 0:
            plt.plot(
                sec_data[quarter_col],
                sec_data["pct_amendment_count"] * 100,
                marker="o",
                label=sector,
                color=color,
                linewidth=1.8,
            )

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    plt.title("Distress PIK Amendment Rate by Sector (% of Seasoned Positions)", fontsize=12, fontweight="bold")
    plt.xlabel("Filing Quarter", fontsize=10)
    plt.ylabel("Amendment Conversion Rate (%)", fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()


CIK_NAME_MAP = {
    1803498: "Blackstone Private Credit (BCRED)",
    1476765: "Golub Capital BDC (GBDC)",
    1422183: "FS KKR Capital (FSK)",
    1287750: "Ares Capital (ARCC)",
    1812554: "Blue Owl Credit Income (OCIC)",
    1868878: "Golub Capital Direct Lending",
    1859919: "Barings Private Credit",
    1634452: "AB Private Credit Investors",
    1513363: "Fidus Investment (FDUS)",
    1396440: "Main Street Capital (MAIN)",
    1280784: "Hercules Capital (HTGC)",
    1535778: "MSC Income Fund (MSIF)",
    1737924: "Nuveen Churchill Direct Lending",
    1736035: "Blackstone Secured Lending (BXSL)",
    1901606: "Golub Capital Unlevered",
    1901612: "Golub Capital BDC 4",
    1838126: "HPS Corporate Lending (HLEND)",
    1655050: "Bain Capital Specialty Finance (BCSF)",
    1655888: "Blue Owl Capital Corp (OBDC)",
    1742313: "Monroe Capital Income Plus",
    1512931: "Monroe Capital (MRCC)",
    1379785: "Barings BDC (BBDC)",
    1287032: "Prospect Capital (PSEC)",
    1572694: "Goldman Sachs BDC (GSBD)",
    1143513: "Gladstone Capital (GLAD)",
    1414932: "BlackRock TCP Capital (TCPC)",
    1593314: "Trinity Capital (TRIN)",
    1851322: "Morgan Stanley / North Haven",
    1918712: "Ares Strategic Income (ASIF)",
    1913724: "TPG Twin Brook Capital",
    1993402: "Antares Strategic Credit",
    1950803: "StepStone Private Credit",
}


def plot_pik_provenance_positions(
    prov_df: pl.DataFrame,
    quarter_col: str = "cal_q",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Two-panel time-series tracking PIK data provenance:
      - Left: % of total portfolio positions (Explicit vs Inferred vs Zero PIK)
      - Right: Composition among positive PIK positions (% Explicit vs Inferred)
    """
    pdf = prov_df.to_pandas()
    pivoted_pct = pdf.pivot(index=quarter_col, columns="pik_provenance", values="pct_positions").fillna(0)
    pivoted_cnt = pdf.pivot(index=quarter_col, columns="pik_provenance", values="n_positions").fillna(0)

    quarters = [str(q) for q in pivoted_pct.index]
    total_pik_cnt = (pivoted_cnt.get("explicit_pik", 0) + pivoted_cnt.get("inferred_pik", 0)).replace(0, np.nan)
    pct_explicit_of_pik = (pivoted_cnt.get("explicit_pik", 0) / total_pik_cnt) * 100
    pct_inferred_of_pik = (pivoted_cnt.get("inferred_pik", 0) / total_pik_cnt) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: % of total positions
    ax1.plot(quarters, pivoted_pct.get("inferred_pik", 0) * 100, marker="s", color="#ff7f0e", linewidth=2, label="Inferred / Reconstructed PIK (%)")
    ax1.plot(quarters, pivoted_pct.get("explicit_pik", 0) * 100, marker="o", color="#1f77b4", linewidth=2, label="Explicit Disclosed PIK (%)")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax1.set_title("A. PIK Provenance Share (% of Total Positions)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Filing Quarter", fontsize=10)
    ax1.set_ylabel("Share of All Positions (%)", fontsize=10)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Panel 2: Composition among PIK positions
    width = 0.55
    ax2.bar(quarters, pct_explicit_of_pik, width=width, label="Explicit Full Disclosure", color="#1f77b4", alpha=0.85)
    ax2.bar(quarters, pct_inferred_of_pik, bottom=pct_explicit_of_pik, width=width, label="Inferred / Reconstructed Tag", color="#ff7f0e", alpha=0.85)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax2.set_title("B. Composition Among Positive PIK Loans (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Filing Quarter", fontsize=10)
    ax2.set_ylabel("Composition (% of PIK Volume)", fontsize=10)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_pik_provenance_dollar_volume(
    prov_df: pl.DataFrame,
    quarter_col: str = "cal_q",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Two-panel time series displaying dollar volume by explicit vs inferred PIK:
      - Left: Fair Value Volume ($ Billions)
      - Right: Par Amount Volume ($ Billions)
    """
    pdf = prov_df.to_pandas()
    fv_piv = pdf.pivot(index=quarter_col, columns="pik_provenance", values="total_fv").fillna(0) / 1e9
    par_piv = pdf.pivot(index=quarter_col, columns="pik_provenance", values="total_par").fillna(0) / 1e9

    quarters = [str(q) for q in fv_piv.index]
    width = 0.45
    x = np.arange(len(quarters))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Fair Value Volume
    ax1.bar(x - width/2, fv_piv.get("explicit_pik", 0), width=width, label="Explicit Disclosed", color="#1f77b4", alpha=0.85)
    ax1.bar(x + width/2, fv_piv.get("inferred_pik", 0), width=width, label="Inferred / Reconstructed", color="#ff7f0e", alpha=0.85)
    ax1.set_title("A. PIK Fair Value Volume ($ Billions)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Filing Quarter", fontsize=10)
    ax1.set_ylabel("Fair Value ($B)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(quarters, rotation=45)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(loc="upper left")

    # Panel 2: Par Volume
    ax2.bar(x - width/2, par_piv.get("explicit_pik", 0), width=width, label="Explicit Disclosed", color="#1f77b4", alpha=0.85)
    ax2.bar(x + width/2, par_piv.get("inferred_pik", 0), width=width, label="Inferred / Reconstructed", color="#ff7f0e", alpha=0.85)
    ax2.set_title("B. PIK Par Value Volume ($ Billions)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Filing Quarter", fontsize=10)
    ax2.set_ylabel("Par Value ($B)", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(quarters, rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_bdc_idiosyncratic_tagging(
    cik_summary_df: pl.DataFrame,
    top_n: int = 12,
    figsize: tuple = (12, 6.5),
    save_path: Optional[str] = None,
) -> None:
    """
    Horizontal bar chart flagging the top BDCs with high volumes of idiosyncratic /
    inferred PIK tagging, mapping CIKs to recognizable market fund names.
    """
    pdf = (
        cik_summary_df.to_pandas()
        .sort_values("n_inferred_pik", ascending=False)
        .head(top_n)
        .sort_values("n_inferred_pik", ascending=True)
    )

    # Attach readable names
    labels = []
    for cik in pdf["cik"]:
        try:
            c_int = int(cik)
            name = CIK_NAME_MAP.get(c_int, f"CIK {cik}")
        except (ValueError, TypeError):
            name = str(cik)
        labels.append(name)

    y = np.arange(len(labels))
    height = 0.55

    fig, ax = plt.subplots(figsize=figsize)
    b1 = ax.barh(y, pdf["n_explicit_pik"], height=height, label="Explicit Full Tagging", color="#1f77b4", alpha=0.85)
    b2 = ax.barh(y, pdf["n_inferred_pik"], left=pdf["n_explicit_pik"], height=height, label="Idiosyncratic / Inferred Tagging", color="#d62728", alpha=0.85)

    # Add percentage label next to each bar
    for idx, (exp, inf, tot) in enumerate(zip(pdf["n_explicit_pik"], pdf["n_inferred_pik"], pdf["total_positions"])):
        pct_inf = (inf / (exp + inf) * 100) if (exp + inf) > 0 else 0
        ax.text(exp + inf + 35, idx, f"{inf:,} ({pct_inf:.1f}% inferred)", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Number of PIK Loan Positions", fontsize=10)
    ax.set_title("Top BDCs by Idiosyncratic / Inferred PIK Tagging Volume", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(pdf["n_explicit_pik"] + pdf["n_inferred_pik"]) * 1.35)
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_pik_distress_vs_contractual_share(
    df: pl.DataFrame,
    quarter_col: str = "cal_q",
    transition_col: str = "pik_transition_state",
    par_col: str = "PAR",
    final_pik_col: str = "PIK_Final",
    fv_col: str = "FV",
    figsize: tuple = (13, 5.5),
    save_path: Optional[str] = None,
) -> None:
    """
    Two-panel chart showing the proportion of PIK payments that are distress amendments
    as opposed to deals originally structured with PIK:
      - Left: Stacked bar chart of quarterly PIK payments ($ Millions) by transition state:
              Contractual (Inception), Seasoned Continuation, and Distress Amendment.
      - Right: Time series of distress amendment proportion (% of PIK payments, % of PIK count, % of PIK FV).
    """
    if transition_col not in df.columns:
        df = classify_loan_pik_transitions(df, quarter_col=quarter_col, final_pik_col=final_pik_col, par_col=par_col)

    # Compute quarterly PIK payment flow: PAR * PIK_Final / 4.0 ($ Millions)
    df_pik = (
        df.filter(pl.col(transition_col).is_in(["distress_amendment_pik", "seasoned_pik_continuation", "contractual_pik"]))
        .with_columns([
            (pl.col(par_col).fill_null(0) * pl.col(final_pik_col).fill_null(0) / 4.0 / 1e6).alias("pik_payment_m"),
            (pl.col(fv_col).fill_null(0) / 1e6).alias("fv_m"),
        ])
    )

    agg = (
        df_pik.group_by([quarter_col, transition_col])
        .agg([
            pl.col("pik_payment_m").sum().alias("payment_m"),
            pl.col("fv_m").sum().alias("fv_m"),
            pl.len().alias("count"),
        ])
        .sort([quarter_col, transition_col])
    )

    pdf = agg.to_pandas()
    pay_piv = pdf.pivot(index=quarter_col, columns=transition_col, values="payment_m").fillna(0)
    cnt_piv = pdf.pivot(index=quarter_col, columns=transition_col, values="count").fillna(0)
    fv_piv = pdf.pivot(index=quarter_col, columns=transition_col, values="fv_m").fillna(0)

    # Exclude 2023Q1 if it has 0 seasoned positions (only inception)
    quarters = [str(q) for q in pay_piv.index if str(q) != "2023Q1"]
    pay_piv = pay_piv.loc[quarters]
    cnt_piv = cnt_piv.loc[quarters]
    fv_piv = fv_piv.loc[quarters]

    c_incept = pay_piv.get("contractual_pik", 0)
    c_cont = pay_piv.get("seasoned_pik_continuation", 0)
    d_amend = pay_piv.get("distress_amendment_pik", 0)
    tot_pay = c_incept + c_cont + d_amend

    tot_cnt = cnt_piv.get("contractual_pik", 0) + cnt_piv.get("seasoned_pik_continuation", 0) + cnt_piv.get("distress_amendment_pik", 0)
    tot_fv = fv_piv.get("contractual_pik", 0) + fv_piv.get("seasoned_pik_continuation", 0) + fv_piv.get("distress_amendment_pik", 0)

    pct_distress_pay = (d_amend / tot_pay.replace(0, np.nan)) * 100
    pct_distress_cnt = (cnt_piv.get("distress_amendment_pik", 0) / tot_cnt.replace(0, np.nan)) * 100
    pct_distress_fv = (fv_piv.get("distress_amendment_pik", 0) / tot_fv.replace(0, np.nan)) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    x = np.arange(len(quarters))
    bar_width = 0.65

    # Panel 1: Stacked Bar Chart of Payments
    ax1.bar(x, c_incept, width=bar_width, label="Contractual (New Inception)", color="#1f77b4", alpha=0.85)
    ax1.bar(x, c_cont, bottom=c_incept, width=bar_width, label="Contractual (Seasoned Continuation)", color="#4fa8d8", alpha=0.85)
    ax1.bar(x, d_amend, bottom=c_incept + c_cont, width=bar_width, label="Distress Amendment PIK", color="#d62728", alpha=0.90)

    ax1.set_title("A. Quarterly PIK Payments ($M) by Deal Structure", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Filing Quarter", fontsize=10)
    ax1.set_ylabel("Quarterly PIK Interest Accrued ($M)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(quarters, rotation=45)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(loc="upper left", framealpha=0.9)

    # Panel 2: Proportion Line Series
    ax2.plot(x, pct_distress_pay, marker="s", color="#d62728", linewidth=2.2, label="Distress Share of PIK Payments (%)")
    ax2.plot(x, pct_distress_cnt, marker="o", color="#2ca02c", linewidth=2.0, linestyle="--", label="Distress Share of PIK Loan Count (%)")
    ax2.plot(x, pct_distress_fv, marker="^", color="#ff7f0e", linewidth=1.8, linestyle=":", label="Distress Share of PIK Fair Value (%)")

    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax2.set_title("B. Distress Amendment Proportion of PIK Market", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Filing Quarter", fontsize=10)
    ax2.set_ylabel("Distress Amendment Share (%)", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(quarters, rotation=45)
    ax2.set_ylim(0, max(pct_distress_pay.max(), pct_distress_cnt.max(), pct_distress_fv.max()) * 1.25)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()



