import polars as pl
import pandas as pd
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
def load_and_prepare_investment_data(
    csv_path: str,
    infer_schema_length: int = 100_000
) -> pl.DataFrame:
    """
    Load investment data from CSV, filter valid fair value rows,
    and cast selected normalized columns to Float64 with cleaner aliases.
    """
    df = pl.read_csv(csv_path, infer_schema_length=infer_schema_length)
    df = (
        df.filter(pl.col("InvestmentOwnedAtFairValue_normalized").is_not_null())
          .with_columns([
              pl.col("InvestmentOwnedAtFairValue_normalized")
                .cast(pl.Float64)
                .alias("FV"),
              pl.col("InvestmentOwnedAtCost_normalized")
                .cast(pl.Float64)
                .alias("COST"),
              pl.col("InvestmentOwnedBalancePrincipalAmount_normalized")
                .cast(pl.Float64)
                .alias("PAR"),
              pl.col("InvestmentInterestRatePaidInCash_normalized")
                .cast(pl.Float64)
                .alias("rate_cash"),
              pl.col("InvestmentInterestRate_normalized")
                .cast(pl.Float64)
                .alias("interest_rate"),
              pl.col("InvestmentInterestRatePaidInKind_normalized")
                .cast(pl.Float64)
                .alias("rate_pik"),
          ])
    )
    return df

def missing_percentage_summary(
    df: pl.DataFrame,
    cols: list[str],
    as_pandas: bool = True
):
    """
    Compute percentage of missing values per column.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe
    cols : list[str]
        Columns to evaluate
    as_pandas : bool, default True
        If True, return pandas DataFrame; otherwise return Polars DataFrame
    """
    n = df.height

    summary = df.select([
        (pl.col(c).is_null().sum() * 100 / n).alias(f"{c}_missing")
        for c in cols
    ])

    return summary.to_pandas() if as_pandas else summary

def interest_rate_null_combinations(
    df: pl.DataFrame,
    interest_cols: list[str],
    as_pandas: bool = True
):
    """
    Compute row counts for all combinations of null / non-null
    across interest rate columns.
    """
    null_flag_cols = [f"{c}_isnull" for c in interest_cols]

    result = (
        df.with_columns([
            pl.col(c).is_null().alias(f"{c}_isnull")
            for c in interest_cols
        ])
        .group_by(null_flag_cols)
        .agg(pl.count().alias("rows"))
        .sort(null_flag_cols)
    )

    return result.to_pandas() if as_pandas else result

def compute_final_interest_rates(
    df: pl.DataFrame,
    interest_cols: list[str] = ["rate_cash", "interest_rate", "rate_pik"]
) -> pl.DataFrame:
    """
    Compute final PIC, PIK, and total interest rate (IR_Final)
    based on available cash, PIK, and total interest rate columns.
    Rows with all three missing are dropped.
    """

    df = df.with_columns(
        missing_count=pl.sum_horizontal([pl.col(c).is_null() for c in interest_cols])
    )

    # drop rows where all interest fields are missing
    df = df.filter(pl.col("missing_count") < 3)

    df = df.with_columns([

        # PIC_Final
        pl.when(pl.col("rate_cash").is_not_null())
            .then(pl.col("rate_cash"))

        .when(pl.col("missing_count") == 1)
            .then(pl.col("interest_rate") - pl.col("rate_pik"))

        .when(pl.col("interest_rate").is_not_null() & pl.col("rate_pik").is_null())
            .then(pl.col("interest_rate"))

        .when(pl.col("rate_pik").is_not_null() & pl.col("interest_rate").is_null())
            .then(0)

        .otherwise(0)
        .alias("PIC_Final"),

        # PIK_Final
        pl.when(pl.col("rate_pik").is_not_null())
            .then(pl.col("rate_pik"))

        .when(pl.col("missing_count") == 1)
            .then(pl.col("interest_rate") - pl.col("rate_cash"))

        .when(pl.col("rate_cash").is_not_null() & pl.col("interest_rate").is_null())
            .then(0)

        .when(pl.col("interest_rate").is_not_null() & pl.col("rate_cash").is_null())
            .then(0)

        .otherwise(0)
        .alias("PIK_Final"),
    ])

    df = df.with_columns(
        (pl.col("PIC_Final") + pl.col("PIK_Final")).alias("IR_Final")
    ).drop("missing_count")

    return df

def qsort_expr():
    y = pl.col("cal_q").str.slice(0, 4).cast(pl.Int32)
    q = pl.col("cal_q").str.slice(-1).cast(pl.Int32)
    return (y * 4 + q).alias("qsort")
def safe_div(x, y):
    return pl.when(y.is_null() | (y == 0)).then(0.0).otherwise(x / y)

def compute_position_level_flows(
    df: pl.DataFrame,
    qsort_expr: Callable[[], pl.Expr],
    safe_div: Callable[[pl.Expr, pl.Expr], pl.Expr],
    group_cols: list[str] = ["cik", "investment_identifier"]
) -> pl.DataFrame:
    """
    Sort positions by quarter, compute lags, flows, and position-level returns.
    """
    df = (
        df.with_columns(qsort_expr())
          .sort(group_cols + ["qsort"])
    )
    df = (
        df.with_columns([
            pl.col("FV").shift(1).over(group_cols).alias("FV_prev"),
            pl.col("COST").shift(1).over(group_cols).alias("COST_prev"),
        ])
        .with_columns([
            (pl.col("FV") - pl.col("FV_prev")).alias("dFV"),
            (pl.col("COST") - pl.col("COST_prev")).alias("dCOST"),
            (pl.col("PIC_Final").fill_null(0) * pl.col("PAR").fill_null(0) / 4).alias("cash_income"),
            (pl.col("PIK_Final").fill_null(0) * pl.col("PAR").fill_null(0) / 4).alias("pik_income"),
        ])
        .with_columns(
            safe_div(
                pl.col("dFV")
                - pl.col("dCOST")
                + pl.col("cash_income")
                + pl.col("pik_income"),
                pl.col("FV_prev")
            ).alias("ret_pos_flow")
        )
    )
    return df

def quarter_count_distribution(
    df: pl.DataFrame,
    group_cols: list[str] = ["cik", "investment_identifier"],
    quarter_col: str = "qsort"
) -> pl.DataFrame:
    """
    Compute distribution of number of unique quarters per (cik, investment_identifier).
    """

    summary = (
        df.group_by(group_cols)
          .agg(pl.col(quarter_col).n_unique().alias("num_quarters"))
    )

    num_unique_pairs = summary.height

    distribution = (
        summary["num_quarters"]
          .value_counts()
          .sort("num_quarters")
          .with_columns(
              (pl.col("count") / num_unique_pairs).alias("proportion")
          )
    )

    return distribution

def quarterly_cik_entry_exit_aum(
    df: pl.DataFrame,
    fv_col: str = "FV",
    cik_col: str = "cik",
    cal_q_col: str = "cal_q",
    qsort_col: str = "qsort",
    scale: float = 1e6
) -> pl.DataFrame:
    """
    Compute quarterly AUM entering and exiting at the CIK level.
    """

    # AUM per CIK-quarter
    aum_cik_q = (
        df.group_by([cik_col, cal_q_col, qsort_col])
          .agg(pl.sum(fv_col).alias("aum"))
          .with_columns((pl.col("aum") / scale).alias("aum_m"))
          .sort([cik_col, qsort_col])
    )

    # Identify entry / exit quarters
    aum_cik_q = aum_cik_q.with_columns([
        pl.col(qsort_col).shift(1).over(cik_col).alias("qsort_prev"),
        pl.col(cal_q_col).shift(1).over(cik_col).alias("cal_q_prev"),
        pl.col(qsort_col).shift(-1).over(cik_col).alias("qsort_next"),
        pl.col(cal_q_col).shift(-1).over(cik_col).alias("cal_q_next"),
    ])

    # Entries: first appearance of a CIK
    cik_entries = (
        aum_cik_q
        .filter(pl.col("qsort_prev").is_null())
        .select([cal_q_col, cik_col, "aum_m"])
    )

    # Exits: last appearance of a CIK
    cik_exits = (
        aum_cik_q
        .filter(pl.col("qsort_next").is_null())
        .select([cal_q_col, cik_col, "aum_m"])
    )

    # Aggregate entries / exits by quarter
    aum_in_cik = (
        cik_entries.group_by(cal_q_col)
                   .agg(
                       pl.sum("aum_m").alias("aum_in_m_cik"),
                       pl.col(cik_col).n_unique().alias("unique_cik_in")
                   )
    )

    aum_out_cik = (
        cik_exits.group_by(cal_q_col)
                 .agg(
                     pl.sum("aum_m").alias("aum_out_m_cik"),
                     pl.col(cik_col).n_unique().alias("unique_cik_out")
                 )
    )

    # Final quarterly summary
    quarterly_summary = (
        aum_in_cik
        .join(aum_out_cik, on=cal_q_col, how="left")
        .fill_null(0)
        .sort(cal_q_col)
        .with_columns(pl.col(pl.NUMERIC_DTYPES).round(2))
    )

    return quarterly_summary

def compute_market_weights(
    df: pl.DataFrame,
    safe_div: Callable[[pl.Expr, pl.Expr], pl.Expr],
    ret_col: str = "ret_pos_flow",
    fv_prev_col: str = "FV_prev",
    quarter_col: str = "cal_q"
) -> pl.DataFrame:
    """
    Compute validity flags, quarterly FV_prev sums, and market weights.
    """

    df = df.with_columns(
        (
            pl.col(ret_col).is_finite()
            & pl.col(fv_prev_col).is_finite()
            & (pl.col(fv_prev_col) > 0)
        ).alias("is_valid")
    )

    df = df.with_columns(
        pl.when(pl.col("is_valid"))
          .then(pl.col(fv_prev_col))
          .otherwise(0.0)
          .sum()
          .over(quarter_col)
          .alias("FV_prev_sum_q")
    )

    df = df.with_columns(
        safe_div(
            pl.when(pl.col("is_valid"))
              .then(pl.col(fv_prev_col))
              .otherwise(0.0),
            pl.col("FV_prev_sum_q")
        ).alias("w_mkt")
    )

    return df


def aggregate_return_decomposition(
    df: pl.DataFrame,
    safe_div: Callable[[pl.Expr, pl.Expr], pl.Expr],
    quarter_col: str = "cal_q"
) -> pl.DataFrame:
    """
    Aggregate market-level quarterly return decomposition.
    """

    df_decomp = (
        df
        .filter(pl.col("w_mkt") > 0)
        .group_by(quarter_col)
        .agg([
            # weighted return contributions
            (pl.col("w_mkt") * safe_div(pl.col("dFV") - pl.col("dCOST"), pl.col("FV_prev")))
                .sum().alias("contrib_price"),

            (pl.col("w_mkt") * safe_div(pl.col("cash_income"), pl.col("FV_prev")))
                .sum().alias("contrib_cash"),

            (pl.col("w_mkt") * safe_div(pl.col("pik_income"), pl.col("FV_prev")))
                .sum().alias("contrib_pik"),

            # counts and averages
            pl.count().alias("n_investments"),
            pl.col("rate_cash").mean().alias("avg_rate_cash"),
            pl.col("rate_pik").mean().alias("avg_rate_pik"),
        ])
        .sort(quarter_col)
    )

    return df_decomp

def plot_quarterly_return_decomposition(
    df_decomp_pd,
    title: str,
    figsize=(14, 8)
):
    """
    Plot stacked quarterly return decomposition:
    price (±), cash interest, and PIK interest.
    """

    plt.figure(figsize=figsize)
    ax = plt.gca()

    quarters = df_decomp_pd["cal_q"][1:].values
    price = df_decomp_pd["contrib_price"][1:].values
    cash = df_decomp_pd["contrib_cash"][1:].values
    pik = df_decomp_pd["contrib_pik"][1:].values
    total = price + cash + pik

    for q, p, c, pi, t in zip(quarters, price, cash, pik, total):

        # --- Price return (positive / negative) ---
        if p < 0:
            ax.bar(q, p, color="red", alpha=0.7,
                   edgecolor="darkred", linewidth=1)
        else:
            ax.bar(q, p, color="green", alpha=0.7,
                   edgecolor="darkgreen", linewidth=1)

        if p != 0:
            ax.text(q, p / 2, f"{p*100:.2f}%",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold")

        # --- Cash interest ---
        ax.bar(q, c, bottom=0,
               color="grey", alpha=0.6,
               edgecolor="black", linewidth=1)

        if c != 0:
            ax.text(q, c / 2, f"{c*100:.2f}%",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold")

        # --- PIK interest ---
        ax.bar(q, pi, bottom=c,
               color="orange", alpha=0.6,
               edgecolor="black", linewidth=1)

        if pi != 0:
            ax.text(q, c + pi / 2, f"{pi*100:.2f}%",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold")

    # --- Legend ---
    legend_handles = [
        Patch(facecolor="green", edgecolor="darkgreen", label="Price Return (Positive)"),
        Patch(facecolor="red", edgecolor="darkred", label="Price Return (Negative)"),
        Patch(facecolor="grey", edgecolor="black", label="Cash Interest"),
        Patch(facecolor="orange", edgecolor="black", label="PIK Interest"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        fontsize=12,
        frameon=True
    )

    # --- Axes formatting ---
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(quarters)
    ax.set_xticklabels(quarters, rotation=45)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # --- Dynamic y-limits ---
    stacked_top = cash + pik
    all_values = np.concatenate([price, stacked_top, total])

    base_min, base_max = np.min(all_values), np.max(all_values)
    margin = max((base_max - base_min) * 0.03, 0.005)

    y_min = min(base_min - margin, -0.01)
    y_max = base_max + margin + 0.01

    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()

import polars as pl

def compute_flow_based_market_index(
    df: pl.DataFrame,
    quarter_col: str = "cal_q",
    weight_col: str = "w_mkt",
    return_col: str = "ret_pos_flow",
    index_name: str = "Flow-based (Market Investment-weighted)",
    clip_min: float = -100,
    clip_max: float = 1000,
    base_level: float = 100.0
) -> pl.DataFrame:
    """
    Compute a flow-based, market investment-weighted index.
    """

    index_df = (
        df
        .filter(pl.col(weight_col) > 0)
        .group_by(quarter_col)
        .agg(
            (pl.col(weight_col) * pl.col(return_col))
            .sum()
            .alias("IndexReturn")
        )
        .sort(quarter_col)
        .with_columns([
            pl.col("IndexReturn")
              .fill_nan(0)
              .fill_null(0)
              .clip(clip_min, clip_max),

            ((1 + pl.col("IndexReturn")).cum_prod() * base_level)
              .alias("IndexLevel"),

            pl.lit(index_name).alias("IndexName"),
        ])
    )

    return index_df

def plot_index_vs_cdli_returns(
    index_df,
    cdli_csv_path: str,
    quarter_col: str = "cal_q",
    index_return_col: str = "IndexReturn",
    cdli_col: str = "CDLI",
    title: str = "Sector Index Returns QoQ(%)",
    figsize=(10, 5),
    ylims=(1, 4),
    start_idx: int = 1
):
    """
    Plot QoQ returns of flow-based index vs CDLI benchmark.

    The index's first quarter has no prior-quarter FV to difference against,
    so its return is structurally undefined; any non-zero value there comes
    from same-quarter duplicate rows standing in for a prior period. start_idx
    drops that leading row, matching plot_quarterly_return_decomposition and
    compute_tracking_error.
    """

    # Load CDLI data
    cdli_returns = pd.read_csv(cdli_csv_path)
    cdli_returns[quarter_col] = cdli_returns["Quarter"].astype(str)

    # Ensure pandas
    if not isinstance(index_df, pd.DataFrame):
        index_df = index_df.to_pandas()

    index_df = index_df.iloc[start_idx:]
    # Align CDLI to the same quarters as the (now-truncated) index line so the
    # shared categorical x-axis stays chronologically ordered.
    cdli_returns = cdli_returns[cdli_returns[quarter_col].isin(index_df[quarter_col])]

    plt.figure(figsize=figsize)

    plt.plot(
        index_df[quarter_col],
        index_df[index_return_col] * 100,
        label="Index Returns"
    )

    plt.plot(
        cdli_returns[quarter_col],
        cdli_returns[cdli_col] * 100,
        label="CDLI Returns"
    )

    plt.legend()
    plt.ylim(*ylims)
    plt.xticks(rotation=45)
    plt.xlabel("Quarter")
    plt.ylabel("Returns QoQ (%)")
    plt.title(title)

    plt.tight_layout()
    plt.show()

def compute_annualized_return_and_vol(
    returns,
    periods_per_year: int = 4,
    num_years: float = 10,
    ddof: int = 1
):
    """
    Compute annualized return and annualized volatility.

    Parameters
    ----------
    returns : array-like
        Periodic returns (e.g., quarterly returns in decimal form)
    periods_per_year : int
        Number of return periods per year (4 for quarterly)
    num_years : float
        Total number of years covered by the returns
    ddof : int
        Degrees of freedom for volatility calculation

    Returns
    -------
    annual_return : float
        Geometric annualized return
    annual_vol : float
        Annualized volatility
    """

    r = np.asarray(returns)

    # Geometric annualized return
    annual_return = np.prod(1 + r) ** (1 / num_years) - 1

    # Annualized volatility
    vol_q = np.std(r, ddof=ddof)
    annual_vol = vol_q * np.sqrt(periods_per_year)

    return annual_return, annual_vol

import pandas as pd

def load_quarterly_rates(
    excel_path: str,
    rate_cols=("EFFR", "SOFR")
) -> pd.DataFrame:
    """
    Load daily rate data from Excel and compute quarterly averages.
    """

    values = pd.read_excel(excel_path).iloc[:, :3]

    values = values.pivot(
        index=values.columns[0],
        columns=values.columns[1],
        values=values.columns[2]
    )

    values.index = pd.to_datetime(values.index)
    values.sort_index(inplace=True)

    quarterly_rates = values[list(rate_cols)].resample("Q").mean()
    quarterly_rates["cal_q"] = quarterly_rates.index.to_period("Q").astype(str)

    return quarterly_rates.reset_index(drop=True)


def build_quarterly_comparison_df(
    quarterly_rates: pd.DataFrame,
    index_mkt_flow,
    cdli_csv_path: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge index returns, CDLI returns, and quarterly SOFR / EFFR averages.
    """
    cdli_returns = pd.read_csv(cdli_csv_path)
    cdli_returns["cal_q"] = cdli_returns["Quarter"].astype(str)
    if not isinstance(index_mkt_flow, pd.DataFrame):
        index_mkt_flow = index_mkt_flow.to_pandas()

    cdli_returns = cdli_returns.rename(columns={"CDLI": "CDLI_Return"})

    plot_df = (
        pd.DataFrame({
            "Quarter": quarterly_rates["cal_q"],
            "EFFR_avg": quarterly_rates["EFFR"] / 400,
            "SOFR_avg": quarterly_rates["SOFR"] / 400,
        })
        .merge(
            index_mkt_flow[["cal_q", "IndexReturn"]],
            left_on="Quarter", right_on="cal_q", how="left"
        )
        .merge(
            cdli_returns[["cal_q", "CDLI_Return"]],
            left_on="Quarter", right_on="cal_q", how="left"
        )
        .drop(columns=["cal_q_x", "cal_q_y"], errors="ignore")
    )

    return plot_df


import numpy as np

def compute_tracking_error(
    plot_df: pd.DataFrame,
    start_idx: int = 2,
    periods_per_year: int = 4
):
    """
    Compute correlation and annualized tracking error.
    """

    corr = plot_df.iloc[start_idx:][["IndexReturn", "CDLI_Return"]].corr()

    diff = (
        plot_df["IndexReturn"].iloc[start_idx:]
        - plot_df["CDLI_Return"].iloc[start_idx:]
    )

    te_q = diff.std()
    te_a = te_q * np.sqrt(periods_per_year)

    return corr, te_q, te_a


import matplotlib.pyplot as plt

def plot_index_cdli_sofr(
    plot_df: pd.DataFrame,
    title: str = "Index Returns Comparison with SOFR Quarterly Averages",
    ylim=(0.5, 4),
    figsize=(10, 5),
    start_idx: int = 2
):
    """
    Plot index returns, CDLI returns, and SOFR quarterly averages.

    start_idx drops the leading rates-only row and the structurally
    undefined first index quarter, matching compute_tracking_error.
    """

    plt.figure(figsize=figsize)

    plt.plot(
        plot_df["Quarter"].iloc[start_idx:],
        plot_df["IndexReturn"].iloc[start_idx:] * 100,
        label="Index Return"
    )

    plt.plot(
        plot_df["Quarter"].iloc[start_idx:],
        plot_df["CDLI_Return"].iloc[start_idx:] * 100,
        label="CDLI Return"
    )

    plt.plot(
        plot_df["Quarter"].iloc[start_idx:],
        plot_df["SOFR_avg"].iloc[start_idx:] * 100,
        label="SOFR Average",
        linestyle="--"
    )

    plt.legend()
    plt.ylim(*ylim)
    plt.xticks(rotation=45)
    plt.xlabel("Quarter")
    plt.ylabel("Returns / Rates (%)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
