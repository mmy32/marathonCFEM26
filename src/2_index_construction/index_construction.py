"""Flow-based BDC loan index (slide 8): positions matched across quarters on borrower + seniority, returns
net of draws and paydowns, market-value weighted, compared with the CDLI and CDLI-S (correlation,
tracking error). Run as a script: index_vs_cdli_panels.png and index_vs_cdli_stats.csv in
outputs/index_construction_plots/. See docs/4_index_construction.md."""
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import polars as pl

# FINAL_DATASET column -> short name, cast to Float64
NUMERIC_ALIASES = {
    "InvestmentOwnedAtFairValue_normalized": "FV",
    "InvestmentOwnedAtCost_normalized": "COST",
    "InvestmentOwnedBalancePrincipalAmount_normalized": "PAR",
    "InvestmentInterestRatePaidInCash_normalized": "rate_cash",
    "InvestmentInterestRate_normalized": "interest_rate",
    "InvestmentInterestRatePaidInKind_normalized": "rate_pik",
}


def load_and_prepare_investment_data(csv_path: str, infer_schema_length: int = 100_000) -> pl.DataFrame:
    """Rows with a fair value, plus the NUMERIC_ALIASES columns as Float64 under short names."""
    return (
        pl.read_csv(csv_path, infer_schema_length=infer_schema_length)
        .filter(pl.col("InvestmentOwnedAtFairValue_normalized").is_not_null())
        .with_columns([pl.col(src).cast(pl.Float64).alias(dst) for src, dst in NUMERIC_ALIASES.items()])
    )


def compute_final_interest_rates(
    df: pl.DataFrame,
    interest_cols: list[str] = ["rate_cash", "interest_rate", "rate_pik"]
) -> pl.DataFrame:
    """
    PIC_Final (cash), PIK_Final and IR_Final = PIC + PIK from the cash, total and PIK rates. A rate
    that is reported is used; with one of the three missing it is derived from the other two
    (cash = total - PIK, PIK = total - cash); a total with no PIK is all cash; anything else is 0.
    Rows with all three missing are dropped.
    """
    missing = pl.sum_horizontal([pl.col(c).is_null() for c in interest_cols])
    cash, total, pik = pl.col("rate_cash"), pl.col("interest_rate"), pl.col("rate_pik")
    return (
        df.with_columns(missing_count=missing)
        .filter(pl.col("missing_count") < 3)
        .with_columns([
            pl.when(cash.is_not_null()).then(cash)
            .when(pl.col("missing_count") == 1).then(total - pik)
            .when(total.is_not_null() & pik.is_null()).then(total)
            .otherwise(0)
            .alias("PIC_Final"),
            pl.when(pik.is_not_null()).then(pik)
            .when(pl.col("missing_count") == 1).then(total - cash)
            .otherwise(0)
            .alias("PIK_Final"),
        ])
        .with_columns((pl.col("PIC_Final") + pl.col("PIK_Final")).alias("IR_Final"))
        .drop("missing_count")
    )


def qsort_expr():
    y = pl.col("cal_q").str.slice(0, 4).cast(pl.Int32)
    q = pl.col("cal_q").str.slice(-1).cast(pl.Int32)
    return (y * 4 + q).alias("qsort")


def safe_div(x, y):
    return pl.when(y.is_null() | (y == 0)).then(0.0).otherwise(x / y)


POSITION_GROUP_COLS = ["cik", "position_key"]
SUM_COLS = ["FV", "COST", "PAR"]
RATE_COLS = ["PIC_Final", "PIK_Final", "IR_Final", "rate_cash", "interest_rate", "rate_pik"]


def add_position_key(
    df: pl.DataFrame,
    borrower_col: str = "borrower_canonical",
    seniority_col: str = "instrument_seniority",
    identifier_col: str = "investment_identifier",
) -> pl.DataFrame:
    """
    Add `borrower_key` and `position_key`, the keys positions are matched on.

    - borrower_key: the validated borrower name. Rows with no borrower name
      (a few hundred) fall back to their investment_identifier text,
      prefixed "ID: " so it can never equal a borrower name.
    - position_key: borrower_key + " | " + instrument_seniority, so a BDC's
      first-lien loan and its equity in the same borrower stay separate.

    investment_identifier is not used for matching: BDCs reword it between
    filings (e.g. most BDCs switched from "Name, One stop 1" to
    "Name | One stop 1 | Non-Affiliated Issuer" in their FY2025 10-Ks), and
    some embed the quarter's rate or % of net assets in it.
    """
    if borrower_col not in df.columns:
        raise KeyError(f"{borrower_col!r} is required to match positions; load FINAL_DATASET.csv")
    b = pl.col(borrower_col).cast(pl.Utf8).str.strip_chars()
    return (
        df.with_columns(
            pl.when(b.is_null() | (b == ""))
            .then(pl.lit("ID: ") + pl.col(identifier_col).cast(pl.Utf8))
            .otherwise(b)
            .alias("borrower_key")
        )
        .with_columns(
            (pl.col("borrower_key") + pl.lit(" | ")
             + pl.col(seniority_col).cast(pl.Utf8).fill_null("OTHER_UNKNOWN")).alias("position_key")
        )
    )


def aggregate_positions(
    df: pl.DataFrame,
    group_cols: list[str] = POSITION_GROUP_COLS,
    quarter_col: str = "cal_q",
) -> pl.DataFrame:
    """
    One row per (group_cols, quarter): a BDC's rows for the same borrower and
    seniority in a quarter (several tranches, delayed-draw pieces, ...) are
    combined so they can be matched to the previous quarter as one position.

    - FV, COST, PAR: summed (null only if every row is null).
    - Rate columns: weighted by PAR over the rows that report the rate, so
      rate x PAR (the interest income) equals the rows' total exactly. With
      no PAR, weighted by |FV|; with neither, a simple mean.
    - Every other column: the value on the largest-FV row. `sector` is the
      same for all rows of a borrower.
    - n_rows: how many rows were combined.
    """
    if "position_key" in group_cols and "position_key" not in df.columns:
        df = add_position_key(df)
    keys = group_cols + [quarter_col]
    sums = [c for c in SUM_COLS if c in df.columns]
    rates = [c for c in RATE_COLS if c in df.columns]

    par = pl.col("PAR").fill_null(0) if "PAR" in df.columns else pl.lit(0.0)
    fv = pl.col("FV").abs().fill_null(0)

    def weighted(c: str) -> pl.Expr:
        has = pl.col(c).is_not_null()
        par_w = pl.when(has).then(par).otherwise(0.0)
        fv_w = pl.when(has).then(fv).otherwise(0.0)
        return (
            pl.when(par_w.sum() > 0).then((pl.col(c).fill_null(0) * par_w).sum() / par_w.sum())
            .when(fv_w.sum() > 0).then((pl.col(c).fill_null(0) * fv_w).sum() / fv_w.sum())
            .otherwise(pl.col(c).mean())
            .alias(c)
        )

    others = [c for c in df.columns if c not in set(keys) | set(sums) | set(rates)]
    return (
        df.sort(keys + ["FV"], descending=[False] * len(keys) + [True], nulls_last=True)
        .group_by(keys, maintain_order=True)
        .agg(
            [pl.len().alias("n_rows")]
            + [pl.when(pl.col(c).is_not_null().any()).then(pl.col(c).sum()).otherwise(None).alias(c) for c in sums]
            + [weighted(c) for c in rates]
            + [pl.col(c).first() for c in others]
        )
    )


def compute_position_level_flows(
    df: pl.DataFrame,
    qsort_expr: Callable[[], pl.Expr],
    safe_div: Callable[[pl.Expr, pl.Expr], pl.Expr],
    group_cols: list[str] = POSITION_GROUP_COLS,
) -> pl.DataFrame:
    """
    Combine rows into positions, link each to its previous quarter, and
    compute lags, flows, and position-level returns.

    A position is a BDC's holdings in one borrower at one seniority
    (`position_key`, see add_position_key) -- matched on the validated
    borrower name, not on investment_identifier text, which BDCs reword
    between filings. aggregate_positions() first combines a BDC's rows for
    the same position in a quarter, so every (group_cols, quarter) is one
    row and the lag below is deterministic. The lag is the position's
    previous observation, normally the previous quarter.
    """
    df = (
        aggregate_positions(df, group_cols)
        .with_columns(qsort_expr())
        .sort(group_cols + ["qsort"])
    )
    df = (
        df.with_columns([
            pl.col("FV").shift(1).over(group_cols).alias("FV_prev"),
            pl.col("COST").shift(1).over(group_cols).alias("COST_prev"),
            pl.col("PAR").shift(1).over(group_cols).alias("PAR_prev"),
            pl.col("PIC_Final").shift(1).over(group_cols).alias("PIC_Final_prev"),
            pl.col("PIK_Final").shift(1).over(group_cols).alias("PIK_Final_prev"),
        ])
        .with_columns([
            (pl.col("FV") - pl.col("FV_prev")).alias("dFV"),
            (pl.col("COST") - pl.col("COST_prev")).alias("dCOST"),
            (pl.col("PIC_Final_prev").fill_null(0) * pl.col("PAR_prev").fill_null(0) / 4).alias("cash_income"),
            (pl.col("PIK_Final_prev").fill_null(0) * pl.col("PAR_prev").fill_null(0) / 4).alias("pik_income"),
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

    is_valid_expr = (
        pl.col(ret_col).is_finite()
        & pl.col(fv_prev_col).is_finite()
        & (pl.col(fv_prev_col) > 0)
    ).fill_null(False)

    valid_fv_expr = (
        pl.when(pl.col("is_valid"))
          .then(pl.col(fv_prev_col))
          .otherwise(0.0)
    )

    return (
        df
        .with_columns(is_valid_expr.alias("is_valid"))
        .with_columns(
            valid_fv_expr.sum().over(quarter_col).alias("FV_prev_sum_q")
        )
        .with_columns(
            safe_div(valid_fv_expr, pl.col("FV_prev_sum_q")).alias("w_mkt")  # market weight of position in quarter
        )
    )


def aggregate_return_decomposition(
    df: pl.DataFrame,
    safe_div: Callable[[pl.Expr, pl.Expr], pl.Expr],
    quarter_col: str = "cal_q"
) -> pl.DataFrame:
    """
    Aggregate market-level quarterly return decomposition.

    Returns one row per quarter in `df`, including quarters with no position
    linked to a prior quarter (the panel's first quarter): those get zero
    contributions, so positional `start_idx` trimming downstream always drops
    exactly the first panel quarter.
    """

    def calc_contrib(numerator: pl.Expr, name: str) -> pl.Expr:
        return (pl.col("w_mkt") * safe_div(numerator, pl.col("FV_prev"))).sum().alias(name)

    def calc_weighted_avg(rate_col: str, name: str) -> pl.Expr:
        # PIC_Final / PIK_Final (unlike raw rate_cash / rate_pik) are always
        # non-null after compute_final_interest_rates, and are the same rate
        # series cash_income / pik_income (and hence contrib_cash / contrib_pik
        # above) are built from -- so no null-renormalization is needed here.
        return (pl.col(rate_col) * pl.col("w_mkt")).sum().alias(name)

    df_decomp = (
        df
        .filter(pl.col("w_mkt") > 0)
        .group_by(quarter_col, maintain_order=True)
        .agg(
            # weighted return contributions
            calc_contrib(pl.col("dFV") - pl.col("dCOST"), "contrib_price"),
            calc_contrib(pl.col("cash_income"), "contrib_cash"),
            calc_contrib(pl.col("pik_income"), "contrib_pik"),

            # counts and value-weighted average rates
            pl.len().alias("n_investments"),
            calc_weighted_avg("PIC_Final", "avg_rate_cash"),
            calc_weighted_avg("PIK_Final", "avg_rate_pik"),
        )
    )
    all_q = df.select(quarter_col).unique(maintain_order=True)
    df_decomp = (
        all_q.join(df_decomp, on=quarter_col, how="left")
        .with_columns(pl.exclude(quarter_col).fill_null(0))
        .sort(quarter_col)
    )

    return df_decomp


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

    Returns one row per quarter in `df`. A quarter with no position linked to
    a prior quarter (the panel's first quarter) gets a zero return, so the
    level starts at base_level and positional `start_idx` trimming downstream
    always drops exactly the first panel quarter.
    """

    returns = (
        df
        .filter(pl.col(weight_col) > 0)
        .group_by(quarter_col, maintain_order=True)
        .agg(
            (pl.col(weight_col) * pl.col(return_col))
            .sum()
            .alias("IndexReturn")
        )
    )
    index_df = (
        df.select(quarter_col).unique(maintain_order=True)
        .join(returns, on=quarter_col, how="left")
        .sort(quarter_col)
        .with_columns(
            pl.col("IndexReturn")
              .fill_nan(0.0)
              .fill_null(0.0)
              .clip(clip_min, clip_max)
        )
        .with_columns(
            ((1 + pl.col("IndexReturn")).cum_prod() * base_level)
              .alias("IndexLevel"),

            pl.lit(index_name).alias("IndexName"),
        )
    )

    return index_df


def _ensure_pandas(df):
    """Coerce a Polars or pandas frame to pandas."""
    return df.to_pandas() if not isinstance(df, pd.DataFrame) else df


def load_cdli_returns(cdli_csv_path: str, quarter_col: str = "cal_q") -> pd.DataFrame:
    """
    Load the CDLI benchmark CSV and rename its Quarter column to quarter_col.
    """
    cdli_returns = pd.read_csv(cdli_csv_path)
    cdli_returns[quarter_col] = cdli_returns["Quarter"].astype(str)
    return cdli_returns


# Annualized volatility on raw marks understates true risk: see unsmooth_ar1_returns /
# compute_unsmoothed_return_and_vol below, and compute_max_drawdown as a smoothing-immune alternative.
def compute_annualized_return_and_vol(
    returns,
    periods_per_year: int = 4,
    num_years: Optional[float] = None,
    ddof: int = 1,
    start_idx: int = 1
):
    """
    Compute annualized return and annualized volatility.

    Parameters
    ----------
    returns : array-like
        Periodic returns (e.g., quarterly returns in decimal form)
    periods_per_year : int
        Number of return periods per year (4 for quarterly)
    num_years : float, optional
        Total number of years covered by the returns. If None (default), it is
        derived from the returns actually used, i.e. after start_idx is applied.
        Pass a value only to override that.
    ddof : int
        Degrees of freedom for volatility calculation
    start_idx : int
        Number of leading periods to drop. The index's first quarter has no
        prior-quarter fair value to difference against, so its return is
        structurally undefined; including it inflates both the geometric return
        and the volatility. Defaults to 1 to drop it, matching benchmark_stats.
        Pass 0 to keep the full series.

    Returns
    -------
    annual_return : float
        Geometric annualized return
    annual_vol : float
        Annualized volatility
    """

    r = np.asarray(returns)[start_idx:]

    if num_years is None:
        num_years = len(r) / periods_per_year

    # Geometric annualized return
    annual_return = np.prod(1 + r) ** (1 / num_years) - 1

    # Annualized volatility
    vol_q = np.std(r, ddof=ddof)
    annual_vol = vol_q * np.sqrt(periods_per_year)

    return annual_return, annual_vol


def lag1_autocorrelation(returns, start_idx: int = 1) -> float:
    """
    Lag-1 autocorrelation of a periodic return series -- the standard
    diagnostic for appraisal/model-mark "smoothing" (Geltner 1991,
    Getmansky-Lo-Makarov 2004). Private-credit FV marks are infrequent and
    often model-based rather than transaction prices, so true economic
    shocks bleed into the next quarter's mark instead of showing up fully
    in the quarter they occur; this shows up as strong positive
    autocorrelation and mechanically shrinks measured volatility (and
    inflates Sharpe) without reducing real risk. A value near 0 implies
    marks are close to i.i.d.; the CDLI benchmark itself typically prints
    ~0.6-0.7 here, since it is also built from valuation-based marks.
    """
    r = np.asarray(returns, dtype=float)[start_idx:]
    return np.corrcoef(r[:-1], r[1:])[0, 1]


def unsmooth_ar1_returns(returns, start_idx: int = 1, theta: Optional[float] = None):
    """
    Geltner/Getmansky-Lo-Makarov-style AR(1) de-smoothing. Models the
    observed return as r_obs_t = (1-theta)*r_true_t + theta*r_obs_{t-1}
    (this quarter's mark is a blend of the true shock and last quarter's
    mark) and inverts it:

        r_true_t = (r_obs_t - theta * r_obs_{t-1}) / (1 - theta)

    theta defaults to the series' own lag-1 autocorrelation (the standard
    simple estimator under this model), clipped to [0, 0.95] to avoid a
    degenerate divide-by-~0 from a noisy negative estimate. With only a
    handful of quarters of history, theta itself is a noisy estimate --
    treat the de-smoothed vol as an order-of-magnitude correction, not a
    precise number.

    Returns (r_unsmoothed, theta_used); r_unsmoothed has one fewer
    observation than the (start_idx-trimmed) input series.
    """
    r = np.asarray(returns, dtype=float)[start_idx:]
    if theta is None:
        theta = np.clip(lag1_autocorrelation(r, start_idx=0), 0.0, 0.95)
    r_unsmoothed = (r[1:] - theta * r[:-1]) / (1 - theta)
    return r_unsmoothed, theta


def compute_unsmoothed_return_and_vol(
    returns,
    periods_per_year: int = 4,
    start_idx: int = 1,
    theta: Optional[float] = None,
    ddof: int = 1,
):
    """
    compute_annualized_return_and_vol(), but first passes the series
    through unsmooth_ar1_returns() -- use this alongside (not instead of)
    the raw figure to see how much of the raw Sharpe ratio is an artifact
    of valuation smoothing rather than real risk-adjusted performance.
    """
    r_unsmoothed, theta_used = unsmooth_ar1_returns(returns, start_idx=start_idx, theta=theta)
    annual_return, annual_vol = compute_annualized_return_and_vol(
        r_unsmoothed, periods_per_year=periods_per_year, ddof=ddof, start_idx=0
    )
    return annual_return, annual_vol, theta_used


def compute_max_drawdown(level) -> float:
    """
    Maximum peak-to-trough decline of an index level series -- a
    smoothing-immune complement to volatility. It doesn't rely on the
    period-to-period variance of marks at all, so it isn't distorted the
    same way: it only requires that a real loss eventually shows up as a
    level below a prior peak, however long the smoothing delays it. A
    monotonically rising level (0% drawdown) across a multi-year window
    that included real rate and credit cycles is itself a red flag that
    the marks aren't capturing realized risk.
    """
    lvl = np.asarray(level, dtype=float)
    running_max = np.maximum.accumulate(lvl)
    drawdown = (lvl - running_max) / running_max
    return float(drawdown.min())


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


def benchmark_stats(
    index_df,
    cdli_csv_path: str,
    index_label: str,
    benchmark_label: str,
    start_idx: int = 1,
    periods_per_year: int = 4,
) -> dict:
    """
    Correlation and tracking error of the index's quarterly returns against a
    CDLI benchmark, plus each calendar year's compounded return for both, over
    the quarters both cover. The index's first quarter is dropped (start_idx), as in
    compute_tracking_error, so the numbers match the ones the script prints.
    """
    idx = _ensure_pandas(index_df)[["cal_q", "IndexReturn"]].iloc[start_idx:]
    bench = load_cdli_returns(cdli_csv_path).rename(columns={"CDLI": "Bench"})[["cal_q", "Bench"]]
    d = idx.merge(bench, on="cal_q")
    diff = d["IndexReturn"] - d["Bench"]
    out = {
        "index": index_label, "benchmark": benchmark_label,
        "first_q": d["cal_q"].iloc[0], "last_q": d["cal_q"].iloc[-1], "quarters": len(d),
        "correlation": d["IndexReturn"].corr(d["Bench"]),
        "tracking_error_q": diff.std(),
        "tracking_error_ann": diff.std() * np.sqrt(periods_per_year),
        "mean_gap_q": diff.mean(),
    }
    for year, g in d.groupby(d["cal_q"].str[:4]):
        out[f"index_{year}"] = float((1 + g["IndexReturn"]).prod() - 1)
        out[f"bench_{year}"] = float((1 + g["Bench"]).prod() - 1)
        out[f"quarters_{year}"] = len(g)
    return out


def plot_benchmark_comparison(panels, save_path=None):
    """
    Presentation chart: one panel per (index, benchmark) pair, quarterly returns
    of our index (blue) against the CDLI benchmark (orange), with the pair's
    correlation and annualized tracking error under the panel title.

    panels: list of dicts with keys title, index_df, cdli_csv_path, index_label,
    benchmark_label, stats (a benchmark_stats result).
    """
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5.2), facecolor=surface, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        s = p["stats"]
        idx = _ensure_pandas(p["index_df"])[["cal_q", "IndexReturn"]]
        bench = load_cdli_returns(p["cdli_csv_path"]).rename(columns={"CDLI": "Bench"})[["cal_q", "Bench"]]
        d = idx.merge(bench, on="cal_q")
        d = d[(d["cal_q"] >= s["first_q"]) & (d["cal_q"] <= s["last_q"])]
        x = np.arange(len(d))
        ax.plot(x, 100 * d["Bench"], color=orange, linewidth=2.4, marker="o", markersize=5, label=p["benchmark_label"])
        ax.plot(x, 100 * d["IndexReturn"], color=blue, linewidth=2.4, marker="o", markersize=5, label=p["index_label"])
        ax.set_facecolor(surface)
        ax.set_title(p["title"], loc="left", fontsize=14, color=ink, weight="bold", pad=30)
        ax.text(0, 1.035, f"Correlation {s['correlation']:.2f} · tracking error {100 * s['tracking_error_ann']:.2f}% a year",
                transform=ax.transAxes, fontsize=11, color=muted, va="bottom")
        ax.set_xticks(x, d["cal_q"], rotation=45)
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.1f}%"))
        ax.set_ylim(0, 4)
        ax.tick_params(colors=muted, labelsize=10, length=0)
        ax.yaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.legend(frameon=False, fontsize=10.5, loc="lower left", labelcolor=ink)
    axes[0].set_ylabel("Return per quarter", color=muted, fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.16, wspace=0.08)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, facecolor=surface)
    plt.close(fig)


SENIOR_LOANS = ["SENIOR_SECURED_1L", "SENIOR_SECURED_2L", "SENIOR_UNSECURED", "SENIOR_SECURED_UNSPEC", "UNITRANCHE"]


if __name__ == "__main__":
    # Draw off-screen: an interactive backend would open a window per chart.
    import matplotlib
    matplotlib.use("Agg")
    import warnings
    warnings.filterwarnings("ignore", message=".*non-interactive.*")
    from paths import FINAL_DIR, INDEX_PLOTS_DIR, REFERENCE_DIR

    CSV_PATH = FINAL_DIR / "FINAL_DATASET.csv"
    CDLI_CSV_PATH = REFERENCE_DIR / "cdli.csv"
    CDLI_S_CSV_PATH = REFERENCE_DIR / "cdli-s.csv"     # CDLI senior-loans benchmark

    print("=== index_construction.py: whole-market flow-based pipeline ===")
    df = load_and_prepare_investment_data(str(CSV_PATH))
    df = compute_final_interest_rates(df)
    print(f"rows after load + rate resolution: {df.height}")

    flows = compute_position_level_flows(df, qsort_expr, safe_div)
    weighted = compute_market_weights(flows, safe_div)
    decomp = aggregate_return_decomposition(weighted, safe_div)
    print("\n--- quarterly return decomposition ---")
    print(decomp)

    index_df = compute_flow_based_market_index(weighted)
    print("\n--- flow-based market index ---")
    print(index_df.select(["cal_q", "IndexReturn", "IndexLevel"]))

    # vs. actual CDLI
    cdli_returns = load_cdli_returns(str(CDLI_CSV_PATH))
    cmp_df = (
        index_df.to_pandas()[["cal_q", "IndexReturn"]]
        .merge(
            cdli_returns.rename(columns={"CDLI": "CDLI_Return"})[["cal_q", "CDLI_Return"]],
            on="cal_q", how="inner",
        )
    )
    corr, te_q, te_a = compute_tracking_error(cmp_df, start_idx=1)
    print("\n--- vs. actual CDLI ---")
    print(f"correlation: {corr.loc['IndexReturn', 'CDLI_Return']:.4f}")
    print(f"tracking error (quarterly): {te_q:.4%}   (annualized): {te_a:.4%}")

    # annualized return/vol, smoothing diagnostics, and drawdown -- see
    # unsmooth_ar1_returns / compute_max_drawdown docstrings for why raw
    # volatility on appraisal-based marks understates true risk
    returns = index_df.sort("cal_q")["IndexReturn"].to_numpy()
    level = index_df.sort("cal_q")["IndexLevel"].to_numpy()
    ann_ret, ann_vol = compute_annualized_return_and_vol(returns)
    rho = lag1_autocorrelation(returns)
    un_ret, un_vol, theta = compute_unsmoothed_return_and_vol(returns)
    mdd = compute_max_drawdown(level)
    print(f"\nannualized return: {ann_ret:.4%}   annualized vol (raw marks): {ann_vol:.4%}")
    print(f"lag-1 autocorrelation: {rho:.3f}   de-smoothed vol: {un_vol:.4%} (theta={theta:.3f})")
    print(f"max drawdown: {mdd:.4%}")

    # senior loans: filter the flowed positions, re-weight within the bucket
    loans = compute_market_weights(flows.filter(pl.col("instrument_seniority").is_in(SENIOR_LOANS)), safe_div)
    index_loans = compute_flow_based_market_index(loans)

    # CDLI comparison for the deck: both pairs side by side, the numbers saved next to the chart
    stats_all = benchmark_stats(index_df, str(CDLI_CSV_PATH), "Our BDC loan index", "CDLI")
    stats_sen = benchmark_stats(index_loans, str(CDLI_S_CSV_PATH), "Our senior-loan index", "CDLI-S")
    pd.DataFrame([stats_all, stats_sen]).to_csv(INDEX_PLOTS_DIR / "index_vs_cdli_stats.csv", index=False)
    plot_benchmark_comparison([
        dict(title="All BDC loans vs CDLI", index_df=index_df, cdli_csv_path=str(CDLI_CSV_PATH),
             index_label="Our BDC loan index", benchmark_label="CDLI", stats=stats_all),
        dict(title="Senior loans vs CDLI-S", index_df=index_loans, cdli_csv_path=str(CDLI_S_CSV_PATH),
             index_label="Our senior-loan index", benchmark_label="CDLI-S (senior loans)", stats=stats_sen),
    ], save_path=INDEX_PLOTS_DIR / "index_vs_cdli_panels.png")
    print(pd.DataFrame([stats_all, stats_sen]).T.to_string())
