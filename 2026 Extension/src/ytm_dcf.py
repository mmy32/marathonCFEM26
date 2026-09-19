"""
ytm_dcf.py

Pipeline:

    1. impute_maturity_date()   -- structured field > parsed identifier
                                    text > paper's 3-year default
    2. build_cashflow_grid()    -- per-loan quarterly cash-coupon /
                                    PIK-compounded terminal principal
    3. solve_ytm_vectorized()   -- batched Newton-Raphson root-find across
                                    the whole panel at once
    4. aggregate_ytm_index()    -- FV-weighted quarterly index series,
                                    same convention as compute_pull_to_par
    5. plot_ytm_vs_index_and_cdli() -- overlay against index_construction.py's
                                    compute_flow_based_market_index() and the
                                    actual CDLI benchmark (data/processed/cdli.csv)

Usage, chained onto the existing pipeline:

    df = load_and_prepare_investment_data(csv_path)
    df = compute_final_interest_rates(df)

    df = impute_maturity_date(df)
    n, coupon, terminal, price, usable = build_cashflow_grid(df)
    y_q, converged = solve_ytm_vectorized(n, coupon, terminal, price, y0=usable["IR_Final"].to_numpy() / 4)

    ytm_df = aggregate_ytm_index(usable, y_q, converged)

    flow_df = compute_position_level_flows(df, qsort_expr, safe_div)
    flow_df = compute_market_weights(flow_df, safe_div)
    flow_index_df = compute_flow_based_market_index(flow_df)

    plot_ytm_vs_index_and_cdli(ytm_df, flow_index_df, cdli_csv_path="data/processed/cdli.csv")
"""

from datetime import date

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from index_construction import load_cdli_returns

# --- maturity imputation --------------------------------------------------

_KW_DATE_FULL = (
    r"(?i)\b(?:due|maturity)\b(?:\s+date)?\s*[:\-]?\s*"
    r"(?P<mo>\d{1,2})/(?P<da>\d{1,2})/(?P<yr>\d{2,4})"
)
_DATE_KW_FULL = (
    r"(?i)(?P<mo>\d{1,2})/(?P<da>\d{1,2})/(?P<yr>\d{2,4})\s*,?\s*\b(?:due|maturity)\b"
)
_KW_DATE_MY = (
    r"(?i)\b(?:due|maturity)\b(?:\s+date)?\s*[:\-]?\s*(?P<mo>\d{1,2})/(?P<yr>\d{4})"
)


def _safe_date(mo, da, yr):
    """Calendar-safe MM/DD/YYYY (or MM/YYYY, day=1) construction. Two-digit
    years are read as 20YY. Returns None on anything not a real date
    (covers the malformed/near-miss matches a regex alone can't rule out,
    e.g. day=31 in a 30-day month)."""
    try:
        m, d, y = int(mo), int(da), int(yr)
        if y < 100:
            y += 2000
        return date(y, m, d)
    except (ValueError, TypeError):
        return None


def parse_maturity_from_identifier(
    df: pl.DataFrame,
    identifier_col: str = "investment_identifier",
) -> pl.DataFrame:
    """
    Recover a maturity date from free-text investment_identifier strings,
    e.g. "Due 10/19/2027", "Maturity Date 03/22/39", "due 8/2030" (month/year
    only -> day defaults to 1), or "5/29/2028 Maturity" (date precedes the
    keyword). Adds `maturity_date_parsed` (pl.Date, null where nothing
    matched).

    Only ~5,500 of the ~304K rows lacking a structured InvestmentMaturityDate
    have a cleanly parseable numeric date in the identifier at all (~1.8%
    incremental recovery) -- most identifiers referencing "maturity" already
    carry the structured field too, and most of the rest don't mention a
    date in a numeric MM/DD/YYYY form (spelled-out months like "September
    2027" are out of scope here). The remaining gap is expected to fall
    through to impute_maturity_date()'s 3-year default, by design.
    """
    extracted = df.select(
        pl.col(identifier_col).str.extract_groups(_KW_DATE_FULL).alias("kw_full"),
        pl.col(identifier_col).str.extract_groups(_DATE_KW_FULL).alias("date_kw"),
        pl.col(identifier_col).str.extract_groups(_KW_DATE_MY).alias("kw_my"),
    ).with_columns(
        pl.coalesce(
            [pl.col("kw_full").struct.field("mo"), pl.col("date_kw").struct.field("mo")]
        ).alias("mo_full"),
        pl.coalesce(
            [pl.col("kw_full").struct.field("da"), pl.col("date_kw").struct.field("da")]
        ).alias("da_full"),
        pl.coalesce(
            [pl.col("kw_full").struct.field("yr"), pl.col("date_kw").struct.field("yr")]
        ).alias("yr_full"),
        pl.col("kw_my").struct.field("mo").alias("mo_my"),
        pl.col("kw_my").struct.field("yr").alias("yr_my"),
    )

    # Calendar validation needs real date arithmetic, not just regex shape,
    # so this last step is row-wise Python over the (small) candidate set
    # rather than a polars expression.
    rows = extracted.select(["mo_full", "da_full", "yr_full", "mo_my", "yr_my"]).to_dicts()
    parsed = []
    for r in rows:
        d = None
        if r["mo_full"] is not None:
            d = _safe_date(r["mo_full"], r["da_full"], r["yr_full"])
        if d is None and r["mo_my"] is not None:
            d = _safe_date(r["mo_my"], "1", r["yr_my"])
        parsed.append(d)

    return df.with_columns(pl.Series("maturity_date_parsed", parsed, dtype=pl.Date))


def impute_maturity_date(
    df: pl.DataFrame,
    identifier_col: str = "investment_identifier",
    structured_col: str = "InvestmentMaturityDate",
    period_col: str = "period",
    default_years: float = 3.0,
    valid_ttm_range: tuple = (0.1, 7.0),
) -> pl.DataFrame:
    """
    Impute a maturity date per row, prioritizing:
      1. the structured InvestmentMaturityDate XBRL field (~38% of rows)
      2. a date parsed out of investment_identifier (recovers a further
         ~1.8% of the rows that lack (1))
      3. the paper's own fallback -- observation period + `default_years`

    A candidate is only accepted if its implied years-to-maturity falls
    inside `valid_ttm_range`; anything outside (already-matured positions,
    implausible multi-decade tenors -- both of which show up in the
    structured field itself) is treated as missing and falls through to
    the next source. Adds maturity_date_imputed, maturity_source
    ("structured" / "parsed" / "default_3yr"), and years_to_maturity.
    """
    df = df.with_columns(
        pl.col(structured_col).str.to_date(strict=False).alias("_structured_dt"),
        pl.col(period_col).str.to_date(strict=False).alias("_period_dt"),
    )
    df = parse_maturity_from_identifier(df, identifier_col)

    df = df.with_columns(
        ((pl.col("_structured_dt") - pl.col("_period_dt")).dt.total_days() / 365.25).alias(
            "_ttm_structured"
        ),
        ((pl.col("maturity_date_parsed") - pl.col("_period_dt")).dt.total_days() / 365.25).alias(
            "_ttm_parsed"
        ),
    )

    lo, hi = valid_ttm_range
    df = df.with_columns(
        pl.when(pl.col("_ttm_structured").is_between(lo, hi))
        .then(pl.col("_structured_dt"))
        .when(pl.col("_ttm_parsed").is_between(lo, hi))
        .then(pl.col("maturity_date_parsed"))
        .otherwise(pl.col("_period_dt").dt.offset_by(f"{default_years:.0f}y"))
        .alias("maturity_date_imputed"),
        pl.when(pl.col("_ttm_structured").is_between(lo, hi))
        .then(pl.lit("structured"))
        .when(pl.col("_ttm_parsed").is_between(lo, hi))
        .then(pl.lit("parsed"))
        .otherwise(pl.lit("default_3yr"))
        .alias("maturity_source"),
    )

    df = df.with_columns(
        ((pl.col("maturity_date_imputed") - pl.col("_period_dt")).dt.total_days() / 365.25).alias(
            "years_to_maturity"
        )
    )

    return df.drop(["_structured_dt", "_period_dt", "_ttm_structured", "_ttm_parsed"])


# --- cash-flow construction + vectorized YTM solve -------------------------

def build_cashflow_grid(
    df: pl.DataFrame,
    fv_col: str = "FV",
    par_col: str = "PAR",
    pic_col: str = "PIC_Final",
    pik_col: str = "PIK_Final",
    ttm_col: str = "years_to_maturity",
    min_periods: int = 1,
    max_periods: int = 28,
):
    """
    Build the per-row quarterly cash-flow inputs needed for a bullet-
    repayment DCF. PAR is reported as the *current* outstanding principal
    as of each filing period, so the coupon base itself compounds forward:
    the balance at the start of period t is PAR * (1 + PIK_Final/4) ** (t-1),
    and the cash coupon paid at t is that balance times PIC_Final/4. This
    still holds PIC/PIK *rates* flat over the remaining life (unavoidable --
    there's no forward curve for a bespoke floating spread + PIK toggle,
    the same flat-forward assumption current yield already makes for
    floating-rate loans), but it no longer holds the coupon *amount* flat,
    since -- under that same flat-rate assumption -- the balance it's
    applied to is already known to grow. The terminal principal payment
    uses the same compounding, PAR * (1 + PIK_Final/4) ** n. No
    amortization schedule is available, so every loan is treated as a
    bullet.

    Returns (n_periods, coupon, terminal, price, usable) plus the filtered
    frame they were built from (needed downstream to re-attach cal_q / FV
    for aggregation), filtered to rows with usable FV, PAR, and a rate
    estimate. coupon is now a (n_rows, max_n) matrix rather than a flat
    per-row scalar, since it varies by period -- entries beyond each row's
    own n_periods are zeroed via the same mask solve_ytm_vectorized
    rebuilds internally. n_periods is clipped to [min_periods, max_periods]
    -- min_periods keeps a near-maturity loan from collapsing to a
    degenerate zero-period solve; max_periods bounds memory in the
    cash-flow matrix.
    """
    usable = df.filter(
        pl.col(fv_col).is_not_null()
        & (pl.col(fv_col) > 0)
        & pl.col(par_col).is_not_null()
        & (pl.col(par_col) > 0)
        & pl.col(pic_col).is_not_null()
        & pl.col(pik_col).is_not_null()
        & pl.col(ttm_col).is_not_null()
    )

    n = (usable[ttm_col].to_numpy() * 4).round().astype(int)
    n = np.clip(n, min_periods, max_periods)

    par = usable[par_col].to_numpy()
    pic = usable[pic_col].to_numpy()
    pik = usable[pik_col].to_numpy()

    max_n = int(n.max())
    t = np.arange(1, max_n + 1)[None, :]          # (1, max_n)
    mask = t <= n[:, None]                          # (n_rows, max_n)

    balance_start = par[:, None] * (1 + pik[:, None] / 4) ** (t - 1)
    coupon = np.where(mask, balance_start * (pic[:, None] / 4), 0.0)

    terminal = par * (1 + pik / 4) ** n
    price = usable[fv_col].to_numpy()

    return n, coupon, terminal, price, usable


def solve_ytm_vectorized(
    n_periods: np.ndarray,
    coupon: np.ndarray,
    terminal: np.ndarray,
    price: np.ndarray,
    y0: np.ndarray,
    max_iter: int = 25,
    tol: float = 1e-6,
    y_bounds: tuple = (-0.5, 2.0),
):
    """
    Batched Newton-Raphson YTM solve across the whole panel at once
    (one scalar-per-row equation, solved for all rows simultaneously via a
    padded cash-flow matrix -- avoids an unvectorized scipy.optimize call
    per row, which would not finish in reasonable time at ~390K+ rows).

    NPV(y) = sum_t coupon_t / (1+y)^t  +  terminal / (1+y)^n  -  price = 0

    `coupon` is a (n_rows, max_n) matrix -- it grows period-over-period
    with PAR's PIK accretion (see build_cashflow_grid), not a flat
    per-row scalar, so the mask and discount factors below are applied
    elementwise against the whole matrix rather than broadcast from a
    per-row value.

    y0 (quarterly) should start close to the root -- the loan's own
    resolved contractual rate (IR_Final / 4) converges in well under
    `max_iter` steps for well-behaved inputs. Returns (y_quarterly,
    converged) -- non-converged rows should fall back to the linear
    yield-to-3yr-takeout approximation for reporting.
    """
    y0 = np.nan_to_num(np.asarray(y0, dtype=float), nan=0.0)

    max_n = coupon.shape[1]
    t = np.arange(1, max_n + 1)[None, :]  # (1, max_n)
    mask = t <= n_periods[:, None]  # (n_rows, max_n) -- coupon only through each row's own n

    y = np.clip(y0, *y_bounds)

    for _ in range(max_iter):
        disc = (1 + y[:, None]) ** t  # (n_rows, max_n)
        pv_coupons = np.where(mask, coupon / disc, 0.0).sum(axis=1)
        disc_n = (1 + y) ** n_periods
        pv_terminal = terminal / disc_n
        npv = pv_coupons + pv_terminal - price

        d_coupons = np.where(mask, -t * coupon / (disc * (1 + y[:, None])), 0.0).sum(axis=1)
        d_terminal = -n_periods * terminal / (disc_n * (1 + y))
        dnpv = d_coupons + d_terminal

        step = np.where(np.abs(dnpv) > 1e-12, npv / dnpv, 0.0)
        y = np.clip(y - step, *y_bounds)

    # final residual check for convergence flag
    disc = (1 + y[:, None]) ** t
    pv_coupons = np.where(mask, coupon / disc, 0.0).sum(axis=1)
    pv_terminal = terminal / (1 + y) ** n_periods
    resid = pv_coupons + pv_terminal - price
    converged = np.abs(resid) < tol * np.maximum(price, 1.0)

    return y, converged


def aggregate_ytm_index(
    usable: pl.DataFrame,
    y_quarterly: np.ndarray,
    converged: np.ndarray,
    fallback_q: np.ndarray = None,
    fv_col: str = "FV",
    quarter_col: str = "cal_q",
) -> pl.DataFrame:
    """
    FV-weighted quarterly YTM index, kept on the same per-quarter basis as
    yield_methodology.py's current_yield_q / yield_to_3yr_takeout_q (no
    annualization) so all three series are directly comparable to each
    other and to the actual constructed quarterly index return. Uses the
    same weighting convention as compute_pull_to_par (current-quarter FV,
    since this is a point-in-time valuation metric, not a flow-based
    return). Non-converged rows use `fallback_q` (pass the row-aligned
    quarterly yield-to-3yr-takeout approximation) if provided, else are
    dropped from the average.
    """
    y_q = y_quarterly
    if fallback_q is not None:
        y_q = np.where(converged, y_q, fallback_q)
        weight_mask = np.ones_like(y_q, dtype=bool)
    else:
        weight_mask = converged

    out = usable.with_columns(
        pl.Series("ytm_q", y_q),
        pl.Series("_include", weight_mask),
    )

    return (
        out.filter(pl.col("_include"))
        .group_by(quarter_col)
        .agg(
            (pl.col("ytm_q") * pl.col(fv_col)).sum().alias("_num"),
            pl.col(fv_col).sum().alias("_den"),
            pl.len().alias("n_positions"),
        )
        .with_columns((pl.col("_num") / pl.col("_den")).alias("ytm_index_q"))
        .select([quarter_col, "ytm_index_q", "n_positions"])
        .sort(quarter_col)
    )


def compute_ytm_cross_sectional_dispersion(
    usable: pl.DataFrame,
    y_quarterly: np.ndarray,
    converged: np.ndarray,
    fv_col: str = "FV",
    quarter_col: str = "cal_q",
) -> pl.DataFrame:
    """
    Per-quarter cross-sectional dispersion of the per-loan DCF-YTM
    (FV-weighted std, and p10/p50/p90), as a risk measure that sidesteps
    valuation smoothing entirely -- unlike aggregate_ytm_index's
    time-series volatility (see index_construction.compute_max_drawdown /
    unsmooth_ar1_returns), this needs only a single quarter's snapshot, so
    it isn't distorted by serial correlation in repeated marks. It
    captures how much credit-risk heterogeneity is priced into the book
    right now, e.g. a fat right tail (p90 far above the median) flags a
    subset of distressed positions being priced for real loss even while
    the FV-weighted average looks calm.
    """
    out = usable.with_columns(
        pl.Series("ytm_q", y_quarterly),
        pl.Series("_include", converged),
    ).filter(pl.col("_include"))

    def _quarter_stats(group: pl.DataFrame) -> dict:
        y = group["ytm_q"].to_numpy()
        w = group[fv_col].to_numpy()
        wmean = np.average(y, weights=w)
        wstd = np.sqrt(np.average((y - wmean) ** 2, weights=w))
        p10, p50, p90 = np.percentile(y, [10, 50, 90])
        return {
            quarter_col: group[quarter_col][0],
            "n_positions": len(y),
            "ytm_fv_weighted_mean": wmean,
            "ytm_fv_weighted_std": wstd,
            "ytm_p10": p10,
            "ytm_p50": p50,
            "ytm_p90": p90,
        }

    rows = [_quarter_stats(g) for _, g in out.group_by(quarter_col, maintain_order=False)]
    return pl.DataFrame(rows).sort(quarter_col)


def _merge_ytm_index_cdli(
    ytm_df: pl.DataFrame,
    flow_index_df: pl.DataFrame,
    cdli_csv_path: str,
    quarter_col: str,
    ytm_col: str,
    index_return_col: str,
    cdli_col: str,
    start_idx: int,
):
    """Shared merge behind plot_ytm_vs_index_and_cdli and
    compute_ytm_index_cdli_stats, so the plot and the reported numbers are
    always computed off the same panel."""
    cdli_returns = load_cdli_returns(cdli_csv_path, quarter_col)

    merged = (
        ytm_df.to_pandas()[[quarter_col, ytm_col]]
        .merge(flow_index_df.to_pandas()[[quarter_col, index_return_col]], on=quarter_col, how="outer")
        .merge(cdli_returns[[quarter_col, cdli_col]], on=quarter_col, how="left")
        .sort_values(quarter_col)
    )
    return merged.iloc[start_idx:]


def plot_ytm_vs_index_and_cdli(
    ytm_df: pl.DataFrame,
    flow_index_df: pl.DataFrame,
    cdli_csv_path: str,
    quarter_col: str = "cal_q",
    ytm_col: str = "ytm_index_q",
    index_return_col: str = "IndexReturn",
    cdli_col: str = "CDLI",
    title: str = "DCF-YTM vs. Flow-Based Index Return vs. CDLI (Quarterly)",
    ylim: tuple = (0.0, 0.05),
    figsize: tuple = (10, 5),
    start_idx: int = 1,
    save_path: str = None,
):
    """
    Compare the DCF-solved YTM index (this module) against
    index_construction.py's realized flow-based market-return index
    (compute_flow_based_market_index) and the actual CDLI benchmark, all
    per-quarter (not annualized). DCF-YTM is a forward-looking priced-in
    yield while the other two are backward-looking realized total returns
    (price + income) -- overlaid as three views of the same quarterly
    panel, not as directly interchangeable quantities.

    `ytm_df` is aggregate_ytm_index()'s output; `flow_index_df` is
    compute_flow_based_market_index()'s output. `cdli_csv_path` is loaded
    via index_construction.load_cdli_returns.

    start_idx drops the leading quarter, matching index_construction.py's
    plot_index_vs_cdli_returns: the index's first quarter has no FV_prev,
    so its realized return is structurally undefined there.

    Pass save_path to also write the figure to disk (e.g. a .png path).
    """
    pdf = _merge_ytm_index_cdli(
        ytm_df, flow_index_df, cdli_csv_path, quarter_col, ytm_col, index_return_col, cdli_col, start_idx
    )

    plt.figure(figsize=figsize)
    plt.plot(pdf[quarter_col], pdf[ytm_col] * 100, label="DCF-YTM (Q, imputed maturity)")
    plt.plot(pdf[quarter_col], pdf[index_return_col] * 100, label="Flow-Based Index Return (Q, index_construction.py)")
    plt.plot(pdf[quarter_col], pdf[cdli_col] * 100, label="CDLI Return (Q, actual)", linestyle="--")

    plt.legend()
    plt.ylim(ylim[0] * 100, ylim[1] * 100)
    plt.xticks(rotation=45)
    plt.xlabel("Quarter")
    plt.ylabel("Yield / Return, per Quarter (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def compute_ytm_index_cdli_stats(
    ytm_df: pl.DataFrame,
    flow_index_df: pl.DataFrame,
    cdli_csv_path: str,
    quarter_col: str = "cal_q",
    ytm_col: str = "ytm_index_q",
    index_return_col: str = "IndexReturn",
    cdli_col: str = "CDLI",
    start_idx: int = 1,
    periods_per_year: int = 4,
) -> dict:
    """
    Correlation and tracking error of DCF-YTM, and separately of
    index_construction.py's flow-based index return, each against the
    actual CDLI benchmark -- computed off the same merged quarterly panel
    plot_ytm_vs_index_and_cdli() plots (see compute_tracking_error in
    index_construction.py for the equivalent index-vs-CDLI-only version).
    """
    pdf = _merge_ytm_index_cdli(
        ytm_df, flow_index_df, cdli_csv_path, quarter_col, ytm_col, index_return_col, cdli_col, start_idx
    )

    ytm_diff = pdf[ytm_col] - pdf[cdli_col]
    index_diff = pdf[index_return_col] - pdf[cdli_col]

    return {
        "n_quarters": len(pdf),
        "ytm_vs_cdli_corr": pdf[ytm_col].corr(pdf[cdli_col]),
        "index_vs_cdli_corr": pdf[index_return_col].corr(pdf[cdli_col]),
        "ytm_vs_cdli_te_q": ytm_diff.std(),
        "ytm_vs_cdli_te_a": ytm_diff.std() * np.sqrt(periods_per_year),
        "index_vs_cdli_te_q": index_diff.std(),
        "index_vs_cdli_te_a": index_diff.std() * np.sqrt(periods_per_year),
    }


if __name__ == "__main__":
    from paths import PROCESSED_DIR
    from index_construction import (
        load_and_prepare_investment_data, compute_final_interest_rates,
        qsort_expr, safe_div, compute_position_level_flows, compute_market_weights,
        compute_flow_based_market_index,
    )

    CSV_PATH = PROCESSED_DIR / "data_private_credit_FINAL_enriched.csv"
    CDLI_CSV_PATH = PROCESSED_DIR / "cdli.csv"

    print("=== ytm_dcf.py: DCF-solved yield-to-maturity ===")
    df = load_and_prepare_investment_data(str(CSV_PATH))
    df = compute_final_interest_rates(df)

    df = impute_maturity_date(df)
    n, coupon, terminal, price, usable = build_cashflow_grid(df)
    y_q, converged = solve_ytm_vectorized(n, coupon, terminal, price, y0=usable["IR_Final"].to_numpy() / 4)
    print(f"usable loans: {usable.height}   Newton-Raphson convergence: {converged.mean():.4%}")

    ytm_df = aggregate_ytm_index(usable, y_q, converged)
    print("\n--- DCF-YTM index (ytm_index_q) ---")
    print(ytm_df)

    flow_df = compute_position_level_flows(df, qsort_expr, safe_div)
    flow_df = compute_market_weights(flow_df, safe_div)
    flow_index_df = compute_flow_based_market_index(flow_df)

    stats = compute_ytm_index_cdli_stats(ytm_df, flow_index_df, cdli_csv_path=str(CDLI_CSV_PATH))
    print("\n--- performance vs. actual CDLI ---")
    for k, v in stats.items():
        print(f"{k}: {v}")

    disp = compute_ytm_cross_sectional_dispersion(usable, y_q, converged)
    print("\n--- cross-sectional DCF-YTM dispersion (last 4 quarters) ---")
    print(disp.tail(4))

    plot_ytm_vs_index_and_cdli(
        ytm_df, flow_index_df, cdli_csv_path=str(CDLI_CSV_PATH),
        save_path="ytm_vs_index_vs_cdli.png",
    )