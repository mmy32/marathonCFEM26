"""
concentration_crowding.py

Objective: concentration and crowding measures across BDCs -- a signal for
which specific BDCs are carrying more credit risk than the market as a
whole, via how much of the market they represent and how much of their book
overlaps with other managers'.

Pipeline:
    1. compute_cik_aum_by_quarter() / compute_market_concentration() --
       Herfindahl-Hirschman Index, effective number of BDCs, and top-N AUM
       share, from current-quarter fair value by CIK.

    2. compute_position_crowding() / top_crowded_positions() -- how much
       market AUM sits in borrowers (borrower_canonical) held by two or
       more distinct BDCs in the same quarter, and which names drive it.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py
from paths import add_code_dirs_to_path
add_code_dirs_to_path()  # index_construction lives in 2_index_construction/

import polars as pl
import matplotlib.pyplot as plt

from index_construction import (
    safe_div,
    normalize_positions,  # FV > 0, normalized borrower name
)

DEFAULT_MIN_HOLDERS = 2


# =====================
# 1. AUM concentration across BDCs
# =====================

def compute_cik_aum_by_quarter(
    df: pl.DataFrame,
    quarter_col: str = "cal_q",
    cik_col: str = "cik",
    fv_col: str = "FV",
) -> pl.DataFrame:
    """
    Current-quarter AUM (sum of FV) per (CIK, quarter) -- the raw input for
    concentration and crowding. Uses the position-level panel directly (no
    market-weighting), since concentration is a statement about how AUM
    itself is distributed across managers, not about return.
    """
    return (
        df.group_by([cik_col, quarter_col])
        .agg(pl.col(fv_col).sum().alias("aum"))
        .sort([cik_col, quarter_col])
    )


def compute_market_concentration(
    aum_by_cik_q: pl.DataFrame,
    quarter_col: str = "cal_q",
    aum_col: str = "aum",
    top_ns: tuple = (5, 10),
) -> pl.DataFrame:
    """
    Herfindahl-Hirschman Index and top-N AUM concentration per quarter, on
    AUM shares among CIKs with positive summed AUM that quarter (a CIK
    whose *net* summed FV is negative that quarter is excluded from the
    share base rather than allowed to inflate other managers' shares
    through a negative denominator contribution).

    HHI is computed on shares in [0, 1], not the conventional 0-10,000
    scale, so `effective_n_ciks = 1 / HHI` is directly interpretable as "how
    many equally-sized managers would produce this same concentration."

    AUM convention note: this sums FV per CIK first and only excludes a CIK
    whose *net* AUM is negative, unlike compute_position_crowding() (which
    filters individual rows to FV > 0 before aggregating) -- concentration
    asks "how much does this manager run, net" (a write-down should net
    against the rest of that manager's book), crowding asks "is this
    specific position held elsewhere" (a non-positive mark on that specific
    position isn't a real holding regardless of what else the manager
    holds). The discrepancy between the two conventions is small on this
    panel (checked, not assumed): well under 1% of AUM, and only a handful
    of (cik, quarter) pairs actually flip inclusion status. See
    compute_position_crowding()'s docstring for the crowding side of this.
    """
    positive = aum_by_cik_q.filter(pl.col(aum_col) > 0)

    shares = positive.with_columns(
        (pl.col(aum_col) / pl.col(aum_col).sum().over(quarter_col)).alias("aum_share")
    ).with_columns(
        pl.col("aum_share")
        .rank(method="ordinal", descending=True)
        .over(quarter_col)
        .alias("_rank")
    )

    agg_exprs = [
        pl.len().alias("n_active_ciks"),
        (pl.col("aum_share") ** 2).sum().alias("hhi"),
        pl.col(aum_col).sum().alias("total_aum"),
    ]
    for n in top_ns:
        agg_exprs.append(
            pl.col("aum_share").filter(pl.col("_rank") <= n).sum().alias(f"top{n}_share")
        )

    return (
        shares.group_by(quarter_col)
        .agg(agg_exprs)
        .sort(quarter_col)
        .with_columns((1.0 / pl.col("hhi")).alias("effective_n_ciks"))
    )


# =====================
# 2. Position crowding across BDCs
# =====================

def compute_position_crowding(
    df: pl.DataFrame,
    quarter_col: str = "cal_q",
    cik_col: str = "cik",
    identifier_col: str = "borrower_canonical",
    fv_col: str = "FV",
    min_holders: int = DEFAULT_MIN_HOLDERS,
):
    """
    Cross-BDC crowding: how many distinct managers (CIKs) lend to the same
    borrower in the same quarter, and how much AUM that represents.

    Matches on the validated borrower name (`borrower_canonical`), normalized
    (whitespace-trimmed, upper-cased); rows with no borrower name are dropped.
    investment_identifier is not used: each BDC words it differently, so
    matching on it found almost no overlap (e.g. First Brands Group, 2025Q2:
    15 BDCs held it, identifier matching reported none). This is
    borrower-level overlap: two BDCs in different tranches of one borrower
    count as crowded.

    Filters to FV > 0 first, at the individual-position level (a
    non-positive fair value isn't a real held position) -- unlike
    compute_market_concentration()'s convention of netting FV per CIK
    first. See that function's docstring for why, and for the measured
    (small) size of the discrepancy this causes.

    Returns (position_level, quarterly_summary):
      - position_level: one row per (quarter, normalized borrower), with
        n_holders (distinct CIKs), n_investments (row count -- >= n_holders
        whenever one CIK holds more than one position in the borrower that
        quarter), total_fv, and `crowded`
        (n_holders >= min_holders).
      - quarterly_summary: one row per quarter, with the share of total
        market AUM sitting in crowded names, counts of crowded vs. total
        names, and the single most-crowded name's holder count.
    """
    base = normalize_positions(df, identifier_col, fv_col)

    position_level = (
        base.group_by([quarter_col, "_ident_norm"])
        .agg(
            [
                pl.col(cik_col).n_unique().alias("n_holders"),
                pl.len().alias("n_investments"),
                pl.col(fv_col).sum().alias("total_fv"),
            ]
        )
        .with_columns((pl.col("n_holders") >= min_holders).alias("crowded"))
        .sort([quarter_col, "n_holders"], descending=[False, True])
    )

    quarterly_summary = (
        position_level.group_by(quarter_col)
        .agg(
            [
                pl.len().alias("n_names"),
                pl.col("crowded").sum().alias("n_crowded_names"),
                pl.col("total_fv").filter(pl.col("crowded")).sum().alias("crowded_fv"),
                pl.col("total_fv").sum().alias("total_market_fv"),
                pl.col("n_holders").max().alias("max_holders"),
            ]
        )
        .with_columns(
            [
                (pl.col("n_crowded_names") / pl.col("n_names")).alias("crowded_name_share"),
                safe_div(pl.col("crowded_fv"), pl.col("total_market_fv")).alias(
                    "crowded_aum_share"
                ),
            ]
        )
        .sort(quarter_col)
    )

    return position_level, quarterly_summary


def top_n_in_quarter(
    df: pl.DataFrame,
    quarter: str,
    sort_cols,
    quarter_col: str = "cal_q",
    descending=True,
    top_n: int = 15,
) -> pl.DataFrame:
    """
    Drill into one quarter's top-N rows by `sort_cols` -- the shared
    filter -> sort -> head shape behind top_crowded_positions() below,
    factored out so it isn't hand-rolled again for the next per-quarter
    top-N metric.
    """
    if isinstance(sort_cols, str):
        sort_cols = [sort_cols]
    if isinstance(descending, bool):
        descending = [descending] * len(sort_cols)
    return (
        df.filter(pl.col(quarter_col) == quarter)
        .sort(sort_cols, descending=descending)
        .head(top_n)
    )


def top_crowded_positions(
    position_level: pl.DataFrame,
    quarter: str,
    quarter_col: str = "cal_q",
    top_n: int = 15,
) -> pl.DataFrame:
    """
    Most-crowded names in one quarter, ranked by number of distinct BDC
    holders (ties broken by total FV) -- a drill-down off
    compute_position_crowding()'s quarterly_summary, for spot-checking which
    borrowers/facilities a rising crowded_aum_share is actually coming from.
    """
    return top_n_in_quarter(
        position_level, quarter, ["n_holders", "total_fv"], quarter_col=quarter_col, top_n=top_n
    )


# Resolved 2026-10-22: compute_position_crowding() now matches on borrower_canonical, which
# fixes the undercount described below. Kept for reference.
# def borrower_holder_check(
#     df: pl.DataFrame,
#     name: str,
#     quarter: str,
#     quarter_col: str = "cal_q",
#     cik_col: str = "cik",
#     identifier_col: str = "investment_identifier",
#     fv_col: str = "FV",
# ) -> dict:
#     """
#     Manual drill-down for one borrower, one quarter: how many distinct BDCs
#     (CIKs) actually hold a position in `name`, next to how many
#     compute_position_crowding()'s own exact-normalized-identifier matching
#     would report.

#     `name` is matched as a case-insensitive, literal substring against the
#     normalized identifier -- the caller supplies the borrower name; this
#     does not discover it (no entity resolution is attempted here either).

#     Returns {"n_holders": <actual distinct CIKs across every matching row>,
#     "matched_positions": <those same rows grouped the way
#     compute_position_crowding() groups them, one row per normalized
#     identifier, each with its own n_holders and total_fv>}.

#     `n_holders` is a plain set count over CIKs, indifferent to how each BDC
#     phrased the identifier, so it doesn't share compute_position_crowding()'s
#     undercount. `matched_positions` shows exactly why that undercount
#     happens: found via First Brands Group, 2025Q2 -- n_holders = 15, but
#     every one of `matched_positions`' 24 rows came back with its own
#     n_holders = 1, because no two of those 15 BDCs phrased the identifier
#     identically. compute_position_crowding() would report zero crowding for
#     a name 15 distinct managers actually held that quarter.
#     """
#     base = normalize_positions(df, identifier_col, fv_col)
#     needle = name.strip().upper()
#     matches = base.filter(
#         (pl.col(quarter_col) == quarter) & pl.col("_ident_norm").str.contains(needle, literal=True)
#     )

#     matched_positions = (
#         matches.group_by("_ident_norm")
#         .agg(
#             [
#                 pl.col(cik_col).n_unique().alias("n_holders"),
#                 pl.col(fv_col).sum().alias("total_fv"),
#                 pl.col(identifier_col).first().alias("sample_identifier"),
#             ]
#         )
#         .sort("n_holders", descending=True)
#     )

#     return {"n_holders": matches[cik_col].n_unique(), "matched_positions": matched_positions}


# =====================
# Plotting
# =====================

def _finish_quarter_plot(title, ylabel, save_path=None):
    """
    Shared tail for plot_crowding below: legend, axes, labels, grid, layout,
    optional save. Kept local rather than reusing index_construction.py's
    _finish_return_plot, since that helper requires an explicit `ylim`
    (every existing caller there passes one) -- this plot auto-scales to
    the metric's own range instead. plot_concentration is a dual-axis
    (twinx) layout with a merged legend, different enough that it isn't a
    second caller of this helper.
    """
    plt.xticks(rotation=45)
    plt.xlabel("Quarter")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        _Path(save_path).parent.mkdir(parents=True, exist_ok=True)  # output folder may not exist yet
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_concentration(
    conc_df: pl.DataFrame,
    quarter_col: str = "cal_q",
    figsize=(10, 5),
    save_path: str = None,
):
    """Effective number of BDCs (1/HHI) and top-10 AUM share, over time."""
    sorted_df = conc_df.sort(quarter_col)
    x = sorted_df[quarter_col].to_list()

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(
        x, sorted_df["effective_n_ciks"].to_list(), color="tab:blue",
        marker="o", label="Effective # of BDCs (1/HHI)",
    )
    ax1.set_ylabel("Effective # of BDCs", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(range(len(x)))
    ax1.set_xticklabels(x, rotation=45)
    ax1.set_xlabel("Quarter")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        x, (sorted_df["top10_share"] * 100).to_list(), color="tab:red",
        linestyle="--", marker="s", label="Top-10 AUM Share (%)",
    )
    ax2.set_ylabel("Top-10 AUM Share (%)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("BDC Market Concentration")
    fig.tight_layout()
    if save_path:
        _Path(save_path).parent.mkdir(parents=True, exist_ok=True)  # output folder may not exist yet
        fig.savefig(save_path, dpi=150)
    plt.show()


def plot_hhi_vs_equal_weight(
    conc_df: pl.DataFrame,
    quarter_col: str = "cal_q",
    figsize=(10, 5),
    save_path: str = None,
):
    """
    Actual HHI against the HHI an equal split across every active BDC would
    produce (1 / n_active_ciks), over time -- both on the same [0, 1] HHI
    scale, so unlike plot_concentration() (which pairs effective-N against a
    percentage share and needs a second axis) this is a single-axis plot.
    The gap between the two lines is the concentration an equal-weight
    market of the same size wouldn't have: actual HHI sitting well above
    1/n_active_ciks means a few managers dominate more than headcount alone
    would suggest; the two converging means AUM is closer to evenly spread
    across whoever is active that quarter.
    """
    sorted_df = conc_df.sort(quarter_col).with_columns(
        (1.0 / pl.col("n_active_ciks")).alias("equal_weight_hhi")
    )
    x = sorted_df[quarter_col].to_list()

    plt.figure(figsize=figsize)
    plt.plot(x, sorted_df["hhi"].to_list(), marker="o", label="Actual HHI")
    plt.plot(
        x, sorted_df["equal_weight_hhi"].to_list(), marker="s", linestyle="--",
        label="Equal-weight HHI (1 / active BDCs)",
    )
    _finish_quarter_plot("HHI vs. Equal-Weight Benchmark", "HHI", save_path)


def plot_crowding(
    crowd_summary: pl.DataFrame,
    quarter_col: str = "cal_q",
    figsize=(10, 5),
    save_path: str = None,
):
    """Share of total market AUM sitting in positions held by >= min_holders
    distinct BDCs simultaneously, over time."""
    sorted_df = crowd_summary.sort(quarter_col)

    plt.figure(figsize=figsize)
    plt.plot(
        sorted_df[quarter_col].to_list(), (sorted_df["crowded_aum_share"] * 100).to_list(),
        label="Crowded AUM Share (%)", marker="o",
    )
    _finish_quarter_plot(
        "Position Crowding Across BDCs", "Share of Market AUM (%)", save_path
    )


if __name__ == "__main__":
    from paths import CONCENTRATION_PLOTS_DIR, FINAL_DIR
    from index_construction import load_and_prepare_investment_data, compute_final_interest_rates

    CSV_PATH = FINAL_DIR / "FINAL_DATASET.csv"

    print("=== concentration_crowding.py: concentration / crowding across BDCs ===")
    df = load_and_prepare_investment_data(str(CSV_PATH))
    df = compute_final_interest_rates(df)
    print(f"rows after load + rate resolution: {df.height}")

    # --- concentration ---
    aum_by_cik_q = compute_cik_aum_by_quarter(df)
    conc = compute_market_concentration(aum_by_cik_q)
    print("\n--- AUM concentration across BDCs (HHI, top-N share) ---")
    print(conc)

    # --- crowding ---
    crowd_positions, crowd_summary = compute_position_crowding(df)
    print("\n--- position crowding across BDCs ---")
    print(crowd_summary)

    if crowd_summary.height:
        latest_q = crowd_summary["cal_q"].tail(1).item()  # already sorted by compute_position_crowding
        print(f"\n--- most-crowded names, {latest_q} ---")
        print(top_crowded_positions(crowd_positions, latest_q))

    plot_concentration(conc, save_path=CONCENTRATION_PLOTS_DIR / "concentration_hhi.png")
    plot_hhi_vs_equal_weight(conc, save_path=CONCENTRATION_PLOTS_DIR / "hhi_vs_equal_weight.png")
    plot_crowding(crowd_summary, save_path=CONCENTRATION_PLOTS_DIR / "crowding_aum_share.png")
