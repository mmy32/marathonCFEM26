"""PIK transitions per loan (classify_loan_pik_transitions): PIK share of the coupon, switches to PIK
(the P2 toggle), PIK acquired after origination. Used by the risk monitor's PIK sign (slide 14)."""

import polars as pl

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py
from paths import add_code_dirs_to_path
add_code_dirs_to_path()  # index_construction lives in src/2_index_construction/
from index_construction import safe_div  # noqa: E402


def quarter_sort_expr(quarter_col: str = "cal_q") -> pl.Expr:
    """Integer sorting key for YYYYQ# quarterly strings."""
    y = pl.col(quarter_col).str.slice(0, 4).cast(pl.Int32)
    q = pl.col(quarter_col).str.slice(-1).cast(pl.Int32)
    return (y * 4 + q).alias("qsort")


TRANCHE_ID = "tranche_id"   # a BDC's chain of one loan, see tranches.link_tranche_chains
MIN_PIK_RATE = 0.0025        # PIK under 25 bp is treated as none (rounding in total - cash)
MIN_SHARE_RISE = 0.10        # P2 on a loan already paying PIK: PIK share of the coupon up >= 10 points (smaller moves are total - cash rounding)


def classify_loan_pik_transitions(
    df: pl.DataFrame,
    quarter_col: str = "cal_q",
    cik_col: str = "cik",
    id_col: str = TRANCHE_ID,
    final_pik_col: str = "PIK_Final",
    ir_col: str = "IR_Final",
    par_col: str = "PAR",
    min_pik_rate: float = MIN_PIK_RATE,
    min_share_rise: float = MIN_SHARE_RISE,
    origination_col: str = "pik_at_origination",
) -> pl.DataFrame:
    """
    Classify PIK transitions along each loan's chain (cik, id_col; run on the output of
    index_construction.compute_position_level_flows over the tranche panel, so a toggle is
    on the same loan, not on a borrower's other tranche):

      - 'contractual_pik': PIK in the chain's first quarter, and the loan had PIK when it
                           first appeared in any BDC (P3), or that is unknown.
      - 'acquired_pik': PIK in the chain's first quarter, but the loan was cash-pay when it
                        first appeared in another BDC (P3 = False): PIK added later.
      - 'distress_amendment_pik': P2, the PIK toggle. A seasoned loan whose PIK appears, or
                                  whose PIK share of the coupon rises by >= min_share_rise.
      - 'stable_cash', 'cured_pik', 'seasoned_pik_continuation': as named.

    PIK below min_pik_rate counts as none. Adds q_incept, PAR_incept, PIK_prev, PAR_prev,
    delta_pik, pik_share (P1, PIK / total coupon), pik_share_prev, P1_pct (within-quarter
    percentile of pik_share), pik_acquired (PIK now, P3 = False), pik_transition_state and
    is_distress_amendment (P2).
    """
    group_cols = [cik_col, id_col]
    has_orig = origination_col in df.columns

    share = safe_div(pl.col(final_pik_col).fill_null(0), pl.col(ir_col)).clip(0.0, 1.0)
    df_tracked = (
        df.sort([cik_col, id_col, quarter_sort_expr(quarter_col)])
        .with_columns(share.alias("pik_share"))
        .with_columns([
            pl.col(quarter_col).first().over(group_cols).alias("q_incept"),
            pl.col(par_col).first().over(group_cols).alias("PAR_incept"),
            pl.col(final_pik_col).shift(1).over(group_cols).alias("PIK_prev"),
            pl.col(par_col).shift(1).over(group_cols).alias("PAR_prev"),
            pl.col("pik_share").shift(1).over(group_cols).alias("pik_share_prev"),
        ]).with_columns([
            (pl.col(quarter_col) == pl.col("q_incept")).alias("is_inception"),
            (pl.col(final_pik_col).fill_null(0) - pl.col("PIK_prev").fill_null(0)).alias("delta_pik"),
            (pl.col(final_pik_col).fill_null(0) >= min_pik_rate).alias("curr_has_pik"),
            (pl.col("PIK_prev").fill_null(0) >= min_pik_rate).alias("prev_has_pik"),
            (pl.col("pik_share").rank("average").over(quarter_col)
             / pl.len().over(quarter_col) * 100).alias("P1_pct"),
        ])
    )

    is_incept = pl.col("is_inception")
    curr_pik = pl.col("curr_has_pik")
    prev_pik = pl.col("prev_has_pik")
    share_up = (pl.col("pik_share") - pl.col("pik_share_prev").fill_null(0)) >= min_share_rise
    cash_at_orig = (pl.col(origination_col) == False).fill_null(False) if has_orig else pl.lit(False)
    toggle = ~is_incept & curr_pik & (~prev_pik | share_up)

    transition_expr = (
        pl.when(is_incept & curr_pik & cash_at_orig).then(pl.lit("acquired_pik"))
        .when(is_incept & curr_pik).then(pl.lit("contractual_pik"))
        .when(is_incept).then(pl.lit("stable_cash"))
        .when(toggle).then(pl.lit("distress_amendment_pik"))
        .when(prev_pik & ~curr_pik).then(pl.lit("cured_pik"))
        .when(~curr_pik).then(pl.lit("stable_cash"))
        .otherwise(pl.lit("seasoned_pik_continuation"))
        .alias("pik_transition_state")
    )

    return (
        df_tracked.with_columns([transition_expr, toggle.alias("is_distress_amendment")])
        # acquired PIK: cash-pay at origination (P3 = False), or this chain toggled to PIK earlier
        .with_columns(
            (curr_pik & (cash_at_orig
                         | pl.col("is_distress_amendment").cast(pl.Int8).cum_max().over(group_cols).cast(pl.Boolean)))
            .alias("pik_acquired")
        )
    )
