"""Crowding per borrower and quarter: number of lender (manager) families, lender HHI, the largest
family's share and the change in families over four quarters. Used by the risk monitor's crowding
sign. See docs/7_concentration_crowding.md."""

import polars as pl


def compute_borrower_crowding_metrics(
    df: pl.DataFrame,
    quarter_col: str = "cal_q",
    family_col: str = "family",
    borrower_col: str = "borrower_canonical",
    par_col: str = "PAR",
    fv_col: str = "FV",
    lag_quarters: int = 4,
) -> pl.DataFrame:
    """
    Borrower-level crowding metrics (Risk Dimensions doc, C1-C4), one row per
    (quarter, borrower), on funded debt (PAR > 0 and FV > 0):

      - C1 `n_families`: manager families holding the borrower (n_ciks: BDCs).
      - C2 `lender_hhi`: sum of squared family shares of BDC-held par
        (1 = one family holds it all).
      - C3 `top_share`: the largest family's share of par.
      - C4 `d_families_4q`: C1 now minus C1 `lag_quarters` earlier. A borrower
        absent then counts as 0 families; null when that quarter is before the
        panel starts.
      - `log_par`: log of par held by all BDCs, the size control.
      - `C1_pct` ... `C4_pct`: within-quarter percentiles.
    """
    q_sort = (pl.col(quarter_col).str.slice(0, 4).cast(pl.Int32) * 4
              + pl.col(quarter_col).str.slice(-1).cast(pl.Int32))
    base = (
        df.filter((pl.col(par_col) > 0) & (pl.col(fv_col) > 0))
        .with_columns(pl.col(borrower_col).cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("_b"))
        .filter(pl.col("_b").is_not_null() & (pl.col("_b") != ""))
    )
    by_fam = base.group_by([quarter_col, "_b", family_col], maintain_order=True).agg(
        pl.col(par_col).sum().alias("_par"), pl.col("cik").n_unique().alias("_ciks"))
    out = (
        by_fam.with_columns((pl.col("_par") / pl.col("_par").sum().over([quarter_col, "_b"])).alias("_sh"))
        .group_by([quarter_col, "_b"], maintain_order=True)
        .agg([
            pl.len().cast(pl.Int64).alias("n_families"),       # signed, so C4 can go negative
            pl.col("_ciks").sum().cast(pl.Int64).alias("n_ciks"),
            (pl.col("_sh") ** 2).sum().alias("lender_hhi"),
            pl.col("_sh").max().alias("top_share"),
            pl.col("_par").sum().alias("total_par"),
        ])
        .with_columns([pl.col("total_par").log().alias("log_par"), q_sort.alias("_q")])
        .rename({"_b": "borrower"})
    )
    first_q = out["_q"].min()
    lagged = out.select(["borrower", (pl.col("_q") + lag_quarters).alias("_q"),
                         pl.col("n_families").alias("_nf_lag")])
    out = (
        out.join(lagged, on=["borrower", "_q"], how="left")
        .with_columns(
            pl.when(pl.col("_q") - lag_quarters < first_q).then(None)
            .otherwise(pl.col("n_families") - pl.col("_nf_lag").fill_null(0))
            .alias("d_families_4q")
        )
        .drop("_nf_lag")
    )
    pct = lambda c: (pl.col(c).rank("average").over(quarter_col)
                     / pl.col(c).count().over(quarter_col) * 100)
    return (
        out.with_columns([pct("n_families").alias("C1_pct"), pct("lender_hhi").alias("C2_pct"),
                          pct("top_share").alias("C3_pct"), pct("d_families_4q").alias("C4_pct")])
        .sort(["_q", "n_families"], descending=[False, True])
        .drop("_q")
    )
