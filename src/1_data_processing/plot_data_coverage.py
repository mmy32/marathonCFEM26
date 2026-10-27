"""Data-coverage charts for the SEC download and iXBRL parsing steps.

Usage:
    python src/1_data_processing/plot_data_coverage.py

Reads:
    cache/all_metadata.csv               every 10-K / 10-Q the downloader found (step 2)
    data/ixbrl/combined/wide_all.csv     parsed filings (step 3); only the accession column is read

Writes to outputs/index_construction_plots/:
    filings_by_type.png    10-K / 10-Q filings per filing year: non-XBRL, XBRL (non-inline), inline XBRL
    parsing_coverage.png   per quarter (by report date), 2023Q1-2026Q2: inline-XBRL filings attempted,
                           filings successfully parsed, and the success rate
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import polars as pl

from paths import FINAL_DIR, INDEX_PLOTS_DIR, IXBRL_WIDE_ALL_CSV, METADATA_CSV, PARSED_DIR

Q_FROM, Q_TO = "2023Q1", "2026Q2"
GREY, GOLD, GREEN, PALE, RED = "#8c8c8c", "#c9a35c", "#2c5545", "#b7bcb0", "#9c3428"


def load_metadata() -> pd.DataFrame:
    m = pd.read_csv(METADATA_CSV, dtype=str)
    m["year"] = m["filingDate"].str[:4].astype(int)
    m["kind"] = "Non-XBRL"
    m.loc[m["isXBRL"] == "1", "kind"] = "XBRL (non-inline)"
    m.loc[m["isInlineXBRL"] == "1", "kind"] = "Inline XBRL"
    m["accession"] = m["accessionNumber"].str.replace("-", "", regex=False)
    return m


def plot_filings_by_type(m: pd.DataFrame, path) -> pd.DataFrame:
    kinds = ["Non-XBRL", "XBRL (non-inline)", "Inline XBRL"]
    t = m.groupby(["year", "kind"]).size().unstack(fill_value=0).reindex(columns=kinds, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = None
    for k, c in zip(kinds, [GREY, GOLD, GREEN]):
        ax.bar(t.index, t[k], bottom=bottom, color=c, label=k)
        bottom = t[k] if bottom is None else bottom + t[k]
    ax.axvline(2023, color=RED, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(2023.15, ax.get_ylim()[1] * 0.93, "2023Q1", color=RED)
    ax.set_title(f"Filings by Type, {t.index.min()}–{t.index.max()}", fontsize=13)
    ax.set_xlabel("Filing year")
    ax.set_ylabel("Number of 10-K / 10-Q filings")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return t


def plot_parsing_coverage(m: pd.DataFrame, path) -> pd.DataFrame:
    parsed = set(pl.scan_csv(IXBRL_WIDE_ALL_CSV, infer_schema_length=0).select("accession").unique().collect()["accession"])
    x = m[m["isInlineXBRL"] == "1"].copy()
    x["cal_q"] = pd.PeriodIndex(pd.to_datetime(x["reportDate"]), freq="Q").astype(str)
    x = x[(x["cal_q"] >= Q_FROM) & (x["cal_q"] <= Q_TO)]
    x["parsed"] = x["accession"].isin(parsed)
    t = x.groupby("cal_q").agg(attempted=("accession", "size"), parsed=("parsed", "sum"))
    t["success_rate"] = t["parsed"] / t["attempted"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    pos = range(len(t))
    ax.bar([p - 0.2 for p in pos], t["attempted"], width=0.4, color=PALE, label="Filings attempted")
    ax.bar([p + 0.2 for p in pos], t["parsed"], width=0.4, color=GREEN, label="Successfully parsed")
    ax.set_xticks(list(pos), t.index, rotation=45)
    ax.set_ylabel("Number of filings")
    ax2 = ax.twinx()
    ax2.plot(list(pos), t["success_rate"], color=RED, marker="o", label="Parsing success rate")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_ylabel("Parsing success rate")
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    ax.set_title(f"Parsing Coverage, {Q_FROM}–{Q_TO} (Inline-XBRL filings only)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return t


def par_retention(meta: pd.DataFrame, parsed_csv, final_csv, scale_cap: float = 10.0) -> pd.DataFrame:
    """How much principal (par) survives each step, summed over the quarter-ends Q_FROM..Q_TO.

    3. Clean and label: par in FINAL_DATASET.
    2. Read: 3 plus the par of parsed rows (data/parsed/ixbrl_clean.csv, one row per position)
       that cleaning dropped. Matched on (accession, context_id); USD-unit rows only. A dropped
       row whose par is more than `scale_cap` times its fair value is a raw scale error (a value
       1,000x too large), so its fair value stands in for its par.
    1. Collect: 2 plus an estimate for the 10-K/10-Q filings that were never read, each set to
       the same BDC's par in its nearest quarter in FINAL_DATASET. BDCs with no readable filing
       at all cannot be estimated and are counted separately.
    Raw parsed totals cannot be compared directly: their scale errors add about $3tn.
    """
    def acc(s):
        return s.str.replace("-", "", literal=True).cast(pl.Int64, strict=False)
    P = "InvestmentOwnedBalancePrincipalAmount"
    f = (pl.scan_csv(final_csv, infer_schema_length=0)
         .select(["cik", "accession", "context_id", "cal_q", f"{P}_normalized"]).collect()
         .with_columns(acc(pl.col("accession")).alias("acc"), pl.col("cik").cast(pl.Int64),
                       pl.col(f"{P}_normalized").cast(pl.Float64, strict=False).alias("par")).to_pandas())
    p = (pl.scan_csv(parsed_csv, infer_schema_length=0)
         .select(["accession", "context_id", "cal_q", P, f"{P}-unitRef", "InvestmentOwnedAtFairValue"]).collect()
         .with_columns(acc(pl.col("accession")).alias("acc"), pl.col(P).cast(pl.Float64, strict=False).alias("par"),
                       pl.col("InvestmentOwnedAtFairValue").cast(pl.Float64, strict=False).alias("fv")).to_pandas())
    p = p[(p["cal_q"] >= Q_FROM) & (p["cal_q"] <= Q_TO)]
    d = p.merge(f[["acc", "context_id"]].assign(kept=True), on=["acc", "context_id"], how="left")
    d = d[d["kept"].isna() & d[f"{P}-unitRef"].fillna("").str.lower().str.contains("usd") & (d["par"] > 0)]
    scale_error = d["fv"].notna() & (d["par"] > scale_cap * d["fv"].clip(lower=1))
    dropped = float(d["par"].where(~scale_error, d["fv"]).sum())

    m = meta[meta["form"].isin(["10-K", "10-Q"])].copy()
    m["cal_q"] = pd.PeriodIndex(pd.to_datetime(m["reportDate"]), freq="Q").astype(str)
    m = m[(m["cal_q"] >= Q_FROM) & (m["cal_q"] <= Q_TO)]
    m["acc"], m["cik"] = m["accession"].astype("int64"), m["cik"].astype(int)
    unread = m[~m["acc"].isin(set(p["acc"].dropna().astype("int64")))].copy()
    fq = f.groupby(["cik", "cal_q"])["par"].sum().reset_index()
    fq["qn"] = pd.PeriodIndex(fq["cal_q"], freq="Q").astype("int64")
    unread["qn"] = pd.PeriodIndex(unread["cal_q"], freq="Q").astype("int64")
    by_cik = {c: g for c, g in fq.groupby("cik")}
    unread["est_par"] = [by_cik[c].loc[(by_cik[c]["qn"] - q).abs().idxmin(), "par"] if c in by_cik else np.nan
                         for c, q in zip(unread["cik"], unread["qn"])]

    kept = float(f["par"].sum())
    read = kept + dropped
    collected = read + float(unread["est_par"].sum())
    no_history = unread["est_par"].isna()
    return pd.DataFrame([
        dict(step="1. Collect (estimated)", filings=len(m), par_usd=collected, share_kept=kept / collected,
             note=f"{len(unread)} unread filings estimated from the BDC's nearest quarter; {int(no_history.sum())} filings "
                  f"from {unread.loc[no_history, 'cik'].nunique()} BDCs with no readable filing are not estimable"),
        dict(step="2. Read", filings=int(p["acc"].nunique()), par_usd=read, share_kept=kept / read,
             note=f"{int(scale_error.sum())} dropped rows with par > {scale_cap:.0f}x fair value counted at fair value"),
        dict(step="3. Clean and label", filings=int(f["acc"].nunique()), par_usd=kept, share_kept=1.0, note="FINAL_DATASET"),
    ])


if __name__ == "__main__":
    INDEX_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    by_type = plot_filings_by_type(meta, INDEX_PLOTS_DIR / "filings_by_type.png")
    print("filings by year and type (last 5 years):"); print(by_type.tail(5).to_string())
    cov = plot_parsing_coverage(meta, INDEX_PLOTS_DIR / "parsing_coverage.png")
    print("\nparsing coverage:"); print(cov.to_string(float_format=lambda v: f"{v:.1%}"))
    print(f"\ntotal: {int(cov['attempted'].sum()):,} attempted, {int(cov['parsed'].sum()):,} parsed")
    ret = par_retention(meta, PARSED_DIR / "ixbrl_clean.csv", FINAL_DIR / "FINAL_DATASET.csv")
    ret.to_csv(INDEX_PLOTS_DIR / "par_retention.csv", index=False)
    print("\nprincipal (par) kept at each step:"); print(ret.to_string(index=False))
