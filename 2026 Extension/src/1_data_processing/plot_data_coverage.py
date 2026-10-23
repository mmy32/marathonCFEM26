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
import pandas as pd
import polars as pl

from paths import INDEX_PLOTS_DIR, IXBRL_WIDE_ALL_CSV, METADATA_CSV

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


if __name__ == "__main__":
    INDEX_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    by_type = plot_filings_by_type(meta, INDEX_PLOTS_DIR / "filings_by_type.png")
    print("filings by year and type (last 5 years):"); print(by_type.tail(5).to_string())
    cov = plot_parsing_coverage(meta, INDEX_PLOTS_DIR / "parsing_coverage.png")
    print("\nparsing coverage:"); print(cov.to_string(float_format=lambda v: f"{v:.1%}"))
    print(f"\ntotal: {int(cov['attempted'].sum()):,} attempted, {int(cov['parsed'].sum()):,} parsed")
