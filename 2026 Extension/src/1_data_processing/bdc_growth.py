"""BDC universe growth: total fair value held (AUM proxy) and number of active BDCs per quarter.

Usage:
    python bdc_growth.py [out_path]

Reads data/final_dataset/FINAL_DATASET.csv. A BDC counts as active in a quarter if it reports
at least one investment that quarter. AUM is the sum of InvestmentOwnedAtFairValue_normalized
over every row (all instruments, not only loans) in the quarter.

Writes out_path (default: bdc_growth.png) and a CSV of the same numbers next to it.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paths import EDA_PLOTS_DIR, FINAL_DIR

FV = "InvestmentOwnedAtFairValue_normalized"
START, END = "2023Q1", "2026Q2"
BLUE = "#2a78d6"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def quarterly_totals():
    d = pd.read_csv(FINAL_DIR / "FINAL_DATASET.csv", usecols=["cik", "cal_q", FV], low_memory=False)
    d = d[d["cal_q"].between(START, END)]
    out = d.groupby("cal_q").agg(aum_usd_bn=(FV, "sum"), active_bdcs=("cik", "nunique"))
    out["aum_usd_bn"] /= 1e9
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(t, path):
    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw=dict(height_ratios=[3, 2], hspace=0.35))
    x = range(len(t))

    top.bar(x, t["aum_usd_bn"], width=0.7, color=BLUE, edgecolor=SURFACE, linewidth=1.2)
    top.set_ylabel("$ billions", color=MUTED, fontsize=9)
    first, last = t["aum_usd_bn"].iloc[0], t["aum_usd_bn"].iloc[-1]
    top.set_title(f"Total fair value held by BDCs: \\${first:.0f}B to \\${last:.0f}B ({last / first:.1f}x)",
                  loc="left", color=INK, fontsize=12, pad=10)
    for i in (0, len(t) - 1):
        top.text(i, t["aum_usd_bn"].iloc[i], f"\\${t['aum_usd_bn'].iloc[i]:.0f}B", ha="center", va="bottom", fontsize=8, color=MUTED)

    bot.bar(x, t["active_bdcs"], width=0.7, color=BLUE, edgecolor=SURFACE, linewidth=1.2)
    bot.set_ylabel("BDCs", color=MUTED, fontsize=9)
    bot.set_ylim(0, t["active_bdcs"].max() * 1.2)
    n0, n1 = t["active_bdcs"].iloc[0], t["active_bdcs"].iloc[-1]
    bot.set_title(f"Active BDCs (reporting at least one investment): {n0} to {n1}", loc="left", color=INK, fontsize=12, pad=10)
    for i in (0, len(t) - 1):
        bot.annotate(f"{t['active_bdcs'].iloc[i]}", (i, t["active_bdcs"].iloc[i]), textcoords="offset points",
                     xytext=(0, 2), ha="center", fontsize=8, color=MUTED)

    bot.set_xticks(list(x), t.index)
    plt.setp(bot.get_xticklabels(), rotation=45, ha="right")
    for ax in (top, bot):
        style(ax)

    fig.patch.set_facecolor(SURFACE)
    fig.text(0.01, 0.01, "Source: BDC 10-K/10-Q iXBRL schedules of investments (FINAL_DATASET.csv), all instruments at fair value.",
             fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else EDA_PLOTS_DIR / "bdc_growth.png"
    out.parent.mkdir(parents=True, exist_ok=True)  # output folder may not exist yet
    t = quarterly_totals()
    t.round(2).to_csv(out.with_suffix(".csv"))
    plot(t, out)
    print(t.round(1).to_string())
