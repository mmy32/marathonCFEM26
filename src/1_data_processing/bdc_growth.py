"""BDC fair value and number of active BDCs per quarter (slide 4), and BDCs within US and global private
credit (slide 3; the ACC and Federal Reserve totals are typed in below). Reads FINAL_DATASET.csv;
writes bdc_growth.png/.csv and private_credit_growth.png/.csv to outputs/eda_plots/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paths import EDA_PLOTS_DIR, FINAL_DIR

FV = "InvestmentOwnedAtFairValue_normalized"
START, END = "2023Q1", "2026Q2"
BLUE = "#2a78d6"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LIGHT_BLUE = "#c9ddf5"
MID_BLUE = "#86b6ef"

# Global private credit AUM, year-end, US$ trillions: the Alternative Credit Council (AIMA) and
# Houlihan Lokey, "Financing the Economy 2025" (Dec 2025: US$3.0tn end-2023, US$3.5tn end-2024)
# and "Financing the Economy 2026" (Oct 2026, p.6 Fig 1: US$4.1tn end-2025). Estimates: survey
# respondents' committed capital (deployed and dry powder) plus leverage facilities, grossed up
# with Preqin data for managers outside the survey. Not the same measure as BDC fair value, so
# the BDC share is approximate; the report itself puts BDCs at ~US$500bn, ~12% (p.39).
PRIVATE_CREDIT_AUM_TN = {2023: 3.0, 2024: 3.5, 2025: 4.1}

# US private credit loans outstanding, Q4, US$ trillions: Federal Reserve Financial Accounts of the
# United States, instrument table F.4.4 "Private credit loans" (series FL893167205; FRED
# BOGZ1FL893167205Q: 1,042,623 / 1,253,643 / 1,493,510 US$ mn), introduced in Brinkman et al.,
# "Introducing Private Credit into the Financial Accounts of the United States", FEDS Notes,
# 16 Oct 2026. Loans to US borrowers held by BDCs, private debt funds, interval/tender offer funds,
# middle-market CLOs and foreign-domiciled funds. A loans-outstanding measure, not AUM, so it is not
# strictly a subset of the ACC figure; the bars nest them for scale only.
US_PRIVATE_CREDIT_TN = {2023: 1.042623, 2024: 1.253643, 2025: 1.493510}


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
    top.set_ylabel("$ billions", color=MUTED, fontsize=12.5)
    first, last = t["aum_usd_bn"].iloc[0], t["aum_usd_bn"].iloc[-1]
    top.set_title(f"Total fair value held by BDCs: \\${first:.0f}B to \\${last:.0f}B ({last / first:.1f}x)",
                  loc="left", color=INK, fontsize=15.5, pad=12)
    for i in (0, len(t) - 1):
        top.text(i, t["aum_usd_bn"].iloc[i], f"\\${t['aum_usd_bn'].iloc[i]:.0f}B", ha="center", va="bottom", fontsize=11.5, color=MUTED)

    bot.bar(x, t["active_bdcs"], width=0.7, color=BLUE, edgecolor=SURFACE, linewidth=1.2)
    bot.set_ylabel("BDCs", color=MUTED, fontsize=12.5)
    bot.set_ylim(0, t["active_bdcs"].max() * 1.2)
    n0, n1 = t["active_bdcs"].iloc[0], t["active_bdcs"].iloc[-1]
    bot.set_title(f"Active BDCs (reporting at least one investment): {n0} to {n1}", loc="left", color=INK, fontsize=15.5, pad=12)
    for i in (0, len(t) - 1):
        bot.annotate(f"{t['active_bdcs'].iloc[i]}", (i, t["active_bdcs"].iloc[i]), textcoords="offset points",
                     xytext=(0, 2), ha="center", fontsize=11.5, color=MUTED)

    bot.set_xticks(list(x), t.index)
    plt.setp(bot.get_xticklabels(), rotation=45, ha="right")
    for ax in (top, bot):
        style(ax)
        ax.tick_params(labelsize=12)

    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def private_credit_vs_bdcs(t):
    """Year-end global private credit AUM (ACC) next to year-end BDC fair value (this panel)."""
    rows = []
    for year, aum in PRIVATE_CREDIT_AUM_TN.items():
        q = f"{year}Q4"
        if q in t.index:
            bdc = t.loc[q, "aum_usd_bn"] / 1000
            us = US_PRIVATE_CREDIT_TN[year]
            rows.append({"year": year, "private_credit_tn": aum, "us_private_credit_tn": us, "bdc_fv_tn": bdc,
                         "bdc_share": bdc / aum, "bdc_share_of_us": bdc / us})
    out = pd.DataFrame(rows)
    out["private_credit_growth"] = out["private_credit_tn"].pct_change()
    out["us_private_credit_growth"] = out["us_private_credit_tn"].pct_change()
    out["bdc_growth"] = out["bdc_fv_tn"].pct_change()
    return out


def plot_private_credit(g, path):
    """One bar per year-end: global private credit (light), US private credit (mid), BDCs (blue)."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = range(len(g))
    ax.bar(x, g["private_credit_tn"], width=0.62, color=LIGHT_BLUE, edgecolor=SURFACE, linewidth=1.5,
           label="Global private credit (ACC estimate)")
    ax.bar(x, g["us_private_credit_tn"], width=0.62, color=MID_BLUE, edgecolor=SURFACE, linewidth=1.5,
           label="US private credit loans (Federal Reserve)")
    ax.bar(x, g["bdc_fv_tn"], width=0.62, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
           label="BDCs (our dataset)")
    for i, r in g.iterrows():
        ax.text(i, r["private_credit_tn"] + 0.06, f"\\${r['private_credit_tn']:.1f}tn", ha="center", va="bottom",
                fontsize=16, color=INK)
        if pd.notna(r["private_credit_growth"]):
            ax.text(i, r["private_credit_tn"] - 0.1, f"+{r['private_credit_growth']:.0%}", ha="center", va="top",
                    fontsize=13, color=MUTED)
        ax.text(i, r["us_private_credit_tn"] + 0.06, f"US \\${r['us_private_credit_tn']:.2f}tn"
                + (f", +{r['us_private_credit_growth']:.0%}" if pd.notna(r["us_private_credit_growth"]) else ""),
                ha="center", va="bottom", fontsize=13, color=INK)
        ax.text(i, r["bdc_fv_tn"] + 0.04, f"BDCs \\${r['bdc_fv_tn'] * 1000:.0f}bn"
                + (f", +{r['bdc_growth']:.0%}" if pd.notna(r["bdc_growth"]) else "")
                + f"\n{r['bdc_share']:.0%} of global\n{r['bdc_share_of_us']:.0%} of US",
                ha="center", va="bottom", fontsize=13, color=INK, weight="bold")
    ax.set_xticks(list(x), [f"End {y}" for y in g["year"]], fontsize=14, color=INK)
    ax.set_ylim(0, g["private_credit_tn"].max() * 1.18)
    ax.yaxis.set_major_formatter(lambda v, _: f"\\${v:.0f}tn")
    style(ax)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=14, colors=INK)
    ax.legend(frameon=False, fontsize=12.5, loc="upper left", labelcolor=INK)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08)
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else EDA_PLOTS_DIR / "bdc_growth.png"
    out.parent.mkdir(parents=True, exist_ok=True)  # output folder may not exist yet
    t = quarterly_totals()
    t.round(2).to_csv(out.with_suffix(".csv"))
    plot(t, out)
    print(t.round(1).to_string())
    g = private_credit_vs_bdcs(t)
    g.round(4).to_csv(out.with_name("private_credit_growth.csv"), index=False)
    plot_private_credit(g, out.with_name("private_credit_growth.png"))
    print(g.round(3).to_string(index=False))
