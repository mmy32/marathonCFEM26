#!/usr/bin/env python3
"""How much of each quarter's markdown dollars the worst 1/5/10% of borrowers cause, and how
concentrated non-accrual dollars are (slide 5); the distribution of quarterly value changes
(slide 9) and the market-wide move (slide 23). Reads FINAL_DATASET.csv; writes tail_concentration*.csv
and the four charts to outputs/risk_monitor/distress/. About 5 minutes."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "1_data_processing"))
from paths import FINAL_DIR, RISK_MONITOR_DIR, add_code_dirs_to_path  # noqa: E402
add_code_dirs_to_path()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from monitor_panel import build_monitor_panel  # noqa: E402
import distress_event_study as D  # noqa: E402
from charts import SURFACE, INK, MUTED, GRID, BLUE  # noqa: E402

warnings.filterwarnings("ignore")

TAILS = (0.01, 0.05, 0.10)
MIN_FV_PREV = 1e6          # same floor as the stress-event rule
VALUE_GREY = "#b9b7b0"
CASE_ORANGE = "#e8743b"
TAIL_COLORS = {0.01: "#104281", 0.05: "#2a78d6", 0.10: "#86b6ef"}  # ordinal: darkest = most extreme tail
CASES = {"FIRST BRANDS GROUP": "2025Q3", "ATHLETICO MANAGEMENT": "2026Q1", "MEDALLIA": "2026Q1"}


def eligible(b: pd.DataFrame) -> pd.DataFrame:
    """Borrower-quarters the stress rule ranks: at least $1M of prior fair value and a valuation return."""
    return b[(b["fv_prev"] >= MIN_FV_PREV) & b["val_ret"].notna()].copy()


def market_move(b: pd.DataFrame) -> pd.Series:
    """Market-wide change in value per quarter, %: valuation returns weighted by prior fair value."""
    return b.groupby("cal_q").apply(lambda d: (d["val_ret"] * d["fv_prev"]).sum() / d["fv_prev"].sum()) * 100


def tail_shares(b: pd.DataFrame) -> pd.DataFrame:
    """Share of markdown dollars and of prior fair value in the worst p% of borrowers, per quarter and pooled."""
    b = eligible(b)
    b["chg"] = b["val_ret"] * b["capital_at_risk"]   # flow-adjusted change in value, $
    b["loss"] = -b["chg"].clip(upper=0)
    b["pct"] = b.groupby("cal_q")["val_ret"].rank(pct=True, method="first")
    rows = []
    for q, g in list(b.groupby("cal_q")) + [("pooled", b)]:
        for p in TAILS:
            t = g[g["pct"] <= p]
            rows.append({"cal_q": q, "tail": p, "borrowers": len(t), "all_borrowers": len(g),
                         "loss_share": t["loss"].sum() / g["loss"].sum(),
                         "fv_share": t["fv_prev"].sum() / g["fv_prev"].sum(),
                         "tail_loss_bn": t["loss"].sum() / 1e9, "all_loss_bn": g["loss"].sum() / 1e9})
    return pd.DataFrame(rows)


def nonaccrual_shares(b: pd.DataFrame, quarter: str) -> pd.DataFrame:
    """Share of non-accrual cost held by the top 1/5/10% of borrowers (ranked by non-accrual cost) in one quarter."""
    L = b[(b["cal_q"] == quarter) & (b["cost"] > 0)].copy()
    L["na_cost"] = (L["na_cost_share"] * L["cost"]).fillna(0)
    L = L.sort_values("na_cost", ascending=False)
    rows = []
    for p in TAILS:
        k = int(round(len(L) * p))
        top = L.head(k)
        rows.append({"cal_q": quarter, "tail": p, "borrowers": k, "all_borrowers": len(L),
                     "borrowers_with_any_na": int((L["na_cost"] > 0).sum()),
                     "na_bn": L["na_cost"].sum() / 1e9,
                     "na_share": top["na_cost"].sum() / L["na_cost"].sum(),
                     "cost_share": top["cost"].sum() / L["cost"].sum()})
    return pd.DataFrame(rows)



def _style(ax, n: int) -> None:
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=15, length=0)
    ax.set_xlim(-0.6, n - 0.4)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")



def case_marks(b: pd.DataFrame) -> pd.DataFrame:
    """For each case study in CASES: its change in value, markdown $, share of that quarter's markdown $,
    and which tail slice it falls in (same filter and ranking as tail_shares)."""
    b = eligible(b)
    b["loss"] = -(b["val_ret"] * b["capital_at_risk"]).clip(upper=0)
    b["pct"] = b.groupby("cal_q")["val_ret"].rank(pct=True, method="first")
    tot = b.groupby("cal_q")["loss"].sum()
    rows = []
    for name, q in CASES.items():
        r = b[(b["borrower"] == name) & (b["cal_q"] == q)]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append({"borrower": name, "cal_q": q, "val_ret": r["val_ret"], "loss": r["loss"],
                     "loss_share": r["loss"] / tot[q], "pct": r["pct"],
                     "tail": next((p for p in TAILS if r["pct"] <= p), None)})
    return pd.DataFrame(rows)


def chart_timeseries(t: pd.DataFrame, na: pd.DataFrame, path: Path, cases: pd.DataFrame | None = None) -> None:
    """Quarter by quarter, stacked: % of markdown dollars caused by the worst 1%, the next 4% and
    the next 5% of borrowers (bar top = worst 10%), with the non-accrual concentration as a note.
    `cases` (from case_marks) adds a numbered badge on the slice each case study falls in."""
    t = t[t["cal_q"] != "pooled"]
    qs = sorted(t["cal_q"].unique())
    x = list(range(len(qs)))
    n_cases = 0 if cases is None else len(cases)
    fig, ax = plt.subplots(figsize=(13, 8 + 0.45 * n_cases), facecolor=SURFACE)

    # The tails are nested (worst 1% is inside the worst 5%), so stack the increments:
    # worst 1%, next 4% (1-5%), next 5% (5-10%). The top of each bar is the worst 10%.
    cum = {p: t[t["tail"] == p].set_index("cal_q")["loss_share"].reindex(qs) * 100 for p in TAILS}
    bottom = pd.Series(0.0, index=qs)
    prev = 0.0
    for p in TAILS:
        seg = cum[p] - bottom
        name = f"Worst {p:.0%}" if prev == 0 else f"Next {p - prev:.0%} ({prev:.0%}–{p:.0%})"
        ax.bar(x, seg, bottom=bottom, width=0.72, color=TAIL_COLORS[p], label=name, edgecolor=SURFACE, lw=0.8)
        for i, (s, b0) in enumerate(zip(seg, bottom)):
            if s >= 6:
                ax.text(i, b0 + s / 2, f"{s:.0f}", ha="center", va="center", fontsize=14,
                        color="white" if p < 0.10 else INK)
        bottom = bottom + seg
        prev = p
    for i, v in enumerate(cum[TAILS[-1]]):
        ax.text(i, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=15, color=INK, weight="bold")

    # Case studies: a numbered badge on the right edge of the slice each one falls in, keyed below.
    H = fig.get_figheight()
    key = []
    if n_cases:
        slice_name = {0.01: "worst 1%", 0.05: "next 4%", 0.10: "next 5%"}
        for k, r in enumerate(cases.itertuples(), start=1):
            i = qs.index(r.cal_q)
            lo = 0.0 if r.tail == TAILS[0] else cum[TAILS[TAILS.index(r.tail) - 1]][r.cal_q]
            y = (lo + cum[r.tail][r.cal_q]) / 2
            ax.text(i + 0.36, y, str(k), ha="center", va="center", fontsize=13, color="white", weight="bold",
                    zorder=5, bbox=dict(boxstyle="circle,pad=0.3", fc=CASE_ORANGE, ec=SURFACE, lw=1.5))
            label = r.borrower.title().replace(" Group", "").replace(" Management", "")
            m = r.loss / 1e6
            dollars = f"${m:,.0f}M" if m >= 10 else f"${m:.1f}M"
            share = f"{r.loss_share:.1%}" if r.loss_share >= 0.001 else "<0.1%"
            key.append(f"{k}  {label} ({r.cal_q}): {r.val_ret:.0%} in value, {dollars} = "
                       f"{share} of that quarter's markdown dollars ({slice_name[r.tail]})")

    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xticks(x, qs)
    ax.legend(frameon=False, fontsize=15, loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=3,
              labelcolor=INK, handlelength=1.2)
    _style(ax, len(qs))

    fig.text(0.02, 1 - 0.2 / H, "A few borrowers cause most markdowns, every quarter",
             ha="left", va="top", fontsize=22, color=INK, weight="bold")
    fig.text(0.02, 1 - 0.64 / H, "% of all markdown dollars caused by the worst borrowers (bar top = worst 10%).\n"
             "Borrowers with at least $1M of BDC loans, ranked each quarter by change in value",
             ha="left", va="top", fontsize=15, color=MUTED, linespacing=1.4)
    # Bottom notes, in inches from the bottom: case-study key, then the non-accrual line.
    line = 0.42
    for k, s in enumerate(key):
        fig.text(0.02, (0.16 + line * (len(key) - k)) / H, s, ha="left", va="bottom", fontsize=13.5, color=INK)
        fig.text(0.02, (0.16 + line * (len(key) - k)) / H, s.split("  ")[0], ha="left", va="bottom",
                 fontsize=13.5, color=CASE_ORANGE, weight="bold")
    # Non-accrual concentration barely moves, so it gets one sentence instead of a panel.
    na1 = na[na["tail"] == 0.01].set_index("cal_q").reindex(qs)["na_share"] * 100
    fig.text(0.02, 0.16 / H, f"Non-accruals are steadier: the top 1% of borrowers hold "
             f"{na1.min():.0f}–{na1.max():.0f}% of non-accrual dollars every quarter",
             ha="left", va="bottom", fontsize=15, color=INK)
    fig.subplots_adjust(left=0.08, right=0.98, top=1 - 1.92 / H,
                        bottom=(0.16 + line * (len(key) + 1) + 1.0) / H)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def chart_markdown_total(t: pd.DataFrame, path: Path) -> None:
    """Total markdown dollars per quarter, all borrowers with at least $1M of prior fair value.
    Gross: each borrower's loss is counted, borrowers that gained value do not offset it."""
    t = t[t["cal_q"] != "pooled"]
    qs = sorted(t["cal_q"].unique())
    x = list(range(len(qs)))
    tot = t[t["tail"] == 0.01].set_index("cal_q")["all_loss_bn"].reindex(qs)
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=SURFACE)
    ax.bar(x, tot, width=0.72, color=VALUE_GREY)
    for i, v in enumerate(tot):
        ax.text(i, v + 0.08, f"${v:.1f}", ha="center", va="bottom", fontsize=15, color=INK)
    ax.set_ylim(0, tot.max() * 1.18)
    ax.set_yticks([0, 2, 4, 6])
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:.0f}bn")
    ax.set_xticks(x, qs)
    _style(ax, len(qs))
    fig.text(0.02, 0.965, "Total markdown dollars, all borrowers ($bn)",
             ha="left", va="top", fontsize=22, color=INK, weight="bold")
    fig.text(0.02, 0.885, "Sum of each borrower's loss in value over the quarter (net of draws, paydowns and PIK);\n"
             "borrowers that gained value do not offset it. Borrowers with at least $1M of BDC loans",
             ha="left", va="top", fontsize=15, color=MUTED, linespacing=1.4)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.72, bottom=0.17)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)



def chart_distribution(b: pd.DataFrame, path: Path) -> None:
    """Pooled distribution of each borrower's quarterly change in value, with the worst 1/3/5% cutoffs
    and the named case studies marked: why the stress event is the worst 1%."""
    import numpy as np
    b = eligible(b)
    r = b["val_ret"] * 100
    cut = {p: r.quantile(p) for p in (0.01, 0.03, 0.05)}
    x = r.clip(-100, 20)                            # gains above +20% share the last bar
    bins = np.arange(-100, 21, 1)
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    counts, edges = np.histogram(x, bins=bins)
    colors = [BLUE if e < cut[0.01] else VALUE_GREY for e in edges[:-1]]
    ax.bar(edges[:-1], counts, width=1, align="edge", color=colors, edgecolor=SURFACE, linewidth=0.3)
    ax.set_yscale("log")
    ax.set_ylim(0.8, counts.max() * 20)
    for p, c in cut.items():
        ax.axvline(c, color=INK if p == 0.01 else MUTED, lw=1.4 if p == 0.01 else 1, ls="-" if p == 0.01 else "--")
        right = p == 0.05                           # the 5% label sits right of its line, above the peak
        ax.text(c + 0.8 if right else c - 0.8, counts.max() * (12 if p == 0.01 else 2.5 if p == 0.03 else 5),
                f"Worst {p:.0%}\n{c:.0f}%", ha="left" if right else "right", va="center", fontsize=12.5,
                color=INK if p == 0.01 else MUTED, weight="bold" if p == 0.01 else "normal")
    # the case studies, placed at their worst quarter. Only x carries data: all markers share one
    # height so it does not read as a count on the y-axis
    y = counts.max() * 0.08
    for name, q in CASES.items():
        v = r[(b["borrower"] == name) & (b["cal_q"] == q)]
        if v.empty:
            continue
        v = v.iloc[0]
        ax.plot(v, y, "v", ms=11, color=CASE_ORANGE, mec=SURFACE, zorder=4)
        label = name.title().replace(" Group", "").replace(" Management", "")
        near_cut = abs(v - cut[0.01]) < 4            # keep the label off the cutoff line
        text = f"{label}\n{v:.0f}%\n({q})" if near_cut else f"{label}\n{v:.0f}% ({q})"
        ax.text(max(v, cut[0.01]) + 1.2 if near_cut else v, y * (1 if near_cut else 2.2), text,
                ha="left" if near_cut else "center", va="center" if near_cut else "bottom",
                fontsize=11.5, color=INK)
    ax.text(20, counts[-1] * 1.4, "+20%\nor more", ha="right", va="bottom", fontsize=10.5, color=MUTED)
    ax.minorticks_off()
    ax.set_xlim(-100, 20)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0%")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlabel("Change in the value of a borrower's loans over one quarter", fontsize=13, color=MUTED)
    ax.set_ylabel("Borrower-quarters (log scale)", fontsize=13, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=12, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    within = (r.abs() < 5).mean()
    fig.text(0.02, 0.975, f"Most borrowers barely move; the worst 1% fall more than {-cut[0.01]:.0f}% in a quarter",
             ha="left", va="top", fontsize=17, color=INK, weight="bold")
    fig.text(0.02, 0.918, f"{len(r):,} borrower-quarters with at least $1M of BDC loans, 2023Q2–2026Q2 pooled. "
             f"{within:.0%} move less than 5%", ha="left", va="top", fontsize=12.5, color=MUTED)
    # footnote: the market-wide move is tiny next to the cutoff (full chart: market_move.png)
    mkt = market_move(b)
    worst, others = mkt.idxmin(), mkt.drop(mkt.idxmin())
    fig.text(0.02, 0.02, f"Market-wide change in value (size-weighted): {mkt[worst]:.1f}% in {worst}, the worst quarter; "
             f"{others.min():+.1f}% to {others.max():+.1f}% in all others.",
             ha="left", va="bottom", fontsize=11, color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.85, bottom=0.15)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def chart_market_move(b: pd.DataFrame, path: Path) -> None:
    """Market-wide change in value each quarter (all borrowers, weighted by prior fair value), drawn on the
    same scale as the fixed stress cutoff: answers 'does the cutoff ignore market moves?'"""
    b = eligible(b)
    cut = b["val_ret"].quantile(0.01) * 100
    mkt = market_move(b)
    worst = mkt.idxmin()
    x = range(len(mkt))
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(x, mkt.values, width=0.6, color=BLUE)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(cut, color=INK, lw=1.6)
    ax.text(len(mkt) - 0.6, cut - 0.6, f"Stress cutoff {cut:.0f}%", ha="right", va="top",
            fontsize=13, color=INK, weight="bold")
    i = list(mkt.index).index(worst)
    ax.text(i, mkt[worst] - 0.5, f"{mkt[worst]:.1f}%", ha="center", va="top", fontsize=12.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(mkt.index, rotation=45, ha="right")
    ax.set_ylim(-25, 5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0%")
    ax.set_ylabel("Change in value over the quarter", fontsize=13, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=12, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    others = mkt.drop(worst)
    fig.text(0.02, 0.975, f"The market moves at most {abs(mkt.min()):.1f}% a quarter; the stress cutoff is {abs(cut):.0f}%",
             ha="left", va="top", fontsize=17, color=INK, weight="bold")
    fig.text(0.02, 0.918, f"Change in value of all borrowers with at least $1M of BDC loans, weighted by size. "
             f"Other quarters: {others.min():+.1f}% to {others.max():+.1f}%",
             ha="left", va="top", fontsize=12.5, color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.85, bottom=0.14)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    out = RISK_MONITOR_DIR / "distress"
    out.mkdir(parents=True, exist_ok=True)
    print("Building the loan panel and the borrower-quarter panel ...")
    panel, rows = build_monitor_panel(FINAL_DIR / "FINAL_DATASET.csv")
    b = D.borrower_quarter_panel(panel, rows)
    t = tail_shares(b)
    quarters = sorted(t.loc[t["cal_q"] != "pooled", "cal_q"].unique())
    na_ts = pd.concat([nonaccrual_shares(b, q) for q in quarters], ignore_index=True)
    na = na_ts[na_ts["cal_q"] == quarters[-1]]
    t.to_csv(out / "tail_concentration.csv", index=False)
    na.to_csv(out / "tail_concentration_nonaccrual.csv", index=False)
    na_ts.to_csv(out / "tail_concentration_nonaccrual_timeseries.csv", index=False)
    chart_timeseries(t, na_ts, out / "tail_concentration_timeseries.png", case_marks(b))
    chart_markdown_total(t, out / "tail_concentration_markdown_total.png")
    chart_distribution(b, out / "markdown_distribution.png")
    chart_market_move(b, out / "market_move.png")
    print(na_ts[na_ts["tail"] < 0.1].round(3).to_string(index=False))
    print(t[t["cal_q"] == "pooled"].round(3).to_string(index=False))
    print(na.round(3).to_string(index=False))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
