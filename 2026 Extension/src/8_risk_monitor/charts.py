"""Shared chart style (light theme, recessive grid) and the case-study chart: each lender family's mark on
one borrower by quarter (slides 9, 10, 13); Medallia is drawn with a par-weighted average."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


AQUA = "#1baf7a"


CASE_COLORS = [BLUE, "#1baf7a", "#eda100", "#6250d6", "#e87ba4", "#008300"]
CASE_NAMES = {"WEBPT": "WebPT", "BVI MEDICAL": "BVI Medical", "LOGRHYTHM": "LogRhythm"}
# Borrowers drawn with a par-weighted average across the lender families shown (midterm slide 10):
# the families become faded lines and the average is the bold orange line.
CASE_AVERAGE = {"MEDALLIA"}


def _spread(ys, gap):
    """Nudge label positions apart so neighbours are at least `gap` apart (keeps their order)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def case_marks(lenders: pd.DataFrame, borrower: str, path, start: str = "2024Q4", max_families: int = 6):
    """One borrower's mark (fair value / par, par-weighted) at each lender family by quarter.
    The family with the lowest latest mark is drawn in orange; quarters on non-accrual get a
    hollow marker. Left out: positions under $1M held at exactly par (usually undrawn revolvers),
    families that no longer hold it, families averaging under $0.5M, and beyond the `max_families` largest (plus the lowest).
    For a borrower in CASE_AVERAGE the families are faded and the par-weighted average across them
    is drawn in orange instead (families that left before the latest quarter, e.g. Apollo for
    Medallia, are in neither)."""
    d = lenders[(lenders["borrower"] == borrower.upper()) & (lenders["cal_q"] >= start)].copy()
    d = d[~((d["mark"] >= 0.999) & (d["par"] < 1e6))]
    d["w"] = d["par"] * d["mark"]
    g = d.groupby(["family", "cal_q"]).agg(par=("par", "sum"), w=("w", "sum"), na=("non_accrual", "max")).reset_index()
    g["mark"] = 100 * g["w"] / g["par"]
    size = g.groupby("family")["par"].mean()
    g = g[g["family"].isin(size[size >= 5e5].index)]
    qs = sorted(g["cal_q"].unique())
    g = g[g["family"].isin(g.loc[g["cal_q"] == qs[-1], "family"])]     # still holds it in the latest quarter
    qs = sorted(g["cal_q"].unique())
    last = g[g["cal_q"] == qs[-1]].sort_values("mark")
    low = last["family"].iloc[0]
    average = borrower.upper() in CASE_AVERAGE
    others = [f for f in size.loc[g["family"].unique()].sort_values(ascending=False).index if f != low][:max_families - 1]
    fams = others + [low]
    if average:   # no family is singled out (same order, own colour); the average takes the orange
        low = None
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    _style(ax, "", "Value per 100 of loan (fair value / par)")
    x = {q: i for i, q in enumerate(qs)}
    ends = []
    alpha = 0.45 if average else 1.0
    for i, fam in enumerate(fams):
        s = g[g["family"] == fam].sort_values("cal_q")
        c = ORANGE if fam == low else CASE_COLORS[i % len(CASE_COLORS)]
        xs = [x[q] for q in s["cal_q"]]
        ax.plot(xs, s["mark"], color=c, linewidth=2.6 if fam == low else 1.8, marker="o", markersize=5, zorder=3,
                alpha=alpha)
        na = s[s["na"] == 1]
        ax.scatter([x[q] for q in na["cal_q"]], na["mark"], s=110, facecolor=SURFACE, edgecolor=c, linewidth=2, zorder=4,
                   alpha=alpha)
        name = fam if len(fam) <= 18 else fam[:17].rstrip() + "…"
        ends.append((xs[-1], s["mark"].iloc[-1], f"{name} {s['mark'].iloc[-1]:.0f}  (${s['par'].iloc[-1] / 1e6:,.0f}M)",
                     fam == low, c))
    if average:
        a = g[g["family"].isin(fams)].groupby("cal_q").agg(par=("par", "sum"), w=("w", "sum")).reset_index()
        a = a.sort_values("cal_q")
        a["mark"] = 100 * a["w"] / a["par"]
        xs = [x[q] for q in a["cal_q"]]
        ax.plot(xs, a["mark"], color=ORANGE, linewidth=3.4, marker="o", markersize=8, zorder=5)
        ends.append((xs[-1], a["mark"].iloc[-1],
                     f"Weighted average {a['mark'].iloc[-1]:.0f}  (${a['par'].iloc[-1] / 1e6:,.0f}M)", True, ORANGE))
    lo, hi = ax.get_ylim()
    ys = _spread([e[1] for e in ends], gap=(hi - lo) * 0.06)
    # Each label gets a key in its line's colour and a leader line in that colour from the line's end,
    # so lines that finish close together can still be told apart. The label text stays in ink.
    x_key = len(qs) - 1 + 0.35
    for (xe, ye, text, is_low, c), yl in zip(ends, ys):
        ax.plot([xe + 0.1, x_key - 0.06], [ye, yl], color=c, linewidth=1.1, alpha=0.75, zorder=2)
        ax.plot([x_key, x_key + 0.32], [yl, yl], color=c, linewidth=4, solid_capstyle="round", zorder=3)
        ax.text(x_key + 0.45, yl, text, va="center", fontsize=8.5, color=INK, weight="bold" if is_low else "normal")
    ax.set_ylim(lo, max(hi, max(ys) + (hi - lo) * 0.03))
    ax.set_xticks(range(len(qs)), qs)
    ax.set_xlim(-0.3, len(qs) + 2.6)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    title = CASE_NAMES.get(borrower.upper(), borrower.title())
    if average:
        subtitle = "par-weighted value of its loan"
    else:
        subtitle = "same loan, different values" if len(fams) > 1 else "value of its loan"
    fig.text(0.02, 0.965, f"{title}: {subtitle}", ha="left", va="top", fontsize=13, color=INK, weight="bold")
    note = (f"Hollow circle = lender has stopped counting the interest as income (non-accrual). "
            f"Label: latest value (par held, {qs[-1]}).")
    if average:
        note = (f"Faded lines = individual lender families. Hollow circle = lender has stopped counting the interest "
                f"as income (non-accrual).\nLabel: latest value (par held, {qs[-1]}).")
    fig.text(0.02, 0.02, note, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.14 if average else 0.12)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
