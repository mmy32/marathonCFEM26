"""Shared chart style (light theme, recessive grid) and the case-study chart: each lender family's mark on
one borrower by quarter (slides 9, 10, 13); Medallia is drawn with a par-weighted average, and also at
its size on the poster (case_marks_poster)."""
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


CASE_COLORS = [BLUE, "#1baf7a", "#eda100", "#6250d6", "#e87ba4", "#008300"]
CASE_NAMES = {"WEBPT": "WebPT", "BVI MEDICAL": "BVI Medical", "LOGRHYTHM": "LogRhythm"}
# Borrowers drawn with a par-weighted average across the lender families shown (midterm slide 10):
# the families become faded lines and the average is the bold orange line.
CASE_AVERAGE = {"MEDALLIA"}
# Case studies on the poster: also drawn at the poster's size by case_marks_poster, whose legend
# uses these shorter family names.
POSTER_CASES = {"MEDALLIA"}
SHORT_FAMILY = {"Onex Falcon Direct Lending BDC Fund": "Onex Falcon"}
# The poster's typeface for the charts drawn at its size, so their labels match the poster's text;
# matplotlib's own DejaVu Sans stands in on a computer without Arial (the PNG then differs).
POSTER_FONT = ["Arial", "DejaVu Sans"]


def _spread(ys, gap):
    """Nudge label positions apart so neighbours are at least `gap` apart (keeps their order)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def _case_families(lenders: pd.DataFrame, borrower: str, start: str, max_families: int):
    """The lender families a case chart draws for one borrower: (each family's par-weighted mark by
    quarter, the families in drawing order, the quarters, the family with the lowest latest mark).
    Left out: positions under $1M held at exactly par (usually undrawn revolvers), families that no
    longer hold it, families averaging under $0.5M, and beyond the `max_families` largest (plus the
    lowest)."""
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
    others = [f for f in size.loc[g["family"].unique()].sort_values(ascending=False).index if f != low][:max_families - 1]
    return g, others + [low], qs, low


def _case_average(g: pd.DataFrame, fams: list) -> pd.DataFrame:
    """The par-weighted average mark across the families drawn, by quarter."""
    a = g[g["family"].isin(fams)].groupby("cal_q").agg(par=("par", "sum"), w=("w", "sum")).reset_index()
    a = a.sort_values("cal_q")
    a["mark"] = 100 * a["w"] / a["par"]
    return a


def case_marks(lenders: pd.DataFrame, borrower: str, path, start: str = "2024Q4", max_families: int = 6):
    """One borrower's mark (fair value / par, par-weighted) at each lender family by quarter
    (the families of _case_families). The family with the lowest latest mark is drawn in orange;
    quarters on non-accrual get a hollow marker.
    For a borrower in CASE_AVERAGE the families are faded and the par-weighted average across them
    is drawn in orange instead (families that left before the latest quarter, e.g. Apollo for
    Medallia, are in neither)."""
    g, fams, qs, low = _case_families(lenders, borrower, start, max_families)
    average = borrower.upper() in CASE_AVERAGE
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
        a = _case_average(g, fams)
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


def _quarter_axis(ax, qs: list, font: float):
    """The quarters as ticks ('Q4', 'Q1', ...) and each year once, centred under its quarters."""
    ax.set_xticks(range(len(qs)), [q[4:] for q in qs])
    for year in sorted({q[:4] for q in qs}):
        xs = [i for i, q in enumerate(qs) if q.startswith(year)]
        ax.annotate(year, (sum(xs) / len(xs), 0), xycoords=("data", "axes fraction"), xytext=(0, -font - 4),
                    textcoords="offset points", ha="center", va="top", fontsize=font, color=MUTED)


@plt.rc_context({"font.family": POSTER_FONT})
def case_marks_poster(lenders: pd.DataFrame, borrower: str, path, start: str = "2024Q4", max_families: int = 6,
                      size: tuple = (4.0, 1.62), font: float = 7.5):
    """case_marks at the size it is placed on the poster (`size`, inches) with every label at
    `font` pt, the poster's chart-label size, in the poster's typeface (POSTER_FONT), so it is
    inserted at 100% with no text boxes over it. No title or note (the poster has its own). The
    legend column on the right gives each line's final mark and par held, highest mark first, with
    the key for the non-accrual marker at the top and room between the two for a note."""
    g, fams, qs, low = _case_families(lenders, borrower, start, max_families)
    average = borrower.upper() in CASE_AVERAGE
    if average:
        low = None
    W, H = size
    left, plot_w, top, plot_h = 0.41, 1.62, 0.07, 1.26          # the plot area, inches
    fig = plt.figure(figsize=size, facecolor=SURFACE)
    ax = fig.add_axes([left / W, (H - top - plot_h) / H, plot_w / W, plot_h / H])
    ax.set_facecolor(SURFACE)
    x = {q: i for i, q in enumerate(qs)}
    alpha = 0.45 if average else 1.0
    entries = []                                                 # (final mark, label, colour, bold)
    for i, fam in enumerate(fams):
        s = g[g["family"] == fam].sort_values("cal_q")
        c = ORANGE if fam == low else CASE_COLORS[i % len(CASE_COLORS)]
        ax.plot([x[q] for q in s["cal_q"]], s["mark"], color=c, linewidth=1.5 if fam == low else 1.0, marker="o",
                markersize=2.2, zorder=3, alpha=alpha)
        na = s[s["na"] == 1]
        ax.scatter([x[q] for q in na["cal_q"]], na["mark"], s=18, facecolor=SURFACE, edgecolor=c, linewidth=0.8,
                   zorder=4, alpha=alpha)
        end = s.iloc[-1]
        entries.append((end["mark"], f"{SHORT_FAMILY.get(fam, fam)} {end['mark']:.0f} · ${end['par'] / 1e6:,.0f}M", c,
                        fam == low))
    if average:
        a = _case_average(g, fams)
        ax.plot([x[q] for q in a["cal_q"]], a["mark"], color=ORANGE, linewidth=1.8, marker="o", markersize=3.2,
                zorder=5)
        entries.append((a["mark"].iloc[-1],
                        f"Weighted avg. {a['mark'].iloc[-1]:.0f} · ${a['par'].iloc[-1] / 1e6:,.0f}M", ORANGE, True))
    _quarter_axis(ax, qs, font)
    ax.set_xlim(-0.3, len(qs) - 0.7)
    ax.set_ylim(35, 102)
    ax.set_yticks(range(40, 101, 10))
    ax.tick_params(colors=MUTED, labelsize=font, length=0, pad=2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel("Mark, per 100 of par", color=INK, fontsize=font, labelpad=3)

    # the legend column, drawn in inches from its top-left corner
    x0 = left + plot_w + 0.10
    lg = fig.add_axes([x0 / W, 0, 1 - x0 / W, 1])
    lg.set_xlim(0, W - x0)
    lg.set_ylim(H, 0)
    lg.axis("off")
    step, text_x = 0.125, 0.15
    lg.scatter([0.06], [top + 0.06], s=18, facecolor=SURFACE, edgecolor=MUTED, linewidth=0.8, clip_on=False)
    lg.text(text_x, top + 0.06, "= non-accrual flagged", va="center", fontsize=font, color=INK)
    entries.sort(key=lambda e: -e[0])
    y_last = top + plot_h - 0.06
    for i, (_, text, c, bold) in enumerate(entries):
        y = y_last - (len(entries) - 1 - i) * step
        lg.plot([0, 0.11], [y, y], color=c, linewidth=2.2, solid_capstyle="round", clip_on=False)
        lg.text(text_x, y, text, va="center", fontsize=font, color=INK, weight="bold" if bold else "normal")
    lg.text(0, y_last - len(entries) * step, "Final mark · par held", va="center", fontsize=font, color=MUTED)
    fig.savefig(path, dpi=600, facecolor=SURFACE)
    plt.close(fig)
