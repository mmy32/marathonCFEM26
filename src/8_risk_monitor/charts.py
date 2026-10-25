"""One chart per layer of the monitor (market, signal, evaluation). Static PNGs, light theme,
reference palette from the dataviz guidance: one axis per panel, thin marks, recessive grid."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def market_layer(fv: pd.DataFrame, ret: pd.DataFrame, path):
    """Total BDC fair value and the quarterly market return, with the report's reference points."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE)
    ax = axes[0]
    ax.plot(fv["cal_q"], fv["fv_bn"], color=BLUE, linewidth=2, marker="o", markersize=4, label="Panel: total BDC fair value")
    ax.axhline(500, color=ORANGE, linewidth=1.5, linestyle="--", label="Report: BDCs ~$500bn (p.39)")
    _style(ax, "BDC market size", "$bn")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    ax = axes[1]
    r = ret[ret["cal_q"] > ret["cal_q"].min()]
    ax.bar(r["cal_q"], 100 * r["IndexReturn"], color=BLUE, width=0.6, label="Panel: FV-weighted total return")
    ax.scatter(["2026Q1"], [0.9], color=ORANGE, s=60, zorder=3, label="Report: MSCI Q1 2026, 0.9% (Fig 15)")
    _style(ax, "Quarterly return (2025: panel 9.6%, MSCI 9.5%)", "% per quarter")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def flag_tail_loans(panel: pd.DataFrame) -> pd.Series:
    """A loan is in the signal tail if any of: PIK toggle at t or t-1; lowest mark among
    families in the bottom decile of the quarter (V2); this lender's distance from consensus
    in the top decile; lender families fell by the most (bottom decile of C4)."""
    pct = lambda c: panel.groupby("cal_q")[c].rank(pct=True)
    return ((panel["P2_recent"] == 1) | (pct("V2_low_mark") <= 0.10) | (pct("mark_distance") >= 0.90)
            | (pct("C4_d_families") <= 0.10))


def signal_layer(panel: pd.DataFrame, path, threshold: float = 0.10) -> pd.DataFrame:
    """Distribution across BDCs of the share of fair value in signal-tail loans, by quarter."""
    p = panel.assign(tail=flag_tail_loans(panel), fv_tail=lambda d: d["FV"].where(d["tail"], 0.0))
    by_bdc = p.groupby(["cal_q", "cik"]).agg(fv=("FV", "sum"), fv_tail=("fv_tail", "sum")).reset_index()
    by_bdc["tail_share"] = by_bdc["fv_tail"] / by_bdc["fv"]
    q = by_bdc.groupby("cal_q")["tail_share"].agg(median="median", p90=lambda s: s.quantile(0.9),
                                                  n_above=lambda s: int((s > threshold).sum()), n="size").reset_index()
    q = q[q["cal_q"] > q["cal_q"].min()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE)
    ax = axes[0]
    ax.plot(q["cal_q"], 100 * q["median"], color=BLUE, linewidth=2, marker="o", markersize=4, label="Median BDC")
    ax.plot(q["cal_q"], 100 * q["p90"], color=ORANGE, linewidth=2, marker="o", markersize=4, label="90th percentile BDC")
    _style(ax, "Share of a BDC's fair value in signal-tail loans", "% of BDC fair value")
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax = axes[1]
    ax.bar(q["cal_q"], q["n_above"], color=BLUE, width=0.6)
    for x, (n_above, n) in enumerate(zip(q["n_above"], q["n"])):
        ax.text(x, n_above, f"{n_above}/{n}", ha="center", va="bottom", fontsize=7, color=MUTED)
    _style(ax, f"BDCs with more than {threshold:.0%} of fair value in the tail", "BDCs")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return by_bdc


AQUA = "#1baf7a"


def distress_profiles(prof: pd.DataFrame, path, columns=None):
    """Event-time profiles: persistent events (with a +-1.96 SE band), recovered events, and
    matched never-distressed controls. One small multiple per signal, event quarter dashed."""
    pers = prof[prof["group"] == "persistent"]
    rec = prof[prof["group"] == "recovered"]
    cols = columns or list(dict.fromkeys(pers["column"]))
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow), facecolor=SURFACE)
    n_p, n_r = int(pers["n_events"].iloc[0]), int(rec["n_events"].iloc[0])
    for ax, col in zip(axes.flat, cols):
        p, r = pers[pers["column"] == col], rec[rec["column"] == col]
        ax.fill_between(p["k"], p["event_mean"] - 1.96 * p["event_se"], p["event_mean"] + 1.96 * p["event_se"],
                        color=BLUE, alpha=0.15, linewidth=0)
        ax.plot(p["k"], p["event_mean"], color=BLUE, linewidth=2, marker="o", markersize=4, label=f"Persistent ({n_p})")
        ax.plot(r["k"], r["event_mean"], color=AQUA, linewidth=2, marker="o", markersize=4, label=f"Recovered ({n_r})")
        ax.plot(p["k"], p["control_mean"], color=ORANGE, linewidth=2, linestyle="--", label="Matched controls")
        ax.axvline(0, color=MUTED, linewidth=1, linestyle=":")
        _style(ax, p["signal"].iloc[0], "")
        ax.set_xticks(list(p["k"]))
    for ax in list(axes.flat)[len(cols):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Borrowers around a bottom-1% quarterly markdown (quarter 0): persistent vs recovered vs matched controls",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def evaluation_layer(buckets: pd.DataFrame, summary: pd.DataFrame, horizon: int, path,
                     outcome_label: str = "Peer-relative markdown"):
    """Event rate by signal bucket, one small multiple per signal, overall rate dashed."""
    b = buckets[buckets["horizon"] == horizon]
    sigs = list(dict.fromkeys(b["signal"]))
    ncol = 4
    nrow = int(np.ceil(len(sigs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), facecolor=SURFACE, sharey=True)
    for ax, sig in zip(axes.flat, sigs):
        d = b[b["signal"] == sig]
        s = summary[(summary["signal"] == sig) & (summary["horizon"] == horizon)].iloc[0]
        ax.bar(d["bucket"], 100 * d["rate"], color=BLUE, width=0.6)
        ax.axhline(100 * s["event_rate"], color=MUTED, linewidth=1, linestyle="--")
        gain = "" if np.isnan(s["gain_lo90"]) else f", gain {s['auc_gain']:+.3f} [{s['gain_lo90']:+.3f}, {s['gain_hi90']:+.3f}]"
        _style(ax, sig, "event rate %")
        ax.set_title(f"{sig}\nAUC {s['auc_signal']:.2f}{gain}", loc="left", fontsize=8, color=INK)
    for ax in list(axes.flat)[len(sigs):]:
        ax.set_visible(False)
    fig.suptitle(f"{outcome_label} within {horizon} quarter(s), by signal bucket "
                 f"(Q1 = lowest risk, Q5 = highest; flags 0/1; dashed = overall rate)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


CASE_COLORS = [BLUE, "#1baf7a", "#eda100", "#6250d6", "#e87ba4", "#008300"]
CASE_NAMES = {"WEBPT": "WebPT", "BVI MEDICAL": "BVI Medical", "LOGRHYTHM": "LogRhythm"}


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
    families that no longer hold it, families averaging under $0.5M, and beyond the `max_families` largest (plus the lowest)."""
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
    fams = others + [low]
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    _style(ax, "", "Value per 100 of loan (fair value / par)")
    x = {q: i for i, q in enumerate(qs)}
    ends = []
    for i, fam in enumerate(fams):
        s = g[g["family"] == fam].sort_values("cal_q")
        c = ORANGE if fam == low else CASE_COLORS[i % len(CASE_COLORS)]
        xs = [x[q] for q in s["cal_q"]]
        ax.plot(xs, s["mark"], color=c, linewidth=2.6 if fam == low else 1.8, marker="o", markersize=5, zorder=3)
        na = s[s["na"] == 1]
        ax.scatter([x[q] for q in na["cal_q"]], na["mark"], s=110, facecolor=SURFACE, edgecolor=c, linewidth=2, zorder=4)
        name = fam if len(fam) <= 18 else fam[:17].rstrip() + "…"
        ends.append((xs[-1], s["mark"].iloc[-1], f"{name} {s['mark'].iloc[-1]:.0f}  (${s['par'].iloc[-1] / 1e6:,.0f}M)", fam == low))
    lo, hi = ax.get_ylim()
    ys = _spread([e[1] for e in ends], gap=(hi - lo) * 0.055)
    for (xe, ye, text, is_low), yl in zip(ends, ys):
        ax.annotate(text, (xe, ye), xytext=(len(qs) - 1 + 0.25, yl), textcoords="data", va="center", fontsize=8.5,
                    color=INK, weight="bold" if is_low else "normal",
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8) if abs(yl - ye) > 0.5 or xe < len(qs) - 1 else None)
    ax.set_ylim(lo, max(hi, max(ys) + (hi - lo) * 0.03))
    ax.set_xticks(range(len(qs)), qs)
    ax.set_xlim(-0.3, len(qs) + 2.1)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    title = CASE_NAMES.get(borrower.upper(), borrower.title())
    subtitle = "same loan, different values" if len(fams) > 1 else "value of its loan"
    fig.text(0.02, 0.965, f"{title}: {subtitle}", ha="left", va="top", fontsize=13, color=INK, weight="bold")
    note = f"Hollow circle = lender has stopped counting the interest as income (non-accrual). Label: latest value (par held, {qs[-1]})."
    fig.text(0.02, 0.02, note, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.12)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
