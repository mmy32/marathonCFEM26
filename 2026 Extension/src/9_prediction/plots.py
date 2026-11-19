"""Charts for the early-warning scorecard (src/9_prediction), in the deck's style
(src/8_risk_monitor/charts.py): event rate by bucket of each sign, the mark path before a
credit event, the scorecard's odds ratios, the out-of-sample event rate by score decile and
the rolling AUC by quarter. The event throughout is the model's one outcome (config.OUTCOME)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from charts import BLUE, GRID, INK, MUTED, ORANGE, SURFACE, _spread, _style
from specs import SCORECARD

EVENT_NOTE = ("Event: within the next two quarters a BDC flags the borrower's debt non-accrual, or a lender switches "
              "its interest to PIK, swaps its debt for equity or writes down its cost")
COUNT_WORDS = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}

SCORECARD_UNITS = {"price": "Mark 10 points lower", "d_price": "Mark fell 5 points\nthis quarter",
                   "pik_switch": "Switched to PIK\nthis quarter",
                   "max_drawdown": "Mark fell 5 more points\nover the past year",
                   "dispersion": "Lenders' marks 1 point\nfurther apart"}
NAMES = {"price": "Mark level (FV / PAR)", "d_price": "Change in the mark this quarter",
         "max_drawdown": "Fall in the mark over the past year", "pik_switch": "PIK switch",
         "dispersion": "Valuation gap", "crowding": "Crowding"}
NAMES = {k: v if k in SCORECARD else v + " (tested, not kept)" for k, v in NAMES.items()}
SHORT = {"single lender": "1 lender", "under 0.5": "< 0.5", "over 5": "> 5", "under 1": "< 1",
         "over 10": "> 10"}     # bucket labels that fit under a bar
XLABELS = {"price": "Mark, cents per dollar of par", "d_price": "Same loans, points of par",
           "max_drawdown": "Largest fall over four quarters, same loans, points", "pik_switch": "",
           "dispersion": "Highest minus lowest lender family's mark, points", "crowding": "Manager families lending"}


def _title(fig, title: str, subtitle: str | None = None):
    fig.text(0.02, 0.97, title, ha="left", va="top", fontsize=13, color=INK, weight="bold")
    if subtitle:
        fig.text(0.02, 0.915, subtitle, ha="left", va="top", fontsize=9, color=MUTED)


def _note(fig, text: str):
    fig.text(0.02, 0.02, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def _rate_bars(ax, x, rate, lo, hi, color=BLUE, width=0.62):
    """Bars of a rate in % with its 95% interval as a whisker."""
    ax.bar(x, rate * 100, width=width, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.vlines(x, lo * 100, hi * 100, color=INK, linewidth=1, zorder=4)


def event_rate_by_predictor(t: pd.DataFrame, path, window: str):
    """Event rate within two quarters by bucket of each sign, with borrower-resampled 95% intervals."""
    base = t["base_rate"].iloc[0] * 100
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), facecolor=SURFACE)
    for ax, col in zip(axes.ravel(), NAMES):
        d = t[t["predictor"] == col]
        _style(ax, NAMES[col], "Event rate within two quarters (%)")
        x = np.arange(len(d))
        _rate_bars(ax, x, d["rate"], d["rate_ci_low"], d["rate_ci_high"])
        ax.axhline(base, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        top = max(d["rate_ci_high"].max() * 100, base)
        for xi, (r, hi) in enumerate(zip(d["rate"], d["rate_ci_high"])):
            ax.text(xi, hi * 100 + top * 0.03, f"{r * 100:.1f}%", ha="center", va="bottom", fontsize=8.5, color=INK,
                    zorder=5)
        ax.set_xticks(x, [f"{SHORT.get(b, b)}\n{int(e)} / {int(n):,}" for b, e, n in zip(d["bucket"], d["events"], d["n"])],
                      fontsize=6.8)
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel(XLABELS[col], color=MUTED, fontsize=8.5)
        ax.set_ylim(0, top * 1.2)
    _title(fig, "Borrowers with a credit event within two quarters, by warning sign",
           f"Accruing borrowers, all BDCs' loans pooled; signs from {window}. "
           f"Dashed line: average event rate, {base:.2f}%. Whiskers: 95% intervals")
    _note(fig, EVENT_NOTE + " (exits unlabelled).\nUnder each bar: events / borrower-quarters. "
               "The valuation gap needs a loan held by 2+ manager families ('single lender' otherwise). "
               "Points = cents per dollar of par.")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.13, hspace=0.62, wspace=0.2)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def odds_ratios(coefs: pd.DataFrame, path, window: str):
    """The scorecard in one picture: odds ratios (95% CI, borrower-clustered) of its signs. The
    mark and its change are shown per fall, so every sign reads as 'times the odds'. If the
    valuation gap is in the scorecard, its indicator (held by 2+ lender families) is in
    coefficients.csv, not drawn."""
    c = coefs[coefs["scorecard"] & coefs["term"].isin(SCORECARD_UNITS)].set_index("term")
    fall = ["price", "d_price"]             # flip: per 10 points lower, per 5-point fall
    c.loc[fall, ["odds_ratio", "or_ci_low", "or_ci_high"]] = 1 / c.loc[fall, ["odds_ratio", "or_ci_high", "or_ci_low"]].to_numpy()
    terms = [t for t in SCORECARD_UNITS if t in c.index][::-1]
    d = c.reindex(terms)
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    _style(ax, "", "")
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    y = np.arange(len(terms))
    ax.hlines(y, d["or_ci_low"], d["or_ci_high"], color=BLUE, linewidth=2, alpha=0.5)
    ax.scatter(d["odds_ratio"], y, s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for yi, v in zip(y, d["odds_ratio"]):         # two decimals near 1, so a small effect does not read as none
        ax.text(v, yi + 0.12, f"{v:.2f}x" if v < 1.1 else f"{v:.1f}x", ha="center", va="bottom", fontsize=8.5,
                color=INK)
    ax.axvline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xscale("log")
    ticks = [v for v in [1, 1.5, 2, 3, 4, 5, 7, 10] if v <= max(5, d["or_ci_high"].max())]
    ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
    ax.set_xticks([], minor=True)
    ax.set_xlim(0.9, max(6, d["or_ci_high"].max() * 1.1))
    ax.set_ylim(-0.5, len(terms) - 0.4)
    ax.set_yticks(y, [SCORECARD_UNITS[t] for t in terms])
    ax.set_xlabel("Times the odds of a credit event within two quarters (log scale; 1 = no effect)", color=MUTED,
                  fontsize=9)
    n = COUNT_WORDS.get(len(terms), str(len(terms)))
    _title(fig, f"{n} warning signs, each multiplying the odds of a credit event",
           f"Logistic regression on the {n.lower()} together; accruing borrowers, signs from {window}; "
           "95% intervals clustered by borrower")
    _note(fig, EVENT_NOTE + ".")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.83, bottom=0.2)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def score_deciles(dec: pd.DataFrame, path, events: str, fitted: str):
    """Out of sample: realized event rate by score decile (bars, borrower-resampled 95% intervals) for the
    scorecard, with its predicted rate (dots), next to the equal-weighted ranking of the same signs."""
    models = list(dec["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), facecolor=SURFACE, sharey=True)
    for ax, model, color in zip(axes, models, [BLUE, ORANGE]):
        d = dec[dec["model"] == model].sort_values("decile")
        _style(ax, model[0].upper() + model[1:], "Borrowers with a credit event (%)" if ax is axes[0] else "")
        x = d["decile"].to_numpy()
        _rate_bars(ax, x, d["event_rate"], d["rate_ci_low"], d["rate_ci_high"], color=color, width=0.7)
        if d["predicted_rate"].notna().any():
            ax.scatter(x, d["predicted_rate"] * 100, s=45, color=SURFACE, edgecolor=INK, linewidth=1.5, zorder=5,
                       label="predicted by the scorecard")
            ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
        tops = np.fmax(d["rate_ci_high"], d["predicted_rate"].fillna(0))
        for xi, r, top in zip(x, d["event_rate"], tops):
            ax.text(xi, top * 100 + tops.max() * 1.5, f"{r * 100:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x, [f"{v}\n{int(e)}" for v, e in zip(x, d["events"])])
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Score decile within the quarter (10 = riskiest tenth)", color=MUTED, fontsize=9)
    axes[0].set_ylim(bottom=0)
    _title(fig, f"Out of sample: credit events in {events}, by score decile",
           f"Fitted on {fitted} only. Bars: realized rate; whiskers: 95% intervals")
    _note(fig, EVENT_NOTE + ".\nUnder each decile: its events. The equal-weighted ranking averages the same signs' "
               "ranks within the quarter; nothing is fitted.")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.17, wspace=0.06)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def event_profile(p: pd.DataFrame, path, window: str):
    """Median mark in the four quarters before a credit event (borrowers flagged non-accrual, and
    borrowers with a concession only), against borrowers with no event."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor=SURFACE)
    _style(ax, "", "Median mark (FV / PAR, cents per dollar)")
    groups = [("no credit event", MUTED), ("concession, no non-accrual", ORANGE), ("entered non-accrual", BLUE)]
    last = [p[p["group"] == g].sort_values("k")["median_price"].iloc[-1] * 100 for g, _ in groups]
    for (group, color), y in zip(groups, _spread(last, gap=1.0)):
        d = p[p["group"] == group].sort_values("k")
        ax.plot(d["k"], d["median_price"] * 100, color=color, linewidth=2, marker="o", markersize=7, zorder=3)
        if group == "entered non-accrual":
            ax.fill_between(d["k"], d["p25_price"] * 100, d["p75_price"] * 100, color=color, alpha=0.12, linewidth=0)
        ax.text(0.15, y, f"{group} ({int(d['borrowers'].iloc[0]):,})", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(-4, 1), ["-4", "-3", "-2", "-1", "event\nquarter"])
    ax.set_xlim(-4.3, 2.6)
    ax.set_xlabel("Quarters before the credit event", color=MUTED, fontsize=9)
    _title(fig, "Marks fall before non-accrual; before a concession they barely move",
           "Borrowers entering non-accrual (shaded: middle half), borrowers with a concession only, and borrowers with no event")
    _note(fig, f"Events in {window} from the labelled risk set (borrower-quarters in brackets). Concession: a switch to PIK, "
               "a debt-for-equity swap or a cost write-down.\nMark = all BDCs' fair value / par on priceable loans.")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.2)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def rolling_auc_by_quarter(t: pd.DataFrame, path, dev_end: str, test_from: str):
    """ROC-AUC of each rolling test quarter (analysis.by_quarter): the scorecard against the
    mark alone. Signal quarters up to `dev_end` chose the signs, those from `test_from` on are the
    out-of-sample test, and any in between are skipped (their windows overlap the test)."""
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    _style(ax, "", "ROC-AUC")
    q = t["cal_q"].to_numpy()
    zones = [(q <= dev_end, "development:\nchose the signs", None),
             ((q > dev_end) & (q < test_from), "skipped: window\noverlaps the test", GRID),
             (q >= test_from, "test quarters", BLUE)]
    for mask, label, shade in zones:
        idx = np.flatnonzero(mask)
        if len(idx):
            lo, hi = idx.min() - 0.5, idx.max() + 0.5
            if shade:
                ax.axvspan(lo, hi, color=shade, alpha=0.6 if shade == GRID else 0.07, linewidth=0, zorder=0)
            ax.text((lo + hi) / 2, 0.52, label, ha="center", va="bottom", fontsize=8, color=MUTED)
    ends = []
    for col, label, color in [("mark alone", "mark alone", MUTED), ("scorecard", "scorecard", BLUE)]:
        ax.plot(x, t[col], color=color, linewidth=2, marker="o", markersize=7, zorder=3)
        ends.append((t[col].iloc[-1], label))
    for (_, label), y in zip(ends, _spread([v for v, _ in ends], gap=0.03)):
        ax.text(len(t) - 1 + 0.2, y, label, va="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"{w}\n{int(n)} events" for w, n in zip(t["outcome_q"], t["events"])], fontsize=8)
    ax.set_xlim(-0.5, len(t) + 0.6)
    ax.set_ylim(0.5, 1.0)
    _title(fig, "Out of sample, quarter by quarter: credit events within two quarters",
           "Each quarter scored by a model refitted on every window closed by then; x-axis: the outcome window")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.17)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
