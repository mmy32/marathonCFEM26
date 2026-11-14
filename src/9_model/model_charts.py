"""Charts for the early-warning scorecard (src/9_model), in the deck's style
(src/8_risk_monitor/charts.py): event rate by bucket of each sign, the mark path before a
credit event, the scorecard's odds ratios, the out-of-sample event rate by score decile and
the rolling AUC by quarter."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from charts import BLUE, GRID, INK, MUTED, ORANGE, SURFACE, _spread, _style

SCORECARD_UNITS = {"price": "Mark 10 points lower", "d_price": "Mark fell 5 points\nthis quarter",
                   "pik_switch": "Switched to PIK\nthis quarter"}
NAMES = {"price": "Mark level (FV / PAR)", "d_price": "Change in the mark this quarter", "pik_switch": "PIK switch",
         "dispersion": "Valuation gap", "crowding": "Crowding"}
SHORT = {"single lender": "1 lender", "under 0.5": "< 0.5", "over 5": "> 5"}     # bucket labels that fit under a bar
XLABELS = {"price": "Mark, cents per dollar of par", "d_price": "Same loans, points of par", "pik_switch": "",
           "dispersion": "Std of lender families' marks, points", "crowding": "Manager families lending"}


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
    """Event rate next quarter by bucket of each sign, with Wilson 95% intervals."""
    base = t["base_rate"].iloc[0] * 100
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), facecolor=SURFACE)
    axes.ravel()[-1].axis("off")
    for ax, col in zip(axes.ravel(), NAMES):
        d = t[t["predictor"] == col]
        _style(ax, NAMES[col], "Event rate next quarter (%)")
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
    _title(fig, "Borrowers with a credit event next quarter, by warning sign",
           f"Accruing borrowers, all BDCs' loans pooled; signs from {window}. "
           f"Dashed line: average event rate, {base:.2f}%. Whiskers: 95% intervals")
    _note(fig, "Credit event: non-accrual, or an exit marked below 80. Under each bar: events / borrower-quarters. The valuation gap needs a loan held by 2+ manager families "
               "('single lender' otherwise). Points = cents per dollar of par.")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.11, hspace=0.62, wspace=0.2)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def odds_ratios(coefs: pd.DataFrame, path, window: str):
    """The scorecard in one picture: odds ratios (95% CI, borrower-clustered) of the mark, its change
    this quarter and a PIK switch. The mark and its change are shown per fall, so every sign reads
    as 'times the odds'."""
    c = coefs[coefs["scorecard"] & coefs["term"].isin(SCORECARD_UNITS)].set_index("term")
    fall = ["price", "d_price"]             # flip: per 10 points lower, per 5-point fall
    c.loc[fall, ["odds_ratio", "or_ci_low", "or_ci_high"]] = 1 / c.loc[fall, ["odds_ratio", "or_ci_high", "or_ci_low"]].to_numpy()
    terms = list(SCORECARD_UNITS)[::-1]
    d = c.reindex(terms)
    fig, ax = plt.subplots(figsize=(10, 3.9), facecolor=SURFACE)
    _style(ax, "", "")
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    y = np.arange(len(terms))
    ax.hlines(y, d["or_ci_low"], d["or_ci_high"], color=BLUE, linewidth=2, alpha=0.5)
    ax.scatter(d["odds_ratio"], y, s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for yi, v in zip(y, d["odds_ratio"]):
        ax.text(v, yi + 0.12, f"{v:.1f}x", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.axvline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_xticks([1, 1.5, 2, 3, 4, 5], ["1", "1.5", "2", "3", "4", "5"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(0.9, 6)
    ax.set_ylim(-0.5, len(terms) - 0.4)
    ax.set_yticks(y, [SCORECARD_UNITS[t] for t in terms])
    ax.set_xlabel("Times the odds of a credit event next quarter (log scale; 1 = no effect)", color=MUTED,
                  fontsize=9)
    _title(fig, "Three warning signs, each multiplying the odds of a credit event",
           f"Logistic regression on the three together; accruing borrowers, signs from {window}; "
           "95% intervals clustered by borrower")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.8, bottom=0.15)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def score_deciles(dec: pd.DataFrame, path, events: str, fitted: str):
    """Out of sample: realized event rate by score decile (bars, Wilson 95% intervals) for the
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
        for xi, r, hi in zip(x, d["event_rate"], d["rate_ci_high"]):
            ax.text(xi, hi * 100 + 0.15, f"{r * 100:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x, [f"{v}\n{int(e)}" for v, e in zip(x, d["events"])])
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Score decile within the quarter (10 = riskiest tenth)", color=MUTED, fontsize=9)
    axes[0].set_ylim(bottom=0)
    _title(fig, f"Out of sample: credit events in {events}, by score decile",
           f"Fitted on {fitted} only. Bars: realized rate; whiskers: 95% intervals")
    _note(fig, "Credit event: non-accrual, or an exit marked below 80. Under each decile: its events. The equal-weighted ranking averages the three signs' ranks "
               "within the quarter; nothing is fitted.")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.17, wspace=0.06)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def event_profile(p: pd.DataFrame, path, window: str):
    """Median mark and share with a PIK switch in the four quarters before a credit event,
    against borrowers that stayed accruing."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    _style(a1, "Median mark (FV / PAR, cents per dollar)", "")
    _style(a2, "Share with a PIK switch that quarter (%)", "")
    for group, color in [("stayed accruing", MUTED), ("had a credit event", BLUE)]:
        d = p[p["group"] == group].sort_values("k")
        a1.plot(d["k"], d["median_price"] * 100, color=color, linewidth=2, marker="o", markersize=7, zorder=3)
        if group == "had a credit event":
            a1.fill_between(d["k"], d["p25_price"] * 100, d["p75_price"] * 100, color=color, alpha=0.12, linewidth=0)
        a2.plot(d["k"], d["share_pik_switch"] * 100, color=color, linewidth=2, marker="o", markersize=7, zorder=3)
        for ax, v in [(a1, d["median_price"].iloc[-1] * 100), (a2, d["share_pik_switch"].iloc[-1] * 100)]:
            ax.text(0.15, v, group, va="center", fontsize=8.5, color=INK)
    for ax in (a1, a2):
        ax.set_xticks(range(-4, 1), ["-4", "-3", "-2", "-1", "event\nquarter"])
        ax.set_xlim(-4.3, 1.6)
        ax.set_xlabel("Quarters before the credit event", color=MUTED, fontsize=9)
    _title(fig, "Marks fall before a credit event",
           "Borrowers with a credit event (shaded: middle half) vs borrowers that stayed accruing, same quarters")
    _note(fig, f"Events in {window} from the labelled risk set; impaired exits have no mark in the event quarter. Mark = all BDCs' fair value / par on priceable loans.")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.17, wspace=0.18)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def rolling_auc_by_quarter(t: pd.DataFrame, path, test_from: str):
    """ROC-AUC of each rolling test quarter (model_analysis.by_quarter): the scorecard against the
    mark alone. Quarters from `test_from` on are the out-of-sample test; the earlier ones chose
    the signs."""
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    _style(ax, "", "ROC-AUC")
    ends = []
    for col, label, color in [("mark alone", "mark alone", MUTED), ("scorecard", "scorecard", BLUE)]:
        ax.plot(x, t[col], color=color, linewidth=2, marker="o", markersize=7, zorder=3)
        ends.append((t[col].iloc[-1], label))
    for (_, label), y in zip(ends, _spread([v for v, _ in ends], gap=0.03)):
        ax.text(len(t) - 1 + 0.2, y, label, va="center", fontsize=8.5, color=INK)
    first_test = int(np.argmax(t["outcome_q"].to_numpy() >= test_from))
    ax.axvline(first_test - 0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(first_test - 0.4, 0.53, "out-of-sample test", fontsize=8, color=MUTED)
    ax.set_xticks(x, [f"{q}\n{int(n)} events" for q, n in zip(t["outcome_q"], t["events"])], fontsize=8)
    ax.set_xlim(-0.4, len(t) + 0.6)
    ax.set_ylim(0.5, 1.0)
    _title(fig, "Out of sample, quarter by quarter: credit events",
           "Each quarter's events scored by a model fitted only on the quarters before it")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.17)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
