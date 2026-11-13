"""Charts for the early-warning models (src/9_model), in the deck's style (src/8_risk_monitor/charts.py):
entry rate by predictor bucket, the mark path before a non-accrual entry, the headline model's
odds ratios, the rolling review lists by horizon and AUC by quarter, and the outline split's
ROC / PR curves and AUCs."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from charts import AQUA, BLUE, GRID, INK, MUTED, ORANGE, SURFACE, _spread, _style
from model_fit import review_capture

HORIZON_STYLE = {1: ("next quarter", BLUE), 2: ("within 2 quarters", ORANGE), 4: ("within 4 quarters", AQUA)}
HEADLINE_LABEL = "Mark, drawdown, PIK switch (logistic)"
HEADLINE_UNITS = {"price": "Mark 10 points lower", "max_drawdown": "Mark fell 5 points\nin the past year",
                  "pik_switch": "Switched to PIK\nthis quarter"}

NAMES = {"price": "Mark level (FV / PAR)", "max_drawdown": "Rolling max drawdown", "pik_switch": "PIK switch",
         "dispersion": "Valuation dispersion"}
MODEL_NAMES = {"logit": "Logistic", "ridge": "Ridge logistic", "lasso": "Lasso logistic",
               "gbt": "Gradient-boosted trees", "mlp": "Small neural network"}


def _title(fig, title: str, subtitle: str | None = None):
    fig.text(0.02, 0.97, title, ha="left", va="top", fontsize=13, color=INK, weight="bold")
    if subtitle:
        fig.text(0.02, 0.915, subtitle, ha="left", va="top", fontsize=9, color=MUTED)


def _note(fig, text: str):
    fig.text(0.02, 0.02, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def event_rate_by_predictor(t: pd.DataFrame, path, window: str):
    """2 x 2 bars: next-quarter non-accrual entry rate by bucket of each predictor."""
    base = t["base_rate"].iloc[0] * 100
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), facecolor=SURFACE)
    xlabels = {"price": "Mark, cents per dollar of par", "max_drawdown": "Largest fall in the mark over four quarters, points",
               "pik_switch": "", "dispersion": "Std of lender families' marks, points"}
    for ax, col in zip(axes.ravel(), ["price", "max_drawdown", "pik_switch", "dispersion"]):
        d = t[t["predictor"] == col]
        _style(ax, NAMES[col], "Entry rate next quarter (%)")
        x = np.arange(len(d))
        ax.bar(x, d["rate"] * 100, width=0.62, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.axhline(base, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        top = max(d["rate"].max() * 100, base)
        for xi, (r, n, e) in enumerate(zip(d["rate"], d["n"], d["events"])):
            ax.text(xi, r * 100 + top * 0.03, f"{r * 100:.1f}%", ha="center", va="bottom", fontsize=8.5, color=INK,
                    zorder=4, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0})
            ax.text(xi, -top * 0.075, f"{int(e)} / {int(n):,}", ha="center", va="top", fontsize=7, color=MUTED)
        ax.set_xticks(x, d["bucket"])
        ax.tick_params(axis="x", length=0, pad=16)
        ax.set_xlabel(xlabels[col], color=MUTED, fontsize=8.5)
        ax.set_ylim(0, top * 1.18)
    _title(fig, "Borrowers entering non-accrual next quarter, by warning sign",
           f"Accruing borrowers, all BDCs' loans pooled; predictor quarters {window}. "
           f"Dashed line: average entry rate, {base:.2f}%")
    _note(fig, "Under each bar: entries / borrower-quarters. Dispersion needs a loan held by 2+ manager families "
               "('single lender' otherwise). Points = cents per dollar of par.")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.11, hspace=0.62, wspace=0.18)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def holdout_curves(preds: pd.DataFrame, path, title_q: str):
    """ROC and precision-recall curves on the holdout: the headline model (mark, drawdown, PIK
    switch), all four signs in boosted trees, and the mark level alone."""
    series = [("headline", "logit", HEADLINE_LABEL, BLUE), ("combined", "gbt", "All four, boosted trees", ORANGE),
              ("price", "logit", "Mark level alone", AQUA)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.4), facecolor=SURFACE)
    _style(a1, "ROC curve", "Share of entries flagged (recall)")
    _style(a2, "Precision-recall curve", "Share of flagged borrowers that enter (precision)")
    base = None
    for spec, model, label, c in series:
        d = preds[(preds["spec"] == spec) & (preds["model"] == model)]
        if d.empty:
            continue
        y, p = d["y"].to_numpy(), d["pred"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        prec, rec = prec[:-1], rec[:-1]          # drop sklearn's added (recall 0, precision 1) end point
        a1.plot(fpr, tpr, color=c, linewidth=2, label=label, zorder=3)
        a2.plot(rec, prec, color=c, linewidth=2, label=label, zorder=3)
        base = y.mean()
    a1.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    a1.text(0.62, 0.55, "random", color=MUTED, fontsize=8, rotation=0)
    a1.set_xlabel("Share of non-entrants flagged (false positive rate)", color=MUTED, fontsize=9)
    a2.set_xlabel("Share of entries flagged (recall)", color=MUTED, fontsize=9)
    if base is not None:
        a2.axhline(base, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        a2.text(1.0, base, f"base rate {base:.1%}", ha="right", va="bottom", fontsize=8, color=MUTED,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0}, zorder=4)
    a1.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    _title(fig, f"Out of sample: predicting {title_q} non-accrual entries",
           "Fitted on earlier quarters only; one row per accruing borrower")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.12, wspace=0.22)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def holdout_auc(metrics: pd.DataFrame, path, title_q: str):
    """ROC-AUC and PR-AUC with bootstrap 95% intervals: each predictor alone, the headline model,
    then all four in each model."""
    d = metrics[(metrics["outcome"] == "y_na") & (metrics["sample"] == "all") & metrics["roc_auc"].notna()].copy()
    d["label"] = [f"{NAMES[s]} alone" if s in NAMES else HEADLINE_LABEL if s == "headline"
                  else f"All four: {MODEL_NAMES[mo]}" for s, mo in zip(d["spec"], d["model"])]
    d["order"] = [list(NAMES).index(s) if s in NAMES else 4 if s == "headline" else 5 + list(MODEL_NAMES).index(mo)
                  for s, mo in zip(d["spec"], d["model"])]
    d = d.sort_values("order", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), facecolor=SURFACE, sharey=True)
    y = np.arange(len(d))
    for ax, m, title, ref, reflabel in [(axes[0], "roc_auc", "ROC-AUC", 0.5, "random 0.50"),
                                        (axes[1], "pr_auc", "PR-AUC", d["base_rate"].iloc[0],
                                         f"base rate {d['base_rate'].iloc[0]:.3f}")]:
        _style(ax, title, "")
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.hlines(y, d[f"{m}_lo"], d[f"{m}_hi"], color=BLUE, linewidth=2, alpha=0.45, zorder=2)
        ax.scatter(d[m], y, s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        for yi, v in zip(y, d[m]):
            ax.text(v, yi + 0.22, f"{v:.2f}" if m == "roc_auc" else f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, color=INK)
        ax.axvline(ref, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(ref, len(d) - 0.35, f" {reflabel}", ha="left", va="center", fontsize=7.5, color=MUTED)
        ax.set_ylim(-0.6, len(d) - 0.1)
    axes[0].set_yticks(y, d["label"])
    axes[0].set_xlim(0.45, 1.0)
    _title(fig, f"Out-of-sample accuracy, {title_q} non-accrual entries",
           "Dot = estimate; bar = bootstrap 95% interval. Each sign alone, the three-sign model, then all four in "
           "each model")
    fig.subplots_adjust(left=0.24, right=0.98, top=0.83, bottom=0.08, wspace=0.08)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def odds_ratios(coefs: pd.DataFrame, path, window: str):
    """The headline model in one picture: odds ratios (95% CI, borrower-clustered) of the mark,
    the drawdown and a PIK switch, for entry next quarter and within 4 quarters. The mark is shown
    per 10 points lower, so every sign reads as 'times the odds'."""
    c = coefs[(coefs["spec"] == "headline") & (coefs["sample"] == "all") & coefs["term"].isin(HEADLINE_UNITS)].copy()
    lower = c["term"] == "price"            # flip the mark: per 10 points lower
    c.loc[lower, ["odds_ratio", "or_ci_low", "or_ci_high"]] = 1 / c.loc[lower, ["odds_ratio", "or_ci_high", "or_ci_low"]].to_numpy()
    terms = list(HEADLINE_UNITS)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    _style(ax, "", "")
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    for outcome, off, color, label in [("y_na", 0.15, BLUE, "next quarter"), ("y_na_h4", -0.15, AQUA, "within 4 quarters")]:
        d = c[c["outcome"] == outcome].set_index("term").reindex(terms)
        y = np.arange(len(terms)) + off
        ax.hlines(y, d["or_ci_low"], d["or_ci_high"], color=color, linewidth=2, alpha=0.5)
        ax.scatter(d["odds_ratio"], y, s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3, label=label)
        for yi, v in zip(y, d["odds_ratio"]):
            ax.text(v, yi + 0.07, f"{v:.1f}x", ha="center", va="bottom", fontsize=8, color=INK)
    ax.axvline(1, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_xticks([1, 1.5, 2, 3, 4, 5], ["1", "1.5", "2", "3", "4", "5"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(0.9, 6)
    ax.set_yticks(np.arange(len(terms)), [HEADLINE_UNITS[t] for t in terms])
    ax.set_xlabel("Times the odds of entering non-accrual (log scale; 1 = no effect)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    _title(fig, "Three warning signs, each multiplying the odds of non-accrual",
           f"Logistic regression on mark, drawdown and PIK switch together; accruing borrowers from {window.split('-')[0]}; "
           "95% intervals clustered by borrower")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.82, bottom=0.13)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def event_profile(p: pd.DataFrame, path, window: str):
    """Median mark and share with a PIK switch in the four quarters before a non-accrual entry,
    against borrowers that stayed accruing."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    _style(a1, "Median mark (FV / PAR, cents per dollar)", "")
    _style(a2, "Share with a PIK switch that quarter (%)", "")
    for group, color in [("stayed accruing", MUTED), ("entered non-accrual", BLUE)]:
        d = p[p["group"] == group].sort_values("k")
        a1.plot(d["k"], d["median_price"] * 100, color=color, linewidth=2, marker="o", markersize=7, zorder=3)
        if group == "entered non-accrual":
            a1.fill_between(d["k"], d["p25_price"] * 100, d["p75_price"] * 100, color=color, alpha=0.12, linewidth=0)
        a2.plot(d["k"], d["share_pik_switch"] * 100, color=color, linewidth=2, marker="o", markersize=7, zorder=3)
        for ax, v in [(a1, d["median_price"].iloc[-1] * 100), (a2, d["share_pik_switch"].iloc[-1] * 100)]:
            ax.text(0.15, v, group, va="center", fontsize=8.5, color=INK)
    for ax in (a1, a2):
        ax.set_xticks(range(-4, 1), ["-4", "-3", "-2", "-1", "entry\nquarter"])
        ax.set_xlim(-4.3, 1.6)
        ax.set_xlabel("Quarters before entering non-accrual", color=MUTED, fontsize=9)
    _title(fig, "Marks fall before borrowers enter non-accrual",
           "Borrowers entering non-accrual (shaded: middle half) vs borrowers that stayed accruing, same quarters")
    _note(fig, f"Entries in {window} from the labelled risk set. Mark = all BDCs' fair value / par on priceable loans.")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.17, wspace=0.18)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def horizon_review_lists(p: pd.DataFrame, common: list[int], quarters: str, path):
    """Review lists by horizon (the headline model, rolling out of sample, common test quarters):
    share of the entries caught and share of the list that enters, against the share of borrowers
    flagged each quarter."""
    shares = np.round(np.arange(0.005, 0.3001, 0.005), 3)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.0), facecolor=SURFACE)
    _style(a1, "Share of the entries the list catches (%)", "")
    _style(a2, "Share of the list that enters non-accrual (%)", "")
    ends = {a1: [], a2: []}
    for h, (label, color) in HORIZON_STYLE.items():
        g = p[(p["h"] == h) & (p["spec"] == "headline") & (p["model"] == "logit") & p["qsort"].isin(common)]
        if g.empty:
            continue
        c = review_capture(g, shares)
        x = c["share_flagged"] * 100
        for ax, col in [(a1, "capture"), (a2, "precision")]:
            ax.plot(x, c[col] * 100, color=color, linewidth=2, zorder=3)
            ends[ax].append((x.iloc[-1], c[col].iloc[-1] * 100, label))
    for ax, e in ends.items():       # end labels, nudged apart where lines finish close together
        lo, hi = ax.get_ylim()
        for (xe, _, label), y in zip(e, _spread([v for _, v, _ in e], gap=(hi - lo) * 0.05)):
            ax.text(xe + 0.5, y, label, va="center", fontsize=8.5, color=INK)
    for ax in (a1, a2):
        ax.axvline(5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xlim(0, 39)
        ax.set_xticks([0, 5, 10, 20, 30])
        ax.set_xlabel("Borrowers on the list each quarter (% of accruing borrowers, riskiest first)", color=MUTED,
                      fontsize=8.5)
    a1.set_ylim(0, 100)
    a1.text(5.4, 3, "top 5%", fontsize=7.5, color=MUTED)
    _title(fig, "What a review list catches, by how far ahead it looks",
           f"Mark, drawdown and PIK switch (logistic), rolling out of sample; test quarters {quarters}")
    _note(fig, "Each quarter the borrowers are ranked by the model fitted only on earlier, fully observed quarters. "
               "An entry 'within 4 quarters' counts in each of the quarters before it.")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.83, bottom=0.15, wspace=0.16)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def rolling_auc_by_quarter(p: pd.DataFrame, path):
    """ROC-AUC of each test quarter (entry next quarter, rolling out of sample): the headline model
    against the mark alone."""
    g = p[(p["h"] == 1) & (p["model"] == "logit") & p["spec"].isin(["headline", "price"])]
    a = (g.groupby(["spec", "outcome_q"]).apply(lambda x: roc_auc_score(x["y"], x["pred"])).rename("auc")
         .reset_index())
    n = g[g["spec"] == "headline"].groupby("outcome_q")["y"].sum()
    qs = sorted(a["outcome_q"].unique())
    x = {q: i for i, q in enumerate(qs)}
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    _style(ax, "", "ROC-AUC")
    ends = []
    for spec, label, color in [("price", "mark level alone", MUTED), ("headline", "mark, drawdown, PIK switch", BLUE)]:
        d = a[a["spec"] == spec].sort_values("outcome_q")
        ax.plot([x[q] for q in d["outcome_q"]], d["auc"], color=color, linewidth=2, marker="o", markersize=7, zorder=3)
        ends.append((d["auc"].iloc[-1], label))
    for (_, label), y in zip(ends, _spread([v for v, _ in ends], gap=0.03)):
        ax.text(len(qs) - 1 + 0.2, y, label, va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(qs)), [f"{q}\n{int(n[q])} entries" for q in qs], fontsize=8)
    ax.set_xlim(-0.4, len(qs) + 0.9)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    _title(fig, "Out of sample, quarter by quarter: entries into non-accrual next quarter",
           "Each quarter scored by a model fitted only on the quarters before it")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.17)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
