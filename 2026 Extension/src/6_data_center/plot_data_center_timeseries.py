"""Time series of BDC loans to data-center-related borrowers, 2023Q1 to 2026Q2.

Usage:
    python plot_data_center_timeseries.py [out_dir]

Reads, through data_center_analysis.load():
    data_center_loan_rows.csv      tagged rows from classify_data_center_borrowers.py
    FINAL_DATASET.csv              full panel, for each BDC's total portfolio fair value
    BDC_filings_2001_present.csv   CIK -> BDC name

A loan is a tagged row that is not equity and carries a principal amount, an interest
rate or a spread. Amounts are quarter-end fair values (stocks, not originations).
The classifier's scope column gives three nested estimates: core; headline = core +
operations support (the conservative estimate, used by the BDC charts); upper bound =
headline + grey area.

Writes to out_dir (default: outputs/data_center_plots):
    dc_loans_timeseries.csv          quarter: each layer and nested estimate, shares of all BDC fair value,
                                     and the headline split by function
    dc_loans_by_bdc_timeseries.csv   BDC x quarter: fair value by layer, portfolio fair value, % of portfolio
    dc_exposure_by_manager_timeseries.csv  manager x quarter: all data-center-related fair value, portfolio, %
    dc_loans_total.png               the three layers over time, and the headline split by function
    dc_loans_by_bdc.png              all data-center-related investments by manager: the 5 largest at the latest
                                     quarter + all other managers
    dc_exposure_share.png            the same investments as a % of portfolio fair value: all BDCs, then the
                                     same 5 managers

The manager charts (these two, bdc_exposure_*.png, dc_top_borrowers.png) count every tagged
investment, any scope or instrument: the criterion is whether the borrower affects data center
operations. Each manager's BDCs are combined (data_center_analysis.manager()).
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
from matplotlib.ticker import MaxNLocator, PercentFormatter

from data_center_analysis import COLORS, FUNCTIONS, FV, GRID, INK, MUTED, SURFACE, _style, load, manager
from paths import DATA_CENTER_PLOTS_DIR

Q_FROM, Q_TO = "2023Q1", "2026Q2"
QUARTERS = [str(q) for q in pd.period_range(Q_FROM, Q_TO, freq="Q")]
DEBT_FIELDS = [
    "InvestmentOwnedBalancePrincipalAmount_normalized",
    "InvestmentInterestRate_normalized",
    "InvestmentBasisSpreadVariableRate_normalized",
]
# scope -> legend label, bottom to top; ordinal blue ramp (validated --ordinal), darkest = most certain
SCOPES = {
    "core": "Core: operators, developers, AI compute",
    "support": "Operations support: data centers are the main market",
    "grey": "Grey area: multi-market suppliers, data center sidelines",
}
SCOPE_COLORS = dict(zip(SCOPES, ["#104281", "#2a78d6", "#86b6ef"]))
HEADLINE = ["core", "support"]
TOP_N = 7  # BDCs shown by name in dc_loans_by_bdc.png; the rest fold into "All other BDCs"
TOP_MANAGERS = 5  # managers shown by name in dc_loans_by_bdc, dc_exposure_share and dc_top_borrowers
# categorical slots 1-7 (validated as adjacent pairs, light surface); gray for the folded tail
BDC_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
OTHER_COLOR = "#c3c2b7"
AREA_ALPHA = 0.10


def loans_only(rows):
    is_loan = (rows["instrument_type"] != "Equity") & rows[DEBT_FIELDS].notna().any(axis=1)
    dropped = rows.loc[~is_loan & rows["cal_q"].isin(QUARTERS), FV].sum() / 1e6
    print(f"Left out {(~is_loan).sum()} non-loan rows (${dropped:,.1f}M fair value summed over all quarters)")
    return rows[is_loan & rows["cal_q"].isin(QUARTERS)]


def quarterly_totals(loans, totals):
    """Quarter x layer and nested estimates (USD mm), shares of all BDC fair value, headline by function."""
    ts = (loans.pivot_table(index="cal_q", columns="scope", values=FV, aggfunc="sum")
          .reindex(index=QUARTERS, columns=list(SCOPES)).fillna(0) / 1e6)
    ts["headline"] = ts[HEADLINE].sum(axis=1)
    ts["upper_bound"] = ts["headline"] + ts["grey"]
    ts["all_bdc_fv"] = totals.groupby("cal_q")["portfolio_fv"].sum().reindex(QUARTERS) / 1e6
    for col in ("core", "headline", "upper_bound"):
        ts[f"pct_{col}"] = 100 * ts[col] / ts["all_bdc_fv"]
    head = loans[loans["scope"].isin(HEADLINE)]
    by_fn = (head.pivot_table(index="cal_q", columns="function", values=FV, aggfunc="sum")
             .reindex(index=QUARTERS, columns=list(FUNCTIONS.values())).fillna(0) / 1e6)
    ts = ts.join(by_fn.add_prefix("headline: "))
    ts["headline_borrowers"] = head.groupby("cal_q")["entity"].nunique().reindex(QUARTERS).fillna(0).astype(int)
    ts["headline_bdcs"] = head.groupby("cal_q")["cik"].nunique().reindex(QUARTERS).fillna(0).astype(int)
    ts.index.name = "cal_q"
    return ts


def by_bdc(loans, totals):
    """BDC x quarter fair value (USD mm) per scope: 0 when the BDC filed but held none, NaN when it did not file."""
    portfolio = totals.pivot(index="cik", columns="cal_q", values="portfolio_fv").reindex(columns=QUARTERS) / 1e6
    dc = {}
    for scope in SCOPES:
        part = loans[loans["scope"] == scope]
        dc[scope] = (part.groupby(["cik", "cal_q"])[FV].sum().unstack("cal_q")
                     .reindex(index=portfolio.index, columns=QUARTERS).fillna(0) / 1e6).where(portfolio.notna())
    return dc, portfolio


def _quarter_axis(ax, every=1):
    ax.set_xticks(range(len(QUARTERS)))
    ax.set_xticklabels([q if i % every == 0 else "" for i, q in enumerate(QUARTERS)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlim(-0.6, len(QUARTERS) - 0.4)


def _title(fig, title, subtitle):
    fig.text(0.012, 0.975, title, ha="left", va="top", color=INK, fontsize=13)
    fig.text(0.012, 0.925, subtitle, ha="left", va="top", color=MUTED, fontsize=9)


def _line(ax, x, y, color, area=False):
    ax.plot(x, y, color=color, lw=1.5, solid_joinstyle="round", solid_capstyle="round", zorder=3)
    ax.plot(x, y, "o", ms=4.5, color=color, mec=SURFACE, mew=1, zorder=4)
    if area:
        ax.fill_between(x, y, color=color, alpha=AREA_ALPHA, lw=0, zorder=2)


def plot_total(ts, out_dir):
    x = range(len(QUARTERS))
    end = len(QUARTERS) - 1
    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"hspace": 0.3})

    base = pd.Series(0.0, index=ts.index)
    for scope, label in SCOPES.items():
        y = ts[scope] / 1e3
        top.bar(x, y, bottom=base, width=0.62, color=SCOPE_COLORS[scope], edgecolor=SURFACE, linewidth=1.0,
                label=label)
        base = base + y
    for col, name in (("core", "Core"), ("headline", "Headline"), ("upper_bound", "Upper bound")):
        v = ts[col].iloc[-1] / 1e3
        top.annotate(f"{name} ${v:.2f}B ({ts['pct_' + col].iloc[-1]:.2f}%)", (end + 0.31, v), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=8, color=INK if col == "headline" else MUTED)
    top.set_ylim(0, base.max() * 1.12)
    _style(top, ylabel="USD billions")
    top.yaxis.grid(True, color=GRID, linewidth=0.8)
    h, lab = top.get_legend_handles_labels()
    top.legend(h[::-1], lab[::-1], frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    top.set_title("Three nested estimates: core, headline (core + support), upper bound (+ grey area)",
                  loc="left", color=INK, fontsize=10)

    # end values go in the legend: the series converge at the right edge, so end labels would collide
    fns = [fn for fn in FUNCTIONS.values() if ts[f"headline: {fn}"].any()]
    for fn in fns:
        _line(bot, x, ts[f"headline: {fn}"] / 1e3, COLORS[fn])
    bot.set_ylim(0, (ts[[f"headline: {fn}" for fn in fns]] / 1e3).max().max() * 1.15)
    _style(bot, ylabel="USD billions")
    bot.yaxis.grid(True, color=GRID, linewidth=0.8)
    bot.set_title(f"Headline estimate by borrower function (legend: {Q_TO} value)", loc="left", color=INK, fontsize=10)
    handles = [plt.Line2D([], [], color=COLORS[fn], lw=1.5, marker="o", ms=4.5, mec=SURFACE) for fn in fns]
    bot.legend(handles, [f"{fn}  ${ts[f'headline: {fn}'].iloc[-1] / 1e3:.2f}B" for fn in fns],
               frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    _quarter_axis(bot)
    for ax in (top, bot):
        ax.set_xlim(-0.6, len(QUARTERS) + 2.9)  # room for the end labels

    fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.1)
    _title(fig, f"Fair value of BDC loans to data-center-related borrowers, {Q_FROM} to {Q_TO}",
           "Quarter-end fair value of loans held by all BDCs in the panel; labels give USD and % of all BDC fair value")
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(out_dir / "dc_loans_total.png", dpi=150)
    plt.close(fig)


def plot_by_manager(dc_m, top, out_dir):
    wide = dc_m.loc[top].T.fillna(0) / 1e3
    wide["All other managers"] = dc_m.drop(index=top).sum().reindex(QUARTERS).fillna(0) / 1e3
    colors = BDC_COLORS[:len(top)] + [OTHER_COLOR]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(QUARTERS))
    base = pd.Series(0.0, index=wide.index)
    for col, color in zip(wide.columns, colors):
        ax.bar(x, wide[col], bottom=base, width=0.62, color=color, edgecolor=SURFACE, linewidth=1.0, label=col)
        base = base + wide[col]
    for i in (0, len(QUARTERS) - 1):
        ax.text(i, base.iloc[i], f"${base.iloc[i]:.2f}B", ha="center", va="bottom", fontsize=9, color=INK)
    _style(ax, ylabel="USD billions")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_ylim(0, base.max() * 1.12)
    _quarter_axis(ax)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.14)
    _title(fig, "Fair value of data-center-related investments by manager",
           f"All tagged investments (core + support + grey, any instrument). The {len(top)} largest managers at "
           f"{Q_TO}, each manager's BDCs combined; the rest in one band")
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(out_dir / "dc_loans_by_bdc.png", dpi=150)
    plt.close(fig)


def manager_exposure(rows, totals, names):
    """Manager x quarter over all data-center-related investments (every scope and instrument type):
    fair value (USD mm), portfolio fair value of all the manager's BDCs that filed, and % of portfolio.
    NaN where none of the manager's BDCs filed that quarter."""
    dc = rows[rows["cal_q"].isin(QUARTERS)].groupby(["cik", "cal_q"])[FV].sum().rename("dc")
    df = totals[totals["cal_q"].isin(QUARTERS)].set_index(["cik", "cal_q"]).join(dc, how="outer")
    missing = df.loc[df["portfolio_fv"].isna(), "dc"].sum()
    assert missing == 0, f"${missing / 1e6:,.1f}M of data center fair value has no portfolio total"
    df["dc"] = df["dc"].fillna(0)
    df["manager"] = [manager(c, names.get(c)) for c in df.index.get_level_values("cik")]
    g = df.groupby(["manager", "cal_q"])[["dc", "portfolio_fv"]].sum() / 1e6
    dc_m = g["dc"].unstack("cal_q").reindex(columns=QUARTERS)
    pf_m = g["portfolio_fv"].unstack("cal_q").reindex(columns=QUARTERS)
    return dc_m, pf_m, 100 * dc_m / pf_m


def plot_share(dc_m, pf_m, share_m, top, out_dir):
    all_pct = 100 * dc_m.sum() / pf_m.sum()
    panels = [("All BDCs (aggregate)", all_pct, dc_m.sum())] + [(m, share_m.loc[m], dc_m.loc[m]) for m in top]
    ncol = 3
    nrow = -(-len(panels) // ncol)
    height = 3.1 * nrow + 1.3
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, height), sharex=True)
    x = pd.Series(range(len(QUARTERS)), index=QUARTERS)
    for ax, (label, y, usd) in zip(axes.flat, panels):
        y = y.reindex(QUARTERS)
        color = INK if label.startswith("All BDCs") else COLORS["Operator / developer"]
        ok = y.notna()
        _line(ax, x[ok], y[ok], color, area=True)
        last = y.dropna()
        ymax = max(y.max() * 1.15, 0.5)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))  # no 0.25 steps: labels have 1 decimal
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0 if ymax >= 5 else 1))
        _style(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_title(label[:40], loc="left", color=INK, fontsize=10, pad=17)
        if len(last):
            q = last.index[-1]
            ax.text(0, 1.03, f"{last.iloc[-1]:.1f}% (\\${usd[q] / 1e3:.2f}B) at {q}", transform=ax.transAxes,
                    va="bottom", color=MUTED, fontsize=8)
        _quarter_axis(ax, every=4)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.05, right=0.99, top=1 - 1.3 / height, bottom=0.11, hspace=0.5, wspace=0.22)
    _title(fig, "Data-center-related investments as a share of portfolio fair value, by manager",
           f"All tagged investments (core + support + grey, any instrument). All BDCs combined, then the "
           f"{len(top)} largest managers at {Q_TO}, each manager's BDCs combined. Own y-axis per panel")
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(out_dir / "dc_exposure_share.png", dpi=150)
    plt.close(fig)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_CENTER_PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, totals, names = load()
    loans = loans_only(rows)
    ts = quarterly_totals(loans, totals)
    dc, portfolio = by_bdc(loans, totals)
    headline = dc["core"] + dc["support"]
    upper = headline + dc["grey"]
    share = 100 * headline / portfolio
    top = headline[Q_TO].nlargest(TOP_N).index.tolist()
    dc_m, pf_m, share_m = manager_exposure(rows, totals, names)
    top_m = dc_m[Q_TO].nlargest(TOP_MANAGERS).index.tolist()

    ts.to_csv(out_dir / "dc_loans_timeseries.csv", float_format="%.2f")
    panel = pd.concat({**{f"dc_{s}_usd_mm": dc[s].stack() for s in SCOPES},
                       "dc_headline_usd_mm": headline.stack(), "dc_upper_bound_usd_mm": upper.stack(),
                       "portfolio_fv_usd_mm": portfolio.stack(), "pct_headline": share.stack(),
                       "pct_upper_bound": (100 * upper / portfolio).stack()}, axis=1).reset_index()
    panel.insert(1, "bdc", panel["cik"].map(names))
    panel = panel[panel["dc_upper_bound_usd_mm"] > 0].sort_values(["cal_q", "dc_headline_usd_mm"],
                                                                  ascending=[True, False])
    panel.to_csv(out_dir / "dc_loans_by_bdc_timeseries.csv", index=False, float_format="%.2f")
    pd.concat({"dc_all_usd_mm": dc_m.stack(), "portfolio_fv_usd_mm": pf_m.stack(), "pct_of_portfolio": share_m.stack()},
              axis=1).reset_index().to_csv(out_dir / "dc_exposure_by_manager_timeseries.csv", index=False,
                                           float_format="%.2f")

    plot_total(ts, out_dir)
    plot_by_manager(dc_m, top_m, out_dir)
    plot_share(dc_m, pf_m, share_m, top_m, out_dir)

    pd.set_option("display.width", 200)
    print(f"\nData-center-related loan fair value, USD mm, {Q_FROM}-{Q_TO}:")
    print(ts[list(SCOPES) + ["headline", "upper_bound", "pct_core", "pct_headline", "pct_upper_bound",
                             "headline_borrowers", "headline_bdcs"]].round(2).to_string())
    print(f"\nLargest holders at {Q_TO}, headline estimate (USD mm, % of portfolio):")
    print(pd.DataFrame({"bdc": [names.get(c, c) for c in top], "dc_loan_fv": headline.loc[top, Q_TO].round(1).values,
                        "pct": share.loc[top, Q_TO].round(1).values}).to_string(index=False))
    print(f"\nLargest managers at {Q_TO}, all data-center-related investments (USD mm, % of portfolio):")
    print(pd.DataFrame({"dc_fv": dc_m.loc[top_m, Q_TO].round(1), "pct": share_m.loc[top_m, Q_TO].round(1)}).to_string())
    print(f"All BDCs: {dc_m[Q_TO].sum():,.1f}")
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
