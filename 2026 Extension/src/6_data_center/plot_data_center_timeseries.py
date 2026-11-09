"""BDC loans to data-center-related borrowers, 2023Q1-2026Q2 (slide 17), as three nested estimates: core,
core + support (headline), and upper bound (+ grey). Writes dc_loans_total.png and
dc_loans_timeseries.csv to outputs/data_center_plots/ and prints the largest holders."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_center_analysis import FUNCTIONS, FV, GRID, INK, MUTED, SURFACE, _style, load
from manager_family import manager   # BDC -> asset manager, shared with crowding and dispersion
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
    "grey": "Grey area: multi-market suppliers",
}
SCOPE_COLORS = dict(zip(SCOPES, ["#104281", "#2a78d6", "#86b6ef"]))
HEADLINE = ["core", "support"]
TOP_N = 7  # largest BDCs printed by name
TOP_MANAGERS = 5  # largest managers printed by name
# categorical slots 1-7 (validated as adjacent pairs, light surface); gray for the folded tail
BDC_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
OTHER_COLOR = "#c3c2b7"


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


def _title(fig, title, subtitle, size=13, sub_size=9, sub_y=0.925):
    fig.text(0.012, 0.975, title, ha="left", va="top", color=INK, fontsize=size)
    fig.text(0.012, sub_y, subtitle, ha="left", va="top", color=MUTED, fontsize=sub_size)


def _big_fonts(ax):
    """Larger tick and axis-label text for the two total charts (slide-ready)."""
    ax.tick_params(labelsize=13)
    ax.yaxis.label.set_fontsize(14)


def plot_total(ts, out_dir):
    x = range(len(QUARTERS))
    end = len(QUARTERS) - 1
    fig, ax = plt.subplots(figsize=(11, 6))
    base = pd.Series(0.0, index=ts.index)
    for scope, label in SCOPES.items():
        y = ts[scope] / 1e3
        ax.bar(x, y, bottom=base, width=0.62, color=SCOPE_COLORS[scope], edgecolor=SURFACE, linewidth=1.0,
               label=label)
        base = base + y
    for col, name in (("core", "Core"), ("headline", "Core + Support"), ("upper_bound", "Upper bound")):
        v = ts[col].iloc[-1] / 1e3
        ax.annotate(f"{name} ${v:.2f}B ({ts['pct_' + col].iloc[-1]:.2f}%)", (end + 0.31, v), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=12, color=INK if col == "headline" else MUTED)
    ax.set_ylim(0, base.max() * 1.12)
    _style(ax, ylabel="USD billions")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    _big_fonts(ax)
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], frameon=False, fontsize=13, labelcolor=INK, loc="upper left")
    _quarter_axis(ax)
    ax.set_xlim(-0.6, len(QUARTERS) + 4.6)  # room for the end labels
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.15)
    _title(fig, f"Fair value of BDC loans to data-center-related borrowers, {Q_FROM} to {Q_TO}",
           "Three nested estimates: core, core + support, upper bound (+ grey area).\n"
           "Labels give USD and % of all BDC fair value", size=19, sub_size=13, sub_y=0.915)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(out_dir / "dc_loans_total.png", dpi=150)
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


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_CENTER_PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, totals, names = load()
    loans = loans_only(rows)
    ts = quarterly_totals(loans, totals)
    dc, portfolio = by_bdc(loans, totals)
    headline = dc["core"] + dc["support"]
    share = 100 * headline / portfolio
    top = headline[Q_TO].nlargest(TOP_N).index.tolist()
    dc_m, pf_m, share_m = manager_exposure(rows, totals, names)
    top_m = dc_m[Q_TO].nlargest(TOP_MANAGERS).index.tolist()

    ts.to_csv(out_dir / "dc_loans_timeseries.csv", float_format="%.2f")
    plot_total(ts, out_dir)

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
