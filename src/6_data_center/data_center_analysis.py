"""Trend and BDC-exposure analysis of data-center-related private credit.

Usage:
    python data_center_analysis.py [out_dir]

Reads:
    data_center_loan_rows.csv             tagged rows from classify_data_center_borrowers.py (data/data_center)
    FINAL_DATASET.csv                     full panel, for each BDC's total portfolio (data/final_dataset)
    BDC_filings_2001_present.csv          CIK -> BDC name (data/universe)

Writes to out_dir (default: outputs/data_center_plots):
    trend_by_function.csv    quarter x function: fair value, cost, net new cost, borrowers, BDCs,
                             share of all BDC fair value
    bdc_exposure.csv         one row per BDC at the latest quarter: manager, $ by function, % of portfolio
    bdc_exposure_by_manager.csv  one row per manager at the latest quarter (its BDCs combined)
    bdc_exposure_panel.csv   BDC x quarter: data center fair value and % of portfolio
    trend_by_function.png, trend_share_of_portfolio.png,
    bdc_exposure_usd.png, bdc_exposure_pct.png (both by manager, each manager's BDCs combined)

Amounts are balance-sheet stocks (what BDCs hold at quarter end), not originations.
These tables count every tagged borrower, i.e. the upper-bound set (core + support + grey);
plot_data_center_timeseries.py splits the three scopes and charts the headline (core + support).
"Net new cost" is the quarter-over-quarter change in cost basis, so it nets fundings
against repayments and exits, and also moves when a BDC enters or leaves the panel.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paths import DATA_CENTER_DIR, DATA_CENTER_PLOTS_DIR, FINAL_DIR, UNIVERSE_DIR

FV = "InvestmentOwnedAtFairValue_normalized"
COST = "InvestmentOwnedAtCost_normalized"

# tier label from the classifier -> short function name, in fixed display / color order
FUNCTIONS = {
    "1 Operator / developer": "Operator / developer",
    "2 AI compute / GPU cloud / miners": "AI compute",
    "4 DC services / supply chain": "DC services / supply chain",
}
COLORS = {"Operator / developer": "#2a78d6", "AI compute": "#eb6834", "DC services / supply chain": "#eda100"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

TOP_N = 15
# BDCs smaller than this are left out of the %-of-portfolio ranking (a single loan dominates them)
MIN_PORTFOLIO_USD = 100e6

# BDC -> asset manager; the map is shared with crowding and dispersion (one family = one holder)
from manager_family import manager  # noqa: E402


def load():
    rows = pd.read_csv(DATA_CENTER_DIR / "data_center_loan_rows.csv", low_memory=False)
    rows["function"] = rows["tier"].map(FUNCTIONS)
    rows = rows[rows[FV].notna() | rows[COST].notna()]

    panel = pd.read_csv(FINAL_DIR / "FINAL_DATASET.csv", usecols=["cik", "cal_q", FV])
    totals = panel.groupby(["cik", "cal_q"])[FV].sum().rename("portfolio_fv").reset_index()

    names = pd.read_csv(UNIVERSE_DIR / "BDC_filings_2001_present.csv", dtype=str)
    names["cik"] = names["CIK"].astype(int)
    names = names.sort_values("Date").groupby("cik")["Company"].last().str.title()
    names = names.str.replace(r"\b(Bdc|Llc|Ii|Us|Hms|Kkr|Hps|Agl|Gso|Fs|Cm|Ul)\b", lambda m: m.group().upper(), regex=True)
    return rows, totals, names


def trend_by_function(rows, totals):
    out = (
        rows.groupby(["cal_q", "function"])
        .agg(fv_usd_mm=(FV, "sum"), cost_usd_mm=(COST, "sum"), borrowers=("entity", "nunique"), bdcs=("cik", "nunique"))
        .reset_index()
    )
    out[["fv_usd_mm", "cost_usd_mm"]] /= 1e6
    # every quarter x function present, so the diff below never skips a gap
    idx = pd.MultiIndex.from_product([sorted(out["cal_q"].unique()), FUNCTIONS.values()], names=["cal_q", "function"])
    out = out.set_index(["cal_q", "function"]).reindex(idx, fill_value=0).reset_index()
    out["net_new_cost_usd_mm"] = out.groupby("function")["cost_usd_mm"].diff()
    all_fv = totals.groupby("cal_q")["portfolio_fv"].sum() / 1e6
    out["pct_of_all_bdc_fv"] = 100 * out["fv_usd_mm"] / out["cal_q"].map(all_fv)
    return out


def bdc_exposure(rows, totals, names):
    panel = rows.groupby(["cik", "cal_q"]).agg(dc_fv=(FV, "sum"), borrowers=("entity", "nunique")).reset_index()
    panel = panel.merge(totals, on=["cik", "cal_q"], how="left")
    panel["pct_of_portfolio"] = 100 * panel["dc_fv"] / panel["portfolio_fv"]
    panel.insert(1, "bdc", panel["cik"].map(names))
    panel[["dc_fv", "portfolio_fv"]] /= 1e6
    panel = panel.rename(columns={"dc_fv": "dc_fv_usd_mm", "portfolio_fv": "portfolio_fv_usd_mm"})

    latest_q = rows["cal_q"].max()
    cur = rows[rows["cal_q"] == latest_q]
    by_fn = (cur.pivot_table(index="cik", columns="function", values=FV, aggfunc="sum", fill_value=0) / 1e6)
    by_fn = by_fn.reindex(columns=list(FUNCTIONS.values()), fill_value=0)
    top = cur.groupby(["cik", "entity"])[FV].sum().reset_index().sort_values(FV).groupby("cik")["entity"].last()
    latest = panel[panel["cal_q"] == latest_q].set_index("cik").join(by_fn).join(top.rename("largest_borrower"))
    latest = latest.sort_values("dc_fv_usd_mm", ascending=False).reset_index()
    latest.insert(2, "manager", [manager(c, b) for c, b in zip(latest["cik"], latest["bdc"])])
    return panel.sort_values(["cal_q", "dc_fv_usd_mm"], ascending=[True, False]), latest


def manager_exposure(latest, totals, names, latest_q):
    """One row per manager at the latest quarter: every BDC of the manager combined.
    The portfolio counts all of the manager's BDCs that filed, including ones with no data center holdings."""
    pf = totals[totals["cal_q"] == latest_q].copy()
    pf["manager"] = [manager(c, names.get(c)) for c in pf["cik"]]
    pf = pf.groupby("manager").agg(portfolio_fv_usd_mm=("portfolio_fv", "sum"), bdcs_filed=("cik", "nunique"))
    pf["portfolio_fv_usd_mm"] /= 1e6
    out = latest.groupby("manager").agg(dc_fv_usd_mm=("dc_fv_usd_mm", "sum"), bdcs_with_dc=("cik", "nunique"),
                                        bdc_names=("bdc", lambda s: "; ".join(s.fillna("?"))))
    out = out.join(pf, how="left")
    out["pct_of_portfolio"] = 100 * out["dc_fv_usd_mm"] / out["portfolio_fv_usd_mm"]
    return out.sort_values("dc_fv_usd_mm", ascending=False).reset_index()


def _style(ax, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_axisbelow(True)


def _save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _stacked(ax, wide, horizontal=False):
    """Stacked bars in FUNCTIONS order with a surface-colored gap between segments."""
    base = pd.Series(0.0, index=wide.index)
    for fn in FUNCTIONS.values():
        kw = dict(color=COLORS[fn], edgecolor=SURFACE, linewidth=1.2, label=fn)
        if horizontal:
            ax.barh(wide.index, wide[fn], left=base, height=0.7, **kw)
        else:
            ax.bar(wide.index, wide[fn], bottom=base, width=0.7, **kw)
        base = base + wide[fn]
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left" if not horizontal else "lower right")
    return base


def plot_trend(trend, out_dir):
    for col, fname, title, unit, fmt in [
        ("fv_usd_mm", "trend_by_function.png",
         "BDC fair value in data-center-related borrowers, by function", "USD billions", "${:.1f}B"),
        ("pct_of_all_bdc_fv", "trend_share_of_portfolio.png",
         "Data-center-related borrowers as a share of all BDC fair value", "% of fair value", "{:.1f}%"),
    ]:
        wide = trend.pivot(index="cal_q", columns="function", values=col)[list(FUNCTIONS.values())]
        if col == "fv_usd_mm":
            wide = wide / 1e3
        fig, ax = plt.subplots(figsize=(11, 5.5))
        total = _stacked(ax, wide)
        for q, v in total.items():
            ax.text(q, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=MUTED)
        _style(ax, ylabel=unit)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        _save(fig, out_dir / fname)


def plot_exposure(latest, by_mgr, latest_q, out_dir, all_bdc_fv_usd_mm):
    label = latest["bdc"].fillna(latest["cik"].astype(str)).str.slice(0, 45)

    top = by_mgr.head(TOP_N).iloc[::-1]
    total = top.set_index(top["manager"].str.slice(0, 45))["dc_fv_usd_mm"] / 1e3
    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.barh(total.index, total, height=0.7, color=COLORS["Operator / developer"])
    for (name, v), pf, pct, n in zip(total.items(), top["portfolio_fv_usd_mm"], top["pct_of_portfolio"],
                                     top["bdcs_with_dc"]):
        # "\$": two dollar signs in one string would otherwise switch matplotlib into math mode
        bdcs = f", {n} BDCs" if n > 1 else ""
        ax.text(v, name, f"  \\${v:.2f}B of \\${pf / 1e3:.1f}B portfolio ({pct:.1f}%){bdcs}",
                va="center", fontsize=8, color=MUTED)
    _style(ax, xlabel="USD billions")
    ax.tick_params(axis="y", labelcolor=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlim(0, total.max() * 1.5)
    dc_total = latest["dc_fv_usd_mm"].sum()
    ax.set_title(f"Largest data center exposure by manager, {latest_q} (each manager's BDCs combined)\n"
                 f"All BDCs: \\${dc_total / 1e3:.2f}B in data-center-related investments, "
                 f"{100 * dc_total / all_bdc_fv_usd_mm:.2f}% of \\${all_bdc_fv_usd_mm / 1e3:,.0f}B total BDC fair value",
                 loc="left", color=INK, fontsize=11, pad=12)
    _save(fig, out_dir / "bdc_exposure_usd.png")

    big = by_mgr[by_mgr["portfolio_fv_usd_mm"] >= MIN_PORTFOLIO_USD / 1e6]
    top = big.sort_values("pct_of_portfolio", ascending=False).head(TOP_N).iloc[::-1]
    names = top["manager"].str.slice(0, 45)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(names, top["pct_of_portfolio"], height=0.7, color=COLORS["Operator / developer"])
    for name, v, dc, pf, n in zip(names, top["pct_of_portfolio"], top["dc_fv_usd_mm"], top["portfolio_fv_usd_mm"],
                                  top["bdcs_with_dc"]):
        bdcs = f", {n} BDCs" if n > 1 else ""
        pf_txt = f"\\${pf / 1e3:.1f}B" if pf >= 1e3 else f"\\${pf:,.0f}M"
        ax.text(v, name, f"  {v:.1f}%  (\\${dc:,.0f}M of {pf_txt}{bdcs})", va="center", fontsize=8, color=MUTED)
    _style(ax, xlabel="% of the manager's total BDC fair value")
    ax.tick_params(axis="y", labelcolor=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlim(0, top["pct_of_portfolio"].max() * 1.4)
    ax.set_title(f"Most concentrated in data-center-related investments by manager, {latest_q}\n"
                 f"Each manager's BDCs combined; combined portfolios of \\${MIN_PORTFOLIO_USD / 1e6:.0f}M or more",
                 loc="left", color=INK, fontsize=11, pad=12)
    _save(fig, out_dir / "bdc_exposure_pct.png")


def main():
    out_dir = DATA_CENTER_PLOTS_DIR
    if len(sys.argv) > 1:
        from pathlib import Path
        out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, totals, names = load()
    latest_q = rows["cal_q"].max()
    trend = trend_by_function(rows, totals)
    panel, latest = bdc_exposure(rows, totals, names)
    by_mgr = manager_exposure(latest, totals, names, latest_q)

    trend.to_csv(out_dir / "trend_by_function.csv", index=False, float_format="%.2f")
    latest.to_csv(out_dir / "bdc_exposure.csv", index=False, float_format="%.2f")
    by_mgr.to_csv(out_dir / "bdc_exposure_by_manager.csv", index=False, float_format="%.2f")
    panel.to_csv(out_dir / "bdc_exposure_panel.csv", index=False, float_format="%.2f")
    plot_trend(trend, out_dir)
    plot_exposure(latest, by_mgr, latest_q, out_dir,
                  totals.loc[totals["cal_q"] == latest_q, "portfolio_fv"].sum() / 1e6)

    pd.set_option("display.width", 200)
    print(f"Fair value by function, USD mm (latest quarter {latest_q}):")
    print(trend.pivot(index="cal_q", columns="function", values="fv_usd_mm")[list(FUNCTIONS.values())]
          .assign(Total=lambda d: d.sum(axis=1)).round(0).to_string())
    print(f"\nTop {TOP_N} managers by data center fair value, {latest_q} (each manager's BDCs combined):")
    print(by_mgr.head(TOP_N)[["manager", "dc_fv_usd_mm", "portfolio_fv_usd_mm", "pct_of_portfolio", "bdcs_with_dc"]]
          .round(1).to_string(index=False))
    big = by_mgr[by_mgr["portfolio_fv_usd_mm"] >= MIN_PORTFOLIO_USD / 1e6]
    print(f"\nTop {TOP_N} managers by share of portfolio, {latest_q}:")
    print(big.sort_values("pct_of_portfolio", ascending=False).head(TOP_N)
          [["manager", "pct_of_portfolio", "dc_fv_usd_mm", "portfolio_fv_usd_mm", "bdcs_with_dc"]].round(1).to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
