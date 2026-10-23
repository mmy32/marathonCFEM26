"""Which data-center-related borrowers owe BDCs the most, and to which managers.

Usage:
    python plot_data_center_lenders.py [quarter] [out_dir]

Defaults: the latest quarter (2026Q2), outputs/data_center_plots. Counts every tagged
investment (core + support + grey, any instrument), the same base as the other manager charts.
BDCs are grouped by asset manager with data_center_analysis.manager(), since a manager's funds
usually co-invest; the 5 managers named are the same as in plot_data_center_timeseries.py.

Writes:
    dc_borrower_lenders.csv   borrower x BDC: fair value (USD mm), manager, scope, all scopes included
    dc_borrowers.csv          one row per borrower: scope, function, fair value, BDCs, managers, note
    dc_top_borrowers.png      the largest borrowers, each bar split by lending manager
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

from data_center_analysis import FV, GRID, INK, MUTED, SURFACE, _style, load, manager
from paths import DATA_CENTER_PLOTS_DIR
from plot_data_center_timeseries import BDC_COLORS, OTHER_COLOR, Q_TO, TOP_MANAGERS, _title

TOP_BORROWERS = 15


def main():
    quarter = sys.argv[1] if len(sys.argv) > 1 else Q_TO
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA_CENTER_PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, _, names = load()
    q = rows[rows["cal_q"] == quarter].copy()
    q["bdc"] = q["cik"].map(names).fillna(q["cik"].astype(str))
    q["manager"] = [manager(c, b) for c, b in zip(q["cik"], q["bdc"])]

    detail = (q.groupby(["scope", "entity", "manager", "bdc"])[FV].sum().div(1e6).rename("fv_usd_mm")
              .reset_index().sort_values(["entity", "fv_usd_mm"], ascending=[True, False]))
    detail.to_csv(out_dir / "dc_borrower_lenders.csv", index=False, float_format="%.2f")

    # one row per borrower, largest first
    borrowers = (q.groupby("entity")
                 .agg(scope=("scope", "first"), function=("function", "first"), fv_usd_mm=(FV, "sum"),
                      bdcs=("cik", "nunique"), managers=("manager", lambda s: "; ".join(sorted(s.unique()))),
                      note=("note", "first"))
                 .sort_values("fv_usd_mm", ascending=False).reset_index())
    borrowers["fv_usd_mm"] /= 1e6
    borrowers.insert(4, "pct_of_dc_total", 100 * borrowers["fv_usd_mm"] / borrowers["fv_usd_mm"].sum())
    borrowers.to_csv(out_dir / "dc_borrowers.csv", index=False, float_format="%.2f")

    wide = q.pivot_table(index="entity", columns="manager", values=FV, aggfunc="sum", fill_value=0) / 1e6
    top_b = wide.sum(axis=1).nlargest(TOP_BORROWERS).index
    wide = wide.loc[top_b]
    top_m = q.groupby("manager")[FV].sum().nlargest(TOP_MANAGERS).index.tolist()
    wide = wide[top_m].assign(**{"Other managers": wide.drop(columns=top_m).sum(axis=1)})
    colors = dict(zip(top_m, BDC_COLORS)) | {"Other managers": OTHER_COLOR}

    fig, ax = plt.subplots(figsize=(11, 7.5))
    plot = wide.iloc[::-1]
    left = pd.Series(0.0, index=plot.index)
    for mgr in wide.columns:
        ax.barh(plot.index, plot[mgr], left=left, height=0.62, color=colors[mgr], edgecolor=SURFACE,
                linewidth=1.0, label=mgr)
        left = left + plot[mgr]
    for name, v in left.items():
        n_bdc = q.loc[q["entity"] == name, "cik"].nunique()
        ax.text(v, name, f"  ${v:,.0f}M  ({n_bdc} BDC{'s' if n_bdc > 1 else ''})", va="center", fontsize=8, color=MUTED)
    _style(ax, xlabel="USD millions, quarter-end fair value")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlim(0, left.max() * 1.22)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right", title="Lending manager",
              title_fontsize=8)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.88, bottom=0.08)
    _title(fig, f"Largest data-center-related borrowers from BDCs, {quarter}, by lending manager",
           f"All tagged investments (core + support + grey, any instrument); top {TOP_BORROWERS} borrowers. "
           "Each manager's BDCs are combined")
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(out_dir / "dc_top_borrowers.png", dpi=150)
    plt.close(fig)

    pd.set_option("display.width", 220)
    print(f"Top {TOP_BORROWERS} borrowers, {quarter}, USD mm by manager:")
    print(wide.assign(Total=wide.sum(axis=1)).round(0).to_string())
    print(f"\nWrote {out_dir / 'dc_top_borrowers.png'} and dc_borrower_lenders.csv")


if __name__ == "__main__":
    main()
