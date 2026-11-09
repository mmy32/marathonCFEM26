"""Data-center-related borrowers by sector, 2026Q2 (slide 18): every tagged investment (core + support +
grey), unresolved sectors filled from the borrower's other rows. Writes dc_sector_mix.png/.csv and
dc_sector_fills.csv to outputs/data_center_plots/. Usage: python plot_data_center_sectors.py [quarter] [out_dir]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_center_analysis import FV, INK, MUTED, SURFACE, load
from paths import DATA_CENTER_PLOTS_DIR
from plot_data_center_timeseries import BDC_COLORS, OTHER_COLOR, Q_TO, _title

OTHER = "Unresolved / other"
MIN_SHARE = 1.0  # % of fair value; smaller sectors fold into OTHER
UNRESOLVED = "Other / Unknown / Unresolved"


def fill_sectors(rows):
    """Give each unresolved row the sector its borrower has in its other rows (any quarter).

    A borrower's sector is the resolved sector holding the most fair value across all its rows.
    Rows that already have a sector keep it; borrowers never resolved anywhere stay unresolved.
    """
    known = rows[rows["sector"] != UNRESOLVED]
    weight = known[FV].fillna(0).clip(lower=0) + 1  # +1 so rows without a fair value still count
    best = (weight.groupby([known["entity"], known["sector"]]).sum()
            .sort_values(ascending=False).reset_index().drop_duplicates("entity")
            .set_index("entity")["sector"])
    out = rows["sector"].copy()
    gap = out == UNRESOLVED
    out[gap] = rows.loc[gap, "entity"].map(best).fillna(UNRESOLVED)
    return out


def main():
    quarter = sys.argv[1] if len(sys.argv) > 1 else Q_TO
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA_CENTER_PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, _, _ = load()
    rows["sector"] = rows["sector"].fillna(UNRESOLVED)
    rows["sector_filled"] = fill_sectors(rows)
    q = rows[rows["cal_q"] == quarter].copy()

    filled = q[q["sector"] != q["sector_filled"]]
    (filled.groupby(["entity", "sector_filled"])[FV].sum().div(1e6).rename("fv_usd_mm").reset_index()
     .to_csv(out_dir / "dc_sector_fills.csv", index=False, float_format="%.2f"))
    q["sector"] = q["sector_filled"]

    table = q.pivot_table(index="sector", columns="scope", values=FV, aggfunc="sum", fill_value=0) / 1e6
    table["total_usd_mm"] = table.sum(axis=1)
    table["pct_of_dc_total"] = 100 * table["total_usd_mm"] / table["total_usd_mm"].sum()
    table["borrowers"] = q.groupby("sector")["entity"].nunique()
    table = table.sort_values("total_usd_mm", ascending=False)
    table.to_csv(out_dir / "dc_sector_mix.csv", float_format="%.2f")

    # pie slices: named sectors largest first, unresolved and tiny sectors last in gray
    share = table["total_usd_mm"].copy()
    small = (table["pct_of_dc_total"] < MIN_SHARE) | share.index.str.startswith("Other")
    slices = share[~small]
    slices[OTHER] = share[small].sum()
    colors = BDC_COLORS[:len(slices) - 1] + [OTHER_COLOR]
    total = share.sum()

    fig, ax = plt.subplots(figsize=(11, 6.5))
    wedges, _ = ax.pie(slices, colors=colors, startangle=90, counterclock=False,
                       wedgeprops=dict(edgecolor=SURFACE, linewidth=2))
    for w, (name, v) in zip(wedges, slices.items()):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        x, y = np.cos(ang), np.sin(ang)
        n = table.loc[name, "borrowers"] if name in table.index else table.loc[small, "borrowers"].sum()
        ax.annotate(f"{name}\n${v:,.0f}M · {100 * v / total:.1f}% · {n} borrowers",
                    xy=(x * 1.0, y * 1.0), xytext=(x * 1.28, y * 1.22),
                    ha="left" if x >= 0 else "right", va="center", fontsize=11, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_aspect("equal")

    _title(fig, f"Technology is the largest sector among data-center borrowers, {quarter}",
           f"Share of BDC fair value, core + support + grey (${total / 1e3:,.1f}B). "
           f"\"Unresolved / other\" includes sectors under {MIN_SHARE:.0f}%.")
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.86, bottom=0.04)
    fig.savefig(out_dir / "dc_sector_mix.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(table.round(1).to_string())


if __name__ == "__main__":
    main()
