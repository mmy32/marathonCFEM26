"""The cleaned dataset in a few numbers and its sector mix (slide 7). Reads FINAL_DATASET.csv; writes
dataset_overview.csv, dataset_overview_sectors.csv and dataset_overview.png to outputs/eda_plots/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paths import EDA_PLOTS_DIR, FINAL_DIR
from manager_family import add_family

FV = "InvestmentOwnedAtFairValue_normalized"
PAR = "InvestmentOwnedBalancePrincipalAmount_normalized"
COLS = ["cik", "accession", "cal_q", "borrower_canonical", FV, PAR, "instrument_seniority", "sector",
        "bdc_type", "non_accrual"]
UNKNOWN = {"Other / Unknown / Unresolved", "Other / Unknown", "Other / Unclassified"}
BLUE, GREY = "#2a78d6", "#c3bfb5"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load():
    d = pd.read_csv(FINAL_DIR / "FINAL_DATASET.csv", usecols=COLS, low_memory=False)
    return add_family(d)


def overview(d: pd.DataFrame) -> pd.DataFrame:
    last = d["cal_q"].max()
    L = d[d["cal_q"] == last]
    fv = L[FV].clip(lower=0)
    debt = L[PAR].fillna(0) > 0
    rows = [
        ("rows (position-quarters)", len(d)),
        ("rows with a principal amount (loans), all quarters", round((d[PAR].fillna(0) > 0).mean(), 3)),
        ("quarters", d["cal_q"].nunique()),
        ("first quarter", d["cal_q"].min()),
        ("last quarter", last),
        ("filings (10-K / 10-Q)", d["accession"].nunique()),
        ("BDCs, any quarter", d["cik"].nunique()),
        ("manager families, any quarter", d["family"].nunique()),
        ("borrowers, any quarter", d["borrower_canonical"].nunique()),
        (f"BDCs, {last}", L["cik"].nunique()),
        (f"borrowers, {last}", L["borrower_canonical"].nunique()),
        (f"fair value $bn, {last}", round(fv.sum() / 1e9, 1)),
        (f"debt share of fair value, {last}", round(fv[debt].sum() / fv.sum(), 3)),
        (f"first-lien share of fair value, {last}",
         round(fv[L["instrument_seniority"] == "SENIOR_SECURED_1L"].sum() / fv.sum(), 3)),
        ("rows with a borrower", round(d["borrower_canonical"].notna().mean(), 3)),
        ("rows with a sector", round((~d["sector"].isin(UNKNOWN) & d["sector"].notna()).mean(), 3)),
        (f"fair value with known non-accrual status, {last}", round(fv[L["non_accrual"].notna()].sum() / fv.sum(), 3)),
    ]
    for t, g in L.groupby("bdc_type"):
        rows.append((f"{t} BDCs, {last}", g["cik"].nunique()))
        rows.append((f"{t} fair value $bn, {last}", round(g[FV].clip(lower=0).sum() / 1e9, 1)))
    return pd.DataFrame(rows, columns=["item", "value"])


def sectors(d: pd.DataFrame) -> pd.DataFrame:
    last = d["cal_q"].max()
    L = d[d["cal_q"] == last].assign(fv=lambda x: x[FV].clip(lower=0))
    s = L.assign(sector=L["sector"].where(~L["sector"].isin(UNKNOWN) & L["sector"].notna(),
                                                 "Unclassified"))
    t = s.groupby("sector")["fv"].sum().rename("fv").reset_index()
    t["share"] = t["fv"] / t["fv"].sum()
    known = t[t["sector"] != "Unclassified"].sort_values("share", ascending=False)
    return pd.concat([known, t[t["sector"] == "Unclassified"]], ignore_index=True).assign(cal_q=last)


def plot(t: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(9, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(t))
    colors = [GREY if s == "Unclassified" else BLUE for s in t["sector"]]
    ax.barh(list(y), 100 * t["share"], color=colors, height=0.66, edgecolor=SURFACE, linewidth=1.5)
    for i, (v, b) in enumerate(zip(t["share"], t["fv"])):
        ax.text(100 * v + 0.3, i, f"{v:.1%}  (${b / 1e9:,.0f}bn)", va="center", fontsize=11, color=INK)
    ax.set_yticks(list(y), t["sector"], fontsize=12, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100 * t["share"].max() * 1.45)
    ax.xaxis.set_visible(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.01, 0.955, f"What BDCs hold, by sector ({t['cal_q'].iat[0]}, share of fair value)",
             fontsize=14, color=INK, weight="bold", va="top")
    fig.subplots_adjust(left=0.33, right=0.98, top=0.9, bottom=0.03)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    EDA_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    d = load()
    o = overview(d)
    o.to_csv(EDA_PLOTS_DIR / "dataset_overview.csv", index=False)
    s = sectors(d)
    s.to_csv(EDA_PLOTS_DIR / "dataset_overview_sectors.csv", index=False)
    plot(s, EDA_PLOTS_DIR / "dataset_overview.png")
    print(o.to_string(index=False))
    print(s.round(3).to_string(index=False))
