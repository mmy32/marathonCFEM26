"""Shared loader and style for the data-center charts: load() returns the tagged rows, each BDC's total
fair value per quarter and BDC names."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py

import pandas as pd

from paths import DATA_CENTER_DIR, FINAL_DIR, UNIVERSE_DIR

FV = "InvestmentOwnedAtFairValue_normalized"
COST = "InvestmentOwnedAtCost_normalized"

# tier label from the classifier -> short function name, in fixed display / color order
FUNCTIONS = {
    "1 Operator / developer": "Operator / developer",
    "2 AI compute / GPU cloud / miners": "AI compute",
    "4 DC services / supply chain": "DC services / supply chain",
}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


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


def _style(ax, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_axisbelow(True)
