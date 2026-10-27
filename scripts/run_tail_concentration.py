#!/usr/bin/env python3
"""
run_tail_concentration.py

Is risk concentrated in the tail? Each quarter, rank borrowers by their flow-adjusted change in
value (same return and same $1M prior-FV floor as the stress-event rule in
distress_event_study.py), then ask what share of all markdown dollars the worst 1%, 5% and 10%
cause, against their share of loan value. Also: how concentrated non-accrual dollars are in the
latest quarter.

Usage:
    python scripts/run_tail_concentration.py

Reads data/final_dataset/FINAL_DATASET.csv only. Writes to outputs/risk_monitor/distress/:
    tail_concentration.csv            per quarter and pooled: tail size, borrowers, share of
                                      markdown dollars, share of prior fair value
    tail_concentration_nonaccrual.csv latest quarter: share of non-accrual cost held by the
                                      top 1/5/10% of borrowers
    tail_concentration.png            the slide chart (pooled 2023Q2-2026Q2)
About 5 minutes (builds the loan panel).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "1_data_processing"))
from paths import FINAL_DIR, RISK_MONITOR_DIR, add_code_dirs_to_path  # noqa: E402
add_code_dirs_to_path()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from monitor_panel import build_monitor_panel  # noqa: E402
import distress_event_study as D  # noqa: E402
from charts import SURFACE, INK, MUTED, GRID, BLUE  # noqa: E402

warnings.filterwarnings("ignore")

TAILS = (0.01, 0.05, 0.10)
MIN_FV_PREV = 1e6          # same floor as the stress-event rule
VALUE_GREY = "#b9b7b0"


def tail_shares(b: pd.DataFrame) -> pd.DataFrame:
    """Share of markdown dollars and of prior fair value in the worst p% of borrowers, per quarter and pooled."""
    b = b[(b["fv_prev"] >= MIN_FV_PREV) & b["val_ret"].notna()].copy()
    b["chg"] = b["val_ret"] * b["fv_prev"]           # flow-adjusted change in value, $
    b["loss"] = -b["chg"].clip(upper=0)
    b["pct"] = b.groupby("cal_q")["val_ret"].rank(pct=True, method="first")
    rows = []
    for q, g in list(b.groupby("cal_q")) + [("pooled", b)]:
        for p in TAILS:
            t = g[g["pct"] <= p]
            rows.append({"cal_q": q, "tail": p, "borrowers": len(t), "all_borrowers": len(g),
                         "loss_share": t["loss"].sum() / g["loss"].sum(),
                         "fv_share": t["fv_prev"].sum() / g["fv_prev"].sum(),
                         "tail_loss_bn": t["loss"].sum() / 1e9, "all_loss_bn": g["loss"].sum() / 1e9})
    return pd.DataFrame(rows)


def nonaccrual_shares(b: pd.DataFrame) -> pd.DataFrame:
    last = b["cal_q"].max()
    L = b[(b["cal_q"] == last) & (b["cost"] > 0)].copy()
    L["na_cost"] = (L["na_cost_share"] * L["cost"]).fillna(0)
    L = L.sort_values("na_cost", ascending=False)
    rows = []
    for p in TAILS:
        k = int(round(len(L) * p))
        top = L.head(k)
        rows.append({"cal_q": last, "tail": p, "borrowers": k, "all_borrowers": len(L),
                     "borrowers_with_any_na": int((L["na_cost"] > 0).sum()),
                     "na_share": top["na_cost"].sum() / L["na_cost"].sum(),
                     "cost_share": top["cost"].sum() / L["cost"].sum()})
    return pd.DataFrame(rows)


def chart(t: pd.DataFrame, na: pd.DataFrame, path: Path) -> None:
    pooled = t[t["cal_q"] == "pooled"].set_index("tail")
    q1 = t[t["cal_q"] == "2026Q1"].set_index("tail")
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(TAILS))
    h = 0.36
    for i, p in enumerate(TAILS):
        fv, loss = pooled.loc[p, "fv_share"] * 100, pooled.loc[p, "loss_share"] * 100
        ax.barh(i - h / 2 - 0.01, fv, height=h, color=VALUE_GREY, label="% of loan value" if i == 0 else None)
        ax.barh(i + h / 2 + 0.01, loss, height=h, color=BLUE, label="% of all markdown dollars" if i == 0 else None)
        ax.text(fv + 0.8, i - h / 2 - 0.01, f"{fv:.1f}%", va="center", fontsize=9, color=MUTED)
        ax.text(loss + 0.8, i + h / 2 + 0.01, f"{loss:.0f}%", va="center", fontsize=10, color=INK, weight="bold")
    ax.set_yticks(list(y), [f"Worst {int(p * 100)}%\nof borrowers" for p in TAILS], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    w5 = pooled.loc[0.05]
    fig.text(0.02, 0.965, f"The worst 5% of borrowers hold {w5['fv_share'] * 100:.0f}% of loan value "
             f"but cause {w5['loss_share'] * 100:.0f}% of all markdowns",
             ha="left", va="top", fontsize=12, color=INK, weight="bold")
    fig.text(0.02, 0.905, "Borrowers ranked each quarter by change in value; 2023Q2–2026Q2 pooled",
             ha="left", va="top", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2,
              labelcolor=INK, handlelength=1.2)
    n1 = na[na["tail"] == 0.01].iloc[0]
    # The non-accrual point is a finding in its own right, so it gets its own, larger line.
    fig.text(0.02, 0.135, f"Non-accruals are more concentrated still: 1% of borrowers ({n1['borrowers']}) hold "
             f"{n1['na_share'] * 100:.0f}% of non-accrual dollars ({n1['cal_q']})", fontsize=10.5, color=INK,
             ha="left", va="bottom")
    note = (f"Exception: in 2026Q1, when spreads widened across the market, the worst 10% caused "
            f"{q1.loc[0.10, 'loss_share'] * 100:.0f}% of markdowns (vs {pooled.loc[0.10, 'loss_share'] * 100:.0f}% pooled).\n"
            "Change in value = change in fair value net of new lending and repayments; borrowers with prior value ≥ $1M. "
            "Source: FINAL_DATASET.")
    fig.text(0.02, 0.015, note, fontsize=7.5, color=MUTED, ha="left", va="bottom", linespacing=1.5)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.78, bottom=0.28)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    out = RISK_MONITOR_DIR / "distress"
    out.mkdir(parents=True, exist_ok=True)
    print("Building the loan panel and the borrower-quarter panel ...")
    panel, rows = build_monitor_panel(FINAL_DIR / "FINAL_DATASET.csv")
    b = D.borrower_quarter_panel(panel, rows)
    t = tail_shares(b)
    na = nonaccrual_shares(b)
    t.to_csv(out / "tail_concentration.csv", index=False)
    na.to_csv(out / "tail_concentration_nonaccrual.csv", index=False)
    chart(t, na, out / "tail_concentration.png")
    print(t[t["cal_q"] == "pooled"].round(3).to_string(index=False))
    print(na.round(3).to_string(index=False))
    print(f"  -> {out / 'tail_concentration.png'}")


if __name__ == "__main__":
    main()
