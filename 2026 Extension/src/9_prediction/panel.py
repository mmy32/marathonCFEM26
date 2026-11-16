"""Borrower-quarter panel for the early-warning model (src/9_prediction): one row per borrower and quarter t,
every BDC's loans to the borrower pooled. Joins the signs (features) and the outcomes
(targets), applies the sample rules, and summarises the sample (waterfall, events, exits).
Built from the tranche-quarter panel (monitor_panel.build_monitor_panel) and the FINAL_DATASET rows.
Definitions and sample rules: docs/14_early_warning_model.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from common import q_label, window_label
from config import MAX_CLO_SHARE, MIN_PRICED_SHARE, MIN_PRIOR_FV
from features import build_features
from targets import add_targets


def build_model_panel(tr: pd.DataFrame, rows: pl.DataFrame) -> pd.DataFrame:
    """Signs (build_features), then outcomes (add_targets), then the sample flags in_sample and
    na_risk_set: an operating borrower, priced at t, large enough, with the same loans priced the
    quarter before (change in the mark and PIK switch defined), and, for the risk set, accruing."""
    b = add_targets(build_features(tr, rows), rows)

    # sample: an operating borrower, priced at t, large enough, and with the same loans priced the
    # quarter before (change in the mark and PIK switch defined)
    b["in_sample"] = ((b["clo_share"] < MAX_CLO_SHARE) & b["price"].notna()
                      & (b["priced_share"] >= MIN_PRICED_SHARE) & (b["fv_debt"] >= MIN_PRIOR_FV)
                      & b["d_price"].notna() & b["pik_switch"].notna())
    b["na_risk_set"] = b["in_sample"] & (b["na_status"] == 0)

    b["cal_q"] = q_label(b["qsort"])
    b["outcome_q"] = window_label(b["qsort"])
    first = ["borrower", "cal_q", "outcome_q", "qsort", "sector", "in_sample", "na_risk_set",
             "y_na", "y_dprice", "y_valret", "exited", "exit_type", "y_na_first", "y_na_same_lender",
             "price", "d_price", "pik_switch", "dispersion", "has_dispersion", "crowding", "max_drawdown",
             "d_pik_pp", "dispersion_range", "price_rel", "d_price_adj", "junior_share", "recent_markdown"]
    return b[first + [c for c in b.columns if c not in first]].sort_values(["qsort", "borrower"]).reset_index(drop=True)


def sample_waterfall(m: pd.DataFrame) -> pd.DataFrame:
    """Borrower-quarters (with the whole outcome window in the data) left after each sample rule,
    and the new non-accrual entries within the window among them. The last step keeps the labelled
    rows: flagged in the window, or held and known accruing through it."""
    m = m[m["window_complete"].astype(bool)]
    steps = [
        ("named borrower-quarters, funded debt", pd.Series(True, index=m.index)),
        ("not a CLO / structured vehicle", m["clo_share"] < MAX_CLO_SHARE),
        ("price at t (priceable par, FV/PAR <= 1.5)", m["price"].notna()),
        (f"priced share >= {MIN_PRICED_SHARE:.0%} of funded par", m["priced_share"] >= MIN_PRICED_SHARE),
        (f"debt fair value >= ${MIN_PRIOR_FV / 1e6:.0f}m", m["fv_debt"] >= MIN_PRIOR_FV),
        ("same loans priced the quarter before (change in mark, PIK switch defined)",
         m["d_price"].notna() & m["pik_switch"].notna()),
        ("non-accrual status known at t and accruing (risk set)", m["na_status"] == 0),
        ("labelled: flagged within t + 1 .. t + 2, or held and accruing through it", m["y_na"].notna()),
    ]
    keep = pd.Series(True, index=m.index)
    out = []
    for name, cond in steps:
        keep &= cond.fillna(False).astype(bool)
        k = m[keep]
        out.append({"step": name, "borrower_quarters": int(keep.sum()), "borrowers": k["borrower"].nunique(),
                    "new_nonaccruals": int((k["na_status"].eq(0) & k["flag_in_window"].astype(bool)).sum())})
    return pd.DataFrame(out)


EXIT_TYPES = ["at scheduled maturity", "before scheduled maturity", "maturity not reported"]


def event_counts(m: pd.DataFrame) -> pd.DataFrame:
    """Per predictor quarter, what happened within t + 1 .. t + 2 to every accruing borrower in the
    sample: entered non-accrual, still accruing through the window, left every BDC's books, and
    status unknown (these four add up to at_risk). Exits are split by type (the three add up to
    exited). events = entered non-accrual; labelled = events + still accruing. Exits and unknown
    status have no label."""
    r = m[m["na_risk_set"].astype(bool) & m["window_complete"].astype(bool)]
    ex = r["exited"].astype(bool)
    cols = {"at_risk": pd.Series(True, index=r.index), "entered_non_accrual": r["y_na"] == 1,
            "still_accruing": r["y_na"] == 0, "exited": ex,
            "status_unknown": r["y_na"].isna() & ~ex,
            **{"exit_" + k.replace(" ", "_"): ex & (r["exit_type"] == k) for k in EXIT_TYPES},
            "events": r["y_na"] == 1, "labelled": r["y_na"].notna()}
    t = pd.DataFrame(cols).groupby([r["cal_q"], r["outcome_q"]]).sum().reset_index()
    t["event_rate"] = t["events"] / t["labelled"]
    total = t.drop(columns=["cal_q", "outcome_q", "event_rate"]).sum()
    total["event_rate"] = total["events"] / total["labelled"]
    return pd.concat([t, pd.DataFrame([{"cal_q": "all", "outcome_q": "", **total}])], ignore_index=True)


def exit_table(m: pd.DataFrame) -> pd.DataFrame:
    """Every exit in event_counts by type, with the borrower's mark at t beside it as context
    only: the mark never types an exit and never labels it. A borrower refinanced by another BDC
    is still held, so it is never an exit; the filings do not separate early repayment,
    refinancing outside the BDCs, a sale and a write-off."""
    r = m[m["na_risk_set"].astype(bool) & m["window_complete"].astype(bool) & m["exited"].astype(bool)]
    band = pd.cut(r["price"], [-np.inf, 0.80, 0.95, np.inf], labels=["below 80", "80-95", "95 and up"], right=False)
    t = pd.crosstab(r["exit_type"], band).reindex(EXIT_TYPES).fillna(0).astype(int)
    t.columns = [f"mark at t {c}" for c in t.columns]
    t.insert(0, "exits", t.sum(axis=1))
    t.insert(1, "share", t["exits"] / t["exits"].sum())
    t.loc["all"] = t.sum()
    return t.rename_axis("exit_type").reset_index()
