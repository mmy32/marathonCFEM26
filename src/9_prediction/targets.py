"""The outcomes of the early-warning model (src/9_prediction), built on top of the borrower-quarter signs.
Classification: y_event = 1 if an accruing borrower has a credit event at t + 1 or t + 2 (any BDC
flags it non-accrual, or a lender makes a concession on one of its loans: a switch to PIK, a
debt-for-equity swap, a cost write-down; credit_events.CONCESSIONS), 0 if it is held and known
accruing in both quarters with no concession, missing otherwise: it left every BDC's books before
the window closed without an event (exited; exit_class repaid / refinanced / impaired /
unresolved, credit_events.py), its status is unknown, or the window runs past the data. A mark
never enters this outcome. Robustness: y_na (non-accrual alone, the outcome until 2026-11-16),
y_na_first, y_na_same_lender, y_all_terms."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

import credit_events as CE
from config import HORIZON
from features import named_tranches


def debt_rows(rows: pl.DataFrame) -> pd.DataFrame:
    """Debt rows (PAR > 0) with a borrower name: borrower, qsort, cik, FV, non_accrual (1 / 0 / NaN).
    All of them, funded or not and including rows marked at zero, so a loan written off still
    counts for non-accrual and for the borrower's fair value."""
    r = rows.select(["cik", "cal_q", "borrower_canonical", "PAR", "FV",
                     pl.col("non_accrual").cast(pl.Float64, strict=False)]).to_pandas()
    return CE.named_rows(r[r["PAR"] > 0])[["borrower", "qsort", "cik", "FV", "non_accrual"]]


def lender_status(d: pd.DataFrame) -> pd.DataFrame:
    """One row per borrower x quarter x BDC: ls = 1 if the BDC flags any of its loans to the
    borrower non-accrual, 0 if its status is known and none is flagged, NaN if unknown (the
    filing marks non-accrual only in its HTML)."""
    g = d.groupby(["borrower", "qsort", "cik"])["non_accrual"].agg(["count", "max"]).reset_index()
    g["ls"] = g["max"].where(g["count"] > 0)
    return g[["borrower", "qsort", "cik", "ls"]]


def borrower_status(d: pd.DataFrame, ls: pd.DataFrame) -> pd.DataFrame:
    """Per borrower x quarter: na_status (1 if any lender flags it, 0 if some lender's status is
    known and none flags it, NaN if no lender's status is known), na_ever (flagged at or before
    t), fv_debt (fair value of all its debt rows) and the lenders with known status."""
    s = ls.groupby(["borrower", "qsort"])["ls"].agg(na_status="max", n_lenders_known="count")
    s["fv_debt"] = d.groupby(["borrower", "qsort"])["FV"].sum()
    s = s.reset_index().sort_values(["borrower", "qsort"])
    s["na_ever"] = s["na_status"].fillna(0).groupby(s["borrower"]).cummax()
    return s


def same_lender_entries(ls: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """(borrower, qsort = t) where a BDC that reported the borrower accruing at t flags it
    non-accrual at t + 1 .. t + horizon: the strict version of a new entry."""
    keys = ["borrower", "qsort", "cik"]
    known0 = ls.loc[ls["ls"] == 0, keys]
    flags = pd.concat([ls.loc[ls["ls"] == 1, keys].assign(qsort=lambda x, k=k: x["qsort"] - k)
                       for k in range(1, horizon + 1)]).drop_duplicates()
    return known0.merge(flags, on=keys)[["borrower", "qsort"]].drop_duplicates()


def build_credit_events(t: pd.DataFrame, rows: pl.DataFrame, d: pd.DataFrame, ls: pd.DataFrame) -> dict:
    """Loan-level restructurings and exits at both cutoffs: {"changes": {cutoff: frame}, "exits":
    {cutoff: frame}} (credit_events.py). t: the named tranche panel."""
    L = CE.loans(t)
    eq = CE.equity_holdings(rows)
    return {"changes": {k: CE.term_changes(L, c) for k, c in CE.CUTOFFS.items()},
            "exits": {k: CE.loan_exits(L, d, ls, eq, c) for k, c in CE.CUTOFFS.items()}}


def add_targets(b: pd.DataFrame, rows: pl.DataFrame, tr: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add the outcomes to the borrower-quarter signs `b` (features.build_features): the
    non-accrual status and debt fair value of every debt row, the restructurings at t and in the
    window, the outcomes over t + 1 .. t + HORIZON and the exits. tr: the
    tranche-quarter panel. Returns (the frame, build_credit_events' loan-level frames).

    y_event is the outcome; event_q is the first quarter of the event; win_pik, win_spread,
    win_maturity, win_equity, win_write_down say which term changes fall in the window.
    Restructurings at t: restr_pik, restr_spread, restr_maturity, restr_equity, restr_write_down,
    restructured. Robustness outcomes: y_na (non-accrual alone; y_na_first, y_na_same_lender are
    its variants) and y_all_terms (non-accrual or any of the five term changes)."""
    t = named_tranches(tr)
    # non-accrual status and debt fair value from every debt row
    d = debt_rows(rows)
    ls = lender_status(d)
    s = borrower_status(d, ls)
    b = b.merge(s, on=["borrower", "qsort"], how="left")

    # outcomes over t + 1 .. t + HORIZON
    for k in range(1, HORIZON + 1):
        nk = s.assign(qsort=s["qsort"] - k)[["borrower", "qsort", "na_status"]].rename(
            columns={"na_status": f"na_status_{k}"})
        nk[f"held_{k}"] = True
        b = b.merge(nk, on=["borrower", "qsort"], how="left")
        b[f"held_{k}"] = b[f"held_{k}"].fillna(False).astype(bool)
    st = b[[f"na_status_{k}" for k in range(1, HORIZON + 1)]]
    held = b[[f"held_{k}" for k in range(1, HORIZON + 1)]].to_numpy()
    b["window_complete"] = b["qsort"] + HORIZON <= max(b["qsort"].max(), s["qsort"].max())
    b["flag_in_window"] = (st == 1).any(axis=1)
    b["entry_q"] = (b["qsort"] + (st == 1).to_numpy().argmax(axis=1) + 1).where(b["flag_in_window"])
    clean = (st == 0).all(axis=1)
    at_risk = (b["na_status"] == 0) & b["window_complete"]

    def label(event: pd.Series) -> pd.Series:
        """1 if the event falls in the window, 0 if held and known accruing through it."""
        return pd.Series(np.select([event, clean], [1.0, 0.0], np.nan), index=b.index).where(at_risk)

    # restructurings (term changes on the same loan) at t and in the window t + 1 .. t + HORIZON
    ev = build_credit_events(t, rows, d, ls)
    r = CE.borrower_events(ev["changes"]["primary"], ev["exits"]["primary"])
    kinds = CE.EVENT_TYPES + ["restructured", "concession"]
    b = b.merge(r.rename(columns={c: "restr_" + c for c in CE.EVENT_TYPES}).drop(columns="concession"),
                on=["borrower", "qsort"], how="left")
    rcols = ["restr_" + c for c in CE.EVENT_TYPES] + ["restructured"]
    b[rcols] = b[rcols].fillna(0).astype(int)
    ahead = [r.assign(qsort=r["qsort"] - k, k=k) for k in range(1, HORIZON + 1)]
    win = pd.concat(ahead).groupby(["borrower", "qsort"])[kinds].max().rename(
        columns={**{c: "win_" + c for c in CE.EVENT_TYPES}, "restructured": "restr_in_window",
                 "concession": "concession_in_window"})
    b = b.merge(win.reset_index(), on=["borrower", "qsort"], how="left")
    b[list(win.columns)] = b[list(win.columns)].fillna(0).astype(int)
    for c in ["restr_in_window", "concession_in_window"]:
        b[c] = b[c].astype(bool)

    # the outcome: non-accrual or a concession in the window; and the quarter it first happens
    b["event_in_window"] = b["flag_in_window"] | b["concession_in_window"]
    b["y_event"] = label(b["event_in_window"])
    hit = []
    for k, a in enumerate(ahead, start=1):
        c = a.loc[a["concession"] == 1, ["borrower", "qsort"]].assign(_c=True)
        hit.append((b[f"na_status_{k}"] == 1).to_numpy()
                   | b[["borrower", "qsort"]].merge(c, how="left")["_c"].fillna(False).to_numpy(dtype=bool))
    hit = np.column_stack(hit)
    b["event_q"] = (b["qsort"] + hit.argmax(axis=1) + 1).where(hit.any(axis=1))
    b["y_na"] = label(b["flag_in_window"])                               # robustness: non-accrual alone
    b["y_all_terms"] = label(b["flag_in_window"] | b["restr_in_window"])   # robustness: any of the five

    # exits: left every BDC's books before the window closed, with no event in it
    b["exited"] = b["window_complete"] & ~b["event_in_window"] & ~held.all(axis=1)
    b["exit_q"] = (b["qsort"] + (~held).argmax(axis=1) + 1).where(b["exited"])
    ex = b[b["exited"]]
    for k, col in [("primary", "exit_class"), ("second", "exit_class_second")]:
        b[col] = ""
        b.loc[ex.index, col] = CE.borrower_exit_class(ev["exits"][k], ex)

    b["na_status_next"] = b["na_status_1"]
    b["has_next"] = b["held_1"]

    b["y_na_first"] = b["y_na"].where(b["na_ever"] == 0)
    same = same_lender_entries(ls).assign(_same=1.0)
    b = b.merge(same, on=["borrower", "qsort"], how="left")
    b["y_na_same_lender"] = b["y_na"].where((b["y_na"] == 0) | (b["_same"] == 1))   # other entries dropped
    b = b.drop(columns="_same")

    return b, ev
