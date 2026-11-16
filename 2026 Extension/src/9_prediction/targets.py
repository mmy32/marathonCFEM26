"""The outcomes of the early-warning model (src/9_prediction), built on top of the borrower-quarter signs.
Classification: y_na = 1 if an accruing borrower is flagged non-accrual by any BDC at t + 1 or t + 2,
0 if it is held and known accruing in both quarters, missing otherwise: it left every BDC's books
before the window closed without a flag (exited; exit_type from its scheduled maturity), its status
is unknown, or the window runs past the data. A mark never enters this outcome. Robustness:
y_na_first, y_na_same_lender. Markdown (continuous): y_dprice and y_valret, the same-loan change in
the mark and val_ret from t to t + 1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from common import norm_name
from config import HORIZON


def debt_rows(rows: pl.DataFrame) -> pd.DataFrame:
    """Debt rows (PAR > 0) with a borrower name: borrower, qsort, cik, FV, non_accrual (1 / 0 / NaN).
    All of them, funded or not and including rows marked at zero, so a loan written off still
    counts for non-accrual and for the borrower's fair value."""
    r = rows.select(["cik", "cal_q", "borrower_canonical", "PAR", "FV",
                     pl.col("non_accrual").cast(pl.Float64, strict=False)]).to_pandas()
    r = r[(r["PAR"] > 0) & r["borrower_canonical"].notna()]
    r = r.assign(borrower=norm_name(r["borrower_canonical"]),
                 qsort=r["cal_q"].str[:4].astype(int) * 4 + r["cal_q"].str[-1].astype(int))
    return r[r["borrower"] != ""][["borrower", "qsort", "cik", "FV", "non_accrual"]]


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


def add_targets(b: pd.DataFrame, rows: pl.DataFrame) -> pd.DataFrame:
    """Add the outcomes to the borrower-quarter signs `b` (features.build_features): the
    non-accrual status and debt fair value of every debt row, the outcomes over t + 1 .. t +
    HORIZON, the exits and the markdown outcomes."""
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
    y = pd.Series(np.select([b["flag_in_window"], clean], [1.0, 0.0], np.nan), index=b.index)
    b["y_na"] = y.where((b["na_status"] == 0) & b["window_complete"])
    # exits: left every BDC's books before the window closed, with no flag in it
    b["exited"] = b["window_complete"] & ~b["flag_in_window"] & ~held.all(axis=1)
    b["exit_q"] = (b["qsort"] + (~held).argmax(axis=1) + 1).where(b["exited"])
    b["exit_type"] = np.select(
        [~b["exited"], b["maturity_q"].isna(), b["maturity_q"] <= b["exit_q"]],
        ["", "maturity not reported", "at scheduled maturity"], "before scheduled maturity")
    # markdown outcomes: the same-loan change in the mark (d_price) and val_ret over t -> t + 1, read
    # from the borrower's next quarter; missing if it left the books or its loans cannot be linked
    nxt = b[["borrower", "qsort", "d_price", "val_ret"]].assign(qsort=b["qsort"] - 1).rename(
        columns={"d_price": "y_dprice", "val_ret": "y_valret"})
    b = b.merge(nxt, on=["borrower", "qsort"], how="left")
    b["na_status_next"] = b["na_status_1"]
    b["has_next"] = b["held_1"]

    b["y_na_first"] = b["y_na"].where(b["na_ever"] == 0)
    same = same_lender_entries(ls).assign(_same=1.0)
    b = b.merge(same, on=["borrower", "qsort"], how="left")
    b["y_na_same_lender"] = b["y_na"].where((b["y_na"] == 0) | (b["_same"] == 1))   # other entries dropped
    b = b.drop(columns="_same")

    return b
