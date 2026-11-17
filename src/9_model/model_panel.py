"""Borrower-quarter panel for the early-warning scorecard. One row per borrower and quarter t,
every BDC's loans to the borrower pooled. The warning signs use only what is filed for quarter t;
the one outcome is a credit event within the next two quarters (t + 1 or t + 2): some BDC flags the
borrower non-accrual, or a lender makes a concession on one of its loans (a switch to PIK, a
debt-for-equity swap, a cost write-down; credit_events.CONCESSIONS). It is read from the filings'
non-accrual flags and loan terms and never from a mark. A borrower that leaves every BDC's books
before the window closes without an event has no label; its exit is put in one of four buckets
(repaid, refinanced, impaired, unresolved; credit_events.py), which never enter the outcome. The
other two term changes (spread step-up, maturity extension) are counted beside it. Built from the
tranche-quarter panel (monitor_panel.build_monitor_panel) and the FINAL_DATASET rows. Definitions
and sample rules: docs/14_early_warning_model.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

import credit_events as CE

MIN_PRIOR_FV = 1_000_000    # sample: the borrower's debt fair value at t is at least $1m
MIN_PRICED_SHARE = 0.5      # sample: priceable par is at least half of the borrower's funded par at t
MAX_CLO_SHARE = 0.5         # sample: CLO / structured tranches under half of par (CLO vehicles are not operating borrowers)
HORIZON = 2                 # the outcome window: a credit event at t + 1 or t + 2
WINDOW_TEXT = "within the next two quarters"
DD_WINDOW = 4               # rolling max drawdown over the preceding four quarters
LOW_MARK = 0.90             # PIK-switch hypotheses: mark below 90% of par ...
MARKDOWN = 0.05             # ... or marked down at least five points in the quarter
JUNIOR_LIENS = {"2L", "SUB", "UNSEC"}


def q_label(qsort) -> list[str]:
    """qsort (year * 4 + quarter) back to 'YYYYQn'."""
    q = np.asarray(qsort, dtype=int)
    return [f"{y}Q{n}" for y, n in zip((q - 1) // 4, (q - 1) % 4 + 1)]


def q_sort(label: str) -> int:
    return int(label[:4]) * 4 + int(label[-1])


def q_next(label: str, k: int = 1) -> str:
    return q_label([q_sort(label) + k])[0]


def window_label(qsort) -> list[str]:
    """The outcome window of signal quarter t: 't+1-t+HORIZON', e.g. '2026Q1-2026Q2'."""
    q = np.asarray(qsort, dtype=int)
    return [f"{a}-{b}" for a, b in zip(q_label(q + 1), q_label(q + HORIZON))]


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def _wmean(df: pd.DataFrame, col: str, w: str, keys: list[str]) -> pd.Series:
    d = df[df[col].notna() & (df[w] > 0)]
    return (d[col] * d[w]).groupby([d[k] for k in keys]).sum() / d[w].groupby([d[k] for k in keys]).sum()


# ------------------------------------------------------------------ inputs from FINAL_DATASET rows

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


def no_par_debt(rows: pl.DataFrame) -> pd.DataFrame:
    """Per borrower x quarter: fair value (fv_nopar) and cost (cost_nopar) of funded debt reported
    without par, which the mark takes at FV / cost (the tranche panel holds only rows with par).
    Rows of unknown seniority without par are mostly equity and warrants, so they are left out."""
    r = rows.select(["cal_q", "borrower_canonical", "instrument_seniority", "PAR", "FV", "COST"]).to_pandas()
    r = CE.named_rows(r[~(r["PAR"] > 0) & (r["FV"] > 0) & (r["COST"] > 0) & r["instrument_seniority"].isin(CE.DEBT)])
    return r.groupby(["borrower", "qsort"]).agg(fv_nopar=("FV", "sum"), cost_nopar=("COST", "sum")).reset_index()


# ------------------------------------------------------------------ predictors from the tranche panel

def _named(tr: pd.DataFrame) -> pd.DataFrame:
    t = tr[~tr["borrower_key"].str.startswith("ID: ")].copy()
    t["borrower"] = _norm(t["borrower_key"])
    return t


def borrower_marks(t: pd.DataFrame) -> pd.DataFrame:
    """Per borrower x quarter over funded loans: par, fv_px and par_px (FV and par of the priceable
    loans; build_model_panel turns them into the mark), junior_share (2L, subordinated and
    unsecured share of priced par), CLO share, PIK switch, dispersion and crowding (n_families:
    manager families lending to it)."""
    keys = ["borrower", "qsort"]
    t = t.assign(_junior=t["PAR_px"].where(t["lien_group"].isin(JUNIOR_LIENS), 0.0),
                 _clo=t["PAR"].where(t["lien_group"] == "CLO", 0.0),
                 _seasoned=t["link_rule"] > 0,
                 _switch=t["is_distress_amendment"].fillna(False).astype(bool))
    b = t.groupby(keys).agg(par=("PAR", "sum"), fv_px=("FV_px", "sum"), par_px=("PAR_px", "sum"),
                            junior_par=("_junior", "sum"), clo_par=("_clo", "sum"),
                            n_families=("family", "nunique"), seasoned=("_seasoned", "max"),
                            pik_switch=("_switch", "max"))
    b["junior_share"] = (b.pop("junior_par") / b["par_px"]).where(b["par_px"] > 0)
    b["clo_share"] = b.pop("clo_par") / b["par"]
    b["log_par"] = np.log(b["par"])
    b["pik_switch"] = b["pik_switch"].astype(float).where(b["seasoned"])   # undefined on a borrower's first quarter
    # valuation gap: highest minus lowest family mark on the same loan (comparable tranches, 2+ families)
    b["dispersion"] = _wmean(t, "V1_range", "PAR", keys)
    b["dispersion_std"] = _wmean(t, "V1_std", "PAR", keys)
    b["has_dispersion"] = b["dispersion"].notna().astype(int)
    b["sector"] = t.sort_values("PAR", ascending=False).drop_duplicates(keys).set_index(keys)["sector"]
    return b.reset_index()


def chain_changes(t: pd.DataFrame) -> pd.DataFrame:
    """Per borrower x quarter t, over the same loans (chains) held at t - 1 and t:
    d_price = par-weighted change in FV / PAR (weights: priced par at t - 1), d_pik_pp =
    par-weighted change in the PIK share of the coupon (percentage points). Draws, repayments and
    loans entering or leaving do not move them."""
    keys = ["borrower", "qsort"]
    c = t[["chain_id", "qsort", "borrower", "FV_px", "PAR_px", "PAR", "pik_share"]].copy()
    c["cprice"] = (c["FV_px"] / c["PAR_px"]).where(c["PAR_px"] > 0)
    c["cprice"] = c["cprice"].where(c["cprice"] <= CE.MAX_MARK)
    prev = c[["chain_id", "qsort", "cprice", "PAR_px", "PAR", "pik_share"]].assign(qsort=c["qsort"] + 1)
    m = c.merge(prev, on=["chain_id", "qsort"], suffixes=("", "_prev"))     # consecutive quarters only
    px = m[m["cprice"].notna() & m["cprice_prev"].notna()]
    w = px["PAR_px_prev"]
    out = (w * (px["cprice"] - px["cprice_prev"])).groupby([px[k] for k in keys]).sum() \
        / w.groupby([px[k] for k in keys]).sum()
    out = out.rename("d_price").to_frame()
    pk = m[m["pik_share"].notna() & m["pik_share_prev"].notna() & (m["PAR_prev"] > 0)]
    wk = pk["PAR_prev"]
    out = out.join((100 * (wk * (pk["pik_share"] - pk["pik_share_prev"])).groupby([pk[k] for k in keys]).sum()
                    / wk.groupby([pk[k] for k in keys]).sum()).rename("d_pik_pp"), how="outer")
    return out.reset_index()


def rolling_max_drawdown(d: pd.DataFrame, col: str = "d_price", window: int = DD_WINDOW) -> pd.DataFrame:
    """Largest peak-to-trough fall of a borrower's chain-linked mark over quarters t - window .. t,
    in price points (0.05 = five points of par), 0 if it never fell. The index cumulates the
    same-loan changes `col`; a quarter without one adds nothing. The drawdown is missing when the
    window holds no observed change."""
    w = d.pivot(index="borrower", columns="qsort", values=col)
    w = w.reindex(columns=range(w.columns.min(), w.columns.max() + 1))
    seen = w.notna().to_numpy()
    level = w.fillna(0.0).cumsum(axis=1).to_numpy()
    dd = np.full(level.shape, np.nan)
    n = np.zeros(level.shape)
    for j in range(level.shape[1]):
        lo = max(0, j - window)
        win = level[:, lo:j + 1]
        dd[:, j] = (np.maximum.accumulate(win, axis=1) - win).max(axis=1)
        n[:, j] = seen[:, lo + 1:j + 1].sum(axis=1) if j > lo else 0
    dd[n == 0] = np.nan
    return pd.DataFrame({"borrower": np.repeat(w.index.to_numpy(), w.shape[1]),
                         "qsort": np.tile(w.columns.to_numpy(), w.shape[0]), "dd": dd.ravel()})


# ------------------------------------------------------------------ the panel

def build_credit_events(t: pd.DataFrame, rows: pl.DataFrame, d: pd.DataFrame, ls: pd.DataFrame) -> dict:
    """Loan-level restructurings and exits at both cutoffs: {"changes": {cutoff: frame}, "exits":
    {cutoff: frame}} (credit_events.py). t: the named tranche panel."""
    L = CE.loans(t)
    eq = CE.equity_holdings(rows)
    return {"changes": {k: CE.term_changes(L, c) for k, c in CE.CUTOFFS.items()},
            "exits": {k: CE.loan_exits(L, d, ls, eq, c) for k, c in CE.CUTOFFS.items()}}


def build_model_panel(tr: pd.DataFrame, rows: pl.DataFrame) -> tuple[pd.DataFrame, dict]:
    """tr: monitor_panel.build_monitor_panel's tranche-quarter panel; rows: its FINAL_DATASET rows.
    Returns (the borrower-quarter panel, build_credit_events' loan-level frames).

    Signs at t: price (mark level), d_price (its change over the quarter, same loans),
    max_drawdown, pik_switch (alternative d_pik_pp), dispersion (the valuation gap: highest minus
    lowest lender family's mark; alternative dispersion_std; has_dispersion = 1 when defined), crowding, plus market-adjusted versions (price_rel,
    d_price_adj) and controls.
    Outcome: y_event = 1 if an accruing borrower has a credit event at t + 1 or t + 2 (any BDC
    flags it non-accrual, or a lender makes a concession: credit_events.CONCESSIONS), 0 if it is
    held and known accruing in both quarters with no concession, missing otherwise: it left every
    BDC's books before the window closed without an event (exited; exit_class repaid / refinanced
    / impaired / unresolved, exit_class_second at the second cutoff), its status is unknown, or
    the window runs past the data. A mark never enters the outcome. event_q is the first quarter
    of the event; win_pik, win_spread, win_maturity, win_equity, win_write_down say which term
    changes fall in the window. Restructurings at t: restr_pik, restr_spread, restr_maturity,
    restr_equity, restr_write_down, restructured. Robustness outcomes: y_na (non-accrual alone,
    the outcome until 2026-11-16; y_na_first, y_na_same_lender are its variants) and y_all_terms
    (non-accrual or any of the five term changes). Sample flags: in_sample, na_risk_set."""
    t = _named(tr)
    b = borrower_marks(t)
    # the mark: FV / par on loans with par, FV / cost on debt without par
    b = b.merge(no_par_debt(rows), on=["borrower", "qsort"], how="left").fillna({"fv_nopar": 0.0, "cost_nopar": 0.0})
    price = ((b["fv_px"] + b["fv_nopar"]) / (b["par_px"] + b["cost_nopar"])).where(b["par_px"] + b["cost_nopar"] > 0)
    b["price"] = price.where(price <= CE.MAX_MARK)
    b["priced_share"] = (b["par_px"] + b["cost_nopar"]) / (b["par"] + b["cost_nopar"])
    ch = chain_changes(t)
    b = b.merge(ch, on=["borrower", "qsort"], how="left")

    # market-adjusted changes and level: the quarter's median borrower subtracted
    b["d_price_adj"] = b["d_price"] - b.groupby("qsort")["d_price"].transform("median")
    b["price_rel"] = b["price"] - b.groupby("qsort")["price"].transform("median")
    dd = rolling_max_drawdown(b, "d_price").rename(columns={"dd": "max_drawdown"})
    b = b.merge(dd, on=["borrower", "qsort"], how="left")
    b["recent_markdown"] = (b["d_price"] <= -MARKDOWN).astype(float).where(b["d_price"].notna())

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

    b["y_na_first"] = b["y_na"].where(b["na_ever"] == 0)
    same = same_lender_entries(ls).assign(_same=1.0)
    b = b.merge(same, on=["borrower", "qsort"], how="left")
    b["y_na_same_lender"] = b["y_na"].where((b["y_na"] == 0) | (b["_same"] == 1))   # other entries dropped
    b = b.drop(columns="_same")

    b["crowding"] = b["n_families"].astype(float)

    # sample: an operating borrower, priced at t, large enough, and with the same loans priced the
    # quarter before (change in the mark and PIK switch defined)
    b["in_sample"] = ((b["clo_share"] < MAX_CLO_SHARE) & b["price"].notna()
                      & (b["priced_share"] >= MIN_PRICED_SHARE) & (b["fv_debt"] >= MIN_PRIOR_FV)
                      & b["d_price"].notna() & b["pik_switch"].notna())
    b["na_risk_set"] = b["in_sample"] & (b["na_status"] == 0)

    b["cal_q"] = q_label(b["qsort"])
    b["outcome_q"] = window_label(b["qsort"])
    first = ["borrower", "cal_q", "outcome_q", "qsort", "sector", "in_sample", "na_risk_set",
             "y_event", "exited", "exit_class", "exit_class_second", "y_na", "y_all_terms", "y_na_first",
             "y_na_same_lender",
             "price", "d_price", "pik_switch", "dispersion", "has_dispersion", "crowding", "max_drawdown",
             "d_pik_pp", "dispersion_std", "price_rel", "d_price_adj", "junior_share", "recent_markdown"]
    b = b[first + [c for c in b.columns if c not in first]].sort_values(["qsort", "borrower"]).reset_index(drop=True)
    return b, ev


def sample_waterfall(m: pd.DataFrame) -> pd.DataFrame:
    """Borrower-quarters (with the whole outcome window in the data) left after each sample rule,
    and the credit events within the window among those accruing at t. The last step keeps the
    labelled rows: an event in the window, or held and known accruing through it."""
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
        ("labelled: an event within t + 1 .. t + 2, or held and accruing through it", m["y_event"].notna()),
    ]
    keep = pd.Series(True, index=m.index)
    out = []
    for name, cond in steps:
        keep &= cond.fillna(False).astype(bool)
        k = m[keep]
        accruing = k["na_status"].eq(0)
        out.append({"step": name, "borrower_quarters": int(keep.sum()), "borrowers": k["borrower"].nunique(),
                    "credit_events": int((accruing & k["event_in_window"].astype(bool)).sum()),
                    "of_which_non_accrual": int((accruing & k["flag_in_window"].astype(bool)).sum())})
    return pd.DataFrame(out)


def event_counts(m: pd.DataFrame) -> pd.DataFrame:
    """Per predictor quarter, what happened within t + 1 .. t + 2 to every accruing borrower in the
    sample: a credit event, no event (held and accruing through the window), left every BDC's books
    without one, and status unknown (these four add up to at_risk). Events are split into those
    with a non-accrual flag and those with a concession only (they add up to events), and counted
    by type (a borrower can have several). Exits are split into the four buckets (they add up to
    exited). no_event_other_change counts the borrowers without an event that had a spread step-up
    or a maturity extension in the window. labelled = events + no_event; exits and unknown status
    have no label."""
    r = m[m["na_risk_set"].astype(bool) & m["window_complete"].astype(bool)]
    ex = r["exited"].astype(bool)
    ev, flag = r["y_event"] == 1, r["flag_in_window"].astype(bool)
    cols = {"at_risk": pd.Series(True, index=r.index), "events": ev, "no_event": r["y_event"] == 0, "exited": ex,
            "status_unknown": r["y_event"].isna() & ~ex,
            "event_non_accrual": ev & flag, "event_concession_only": ev & ~flag,
            **{"event_" + k: ev & (r["win_" + k] == 1) for k in CE.CONCESSIONS},
            **{"exit_" + k: ex & (r["exit_class"] == k) for k in CE.EXIT_CLASSES},
            "no_event_other_change": (r["y_event"] == 0) & r["restr_in_window"].astype(bool),
            "labelled": r["y_event"].notna()}
    t = pd.DataFrame(cols).groupby([r["cal_q"], r["outcome_q"]]).sum().reset_index()
    t["event_rate"] = t["events"] / t["labelled"]
    total = t.drop(columns=["cal_q", "outcome_q", "event_rate"]).sum()
    total["event_rate"] = total["events"] / total["labelled"]
    return pd.concat([t, pd.DataFrame([{"cal_q": "all", "outcome_q": "", **total}])], ignore_index=True)


def unlabelled_exits(m: pd.DataFrame) -> pd.DataFrame:
    """The exits in event_counts (borrowers that left every BDC's books without an event, no
    label) by bucket, at both cutoffs. Refinanced is empty by construction: no BDC holds the
    borrower any more."""
    r = m[m["na_risk_set"].astype(bool) & m["window_complete"].astype(bool) & m["exited"].astype(bool)]
    t = pd.DataFrame({"exits_primary": r["exit_class"].value_counts(), "exits_second": r["exit_class_second"].value_counts()})
    t = t.reindex(CE.EXIT_CLASSES).fillna(0).astype(int)
    t.loc["all"] = t.sum()
    for k in ["primary", "second"]:
        t[f"share_{k}"] = t[f"exits_{k}"] / t.loc["all", f"exits_{k}"]
    return t[["exits_primary", "share_primary", "exits_second", "share_second"]].rename_axis("exit_class").reset_index()


def restructuring_vs_nonaccrual(m: pd.DataFrame) -> pd.DataFrame:
    """Accruing borrowers with each term change at t, and how many entered non-accrual within the
    next two quarters, next to those with none: which term changes precede a default. Rows with
    the non-accrual outcome observed (y_na)."""
    d = m[m["na_risk_set"].astype(bool) & m["y_na"].notna()]
    out = []
    for name, mask in [*((k, d["restr_" + k] == 1) for k in CE.EVENT_TYPES), ("any", d["restructured"] == 1),
                       ("none", d["restructured"] == 0)]:
        g = d.loc[mask, "y_na"]
        out.append({"restructuring_at_t": name, "in_outcome": name in CE.CONCESSIONS, "borrower_quarters": len(g),
                    "entered_non_accrual": int(g.sum()), "entry_rate": g.mean() if len(g) else np.nan})
    return pd.DataFrame(out)


def filing_lags(final_csv, metadata_csv, first_q: str = "2023Q1") -> pd.DataFrame:
    """When each quarter's filings became public: days from the calendar quarter-end to the SEC
    filing date of every filing in FINAL_DATASET (dates from the EDGAR index, cache/all_metadata.csv).
    Everything a quarter's scorecard uses (its signs, and the outcomes observed up to it) is public
    on `last_filing`, the earliest date the scorecard can be made. A filing dated before the
    quarter-end (filed_before_quarter_end) is a fiscal quarter that ended a month or two earlier."""
    f = (pl.scan_csv(final_csv, infer_schema_length=0).select(["accession", "cal_q"]).unique().collect().to_pandas())
    u = pd.read_csv(metadata_csv, dtype=str, usecols=["accessionNumber", "filingDate", "form"])
    key = lambda x: x.astype(str).str.replace("-", "", regex=False).str.lstrip("0")   # noqa: E731
    u = u.assign(key=key(u["accessionNumber"]), filed=pd.to_datetime(u["filingDate"].str[:10], errors="coerce"))
    j = f.assign(key=key(f["accession"])).merge(u.drop_duplicates("key")[["key", "filed", "form"]], on="key", how="left")
    j = j[j["cal_q"] >= first_q]
    j["lag"] = (j["filed"] - pd.PeriodIndex(j["cal_q"], freq="Q").end_time.normalize()).dt.days
    g = j.groupby("cal_q")
    t = g.agg(filings=("key", "size"), with_filing_date=("filed", "count"), annual_reports=("form", lambda x: (x == "10-K").sum()),
              median_days=("lag", "median"), max_days=("lag", "max"), first_filing=("filed", "min"),
              last_filing=("filed", "max")).reset_index()
    for name, f_ in [("within_45_days", lambda x: (x["lag"] <= 45).mean()), ("within_90_days", lambda x: (x["lag"] <= 90).mean()),
                     ("filed_before_quarter_end", lambda x: (x["lag"] < 0).sum())]:
        t[name] = g.apply(f_).to_numpy()
    for c in ["first_filing", "last_filing"]:
        t[c] = t[c].dt.date
    return t
