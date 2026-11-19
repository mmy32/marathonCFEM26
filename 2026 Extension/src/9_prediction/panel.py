"""Borrower-quarter panel for the early-warning scorecard (src/9_prediction): one row per borrower and
quarter t, every BDC's loans to the borrower pooled. Joins the signs (features) and the outcomes
(targets), applies the sample rules, and summarises the sample (waterfall, events, exits). The
warning signs use only what is filed for quarter t; the one outcome is a credit event within the
next two quarters (t + 1 or t + 2): some BDC flags the borrower non-accrual, or a lender makes a
concession on one of its loans (a switch to PIK, a debt-for-equity swap, a cost write-down;
credit_events.CONCESSIONS). It is read from the filings' non-accrual flags and loan terms and never
from a mark. A borrower that leaves every BDC's books before the window closes without an event has
no label; its exit is put in one of four buckets (repaid, refinanced, impaired, unresolved;
credit_events.py), which never enter the outcome. The other two term changes (spread step-up,
maturity extension) are counted beside it. Built from the tranche-quarter panel
(monitor_panel.build_monitor_panel) and the FINAL_DATASET rows. Definitions and sample rules:
docs/14_early_warning_model.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

import credit_events as CE
from common import q_label, window_label
from config import MAX_CLO_SHARE, MIN_PRICED_SHARE, MIN_PRIOR_FV
from features import build_features
from targets import add_targets


def build_model_panel(tr: pd.DataFrame, rows: pl.DataFrame) -> tuple[pd.DataFrame, dict]:
    """tr: monitor_panel.build_monitor_panel's tranche-quarter panel; rows: its FINAL_DATASET rows.
    Returns (the borrower-quarter panel, the loan-level restructurings and exits at both cutoffs).
    Signs (features.build_features), then outcomes (targets.add_targets: y_event and its
    robustness variants, restructurings, exits), then the sample flags in_sample and na_risk_set:
    an operating borrower, priced at t, large enough, with the same loans priced the quarter before
    (change in the mark and PIK switch defined), and, for the risk set, accruing."""
    b, ev = add_targets(build_features(tr, rows), rows, tr)

    # sample: an operating borrower, priced at t, large enough, and with the same loans priced the
    # quarter before (change in the mark and PIK switch defined)
    b["in_sample"] = ((b["clo_share"] < MAX_CLO_SHARE) & b["price"].notna()
                      & (b["priced_share"] >= MIN_PRICED_SHARE) & (b["fv_debt"] >= MIN_PRIOR_FV)
                      & b["d_price"].notna() & b["pik_switch"].notna())
    b["na_risk_set"] = b["in_sample"] & (b["na_status"] == 0)

    b["cal_q"] = q_label(b["qsort"])
    b["outcome_q"] = window_label(b["qsort"])
    first = ["borrower", "cal_q", "outcome_q", "qsort", "sector", "in_sample", "na_risk_set",
             "y_event", "exited", "exit_class", "exit_class_second", "y_na", "y_all_terms", "y_dprice", "y_valret",
             "y_na_first", "y_na_same_lender",
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
