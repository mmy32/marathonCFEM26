"""Restructurings and exits, read from how each BDC's schedule changes from one quarter to the
next (the sponsor's definitions of 2026-11-16). Non-accrual stays the model's one outcome; these
are counted beside it.

Restructuring: a change in terms on the same loan held by the same BDC (a tranche chain,
tranches.link_tranche_chains) between consecutive filings:
  pik         cash interest switches to PIK
  spread      the spread steps up, and the all-in rate rises with it
  maturity    the maturity is pushed out
  write_down  cost falls while the loan stays in the schedule
  equity      the BDC's debt to the borrower is replaced by an equity row

Exit: a loan whose chain ends while the data go on, put in exactly one of four buckets, tested in
this order: impaired (non-accrual, marked well below par, or swapped for equity), refinanced
(a new tranche at the same BDC next quarter, or the borrower still held by another BDC), repaid
(last mark at or near par), unresolved (the rest). No exit is dropped.

The cutoffs were fixed on 2026-11-16 before any count was run, and every count is shown at the
second cutoff too. Rules and counts: docs/14_early_warning_model.md, section 2."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

CUTOFFS = {
    "primary": {"near_par": 0.95, "well_below_par": 0.80, "spread_step": 0.0050, "extension_days": 90,
                "write_down": 0.05, "pik": "toggle"},
    "second": {"near_par": 0.98, "well_below_par": 0.90, "spread_step": 0.0100, "extension_days": 365,
               "write_down": 0.10, "pik": "cash to PIK"},
}
# Data filters, not cutoffs (the same at both):
RATE_SLACK = 0.0075   # a step-up counts only if the all-in rate rose by the step less 75 bp (base-rate moves
                      # stayed within that in 2023-2026); a relabel from prime to SOFR raises the reported
                      # spread by about three points while the all-in rate falls
PAR_KEPT = 0.98       # write-down: par stays (a repayment would cut par along with cost)
MIN_PIK_RATE = 0.0025  # PIK under 25 bp is rounding (as in pik_metrics)
MAX_MARK = 1.5        # a mark above 150% of par is par left in local currency (a cleaning issue): no mark

EVENT_TYPES = ["pik", "spread", "maturity", "equity", "write_down"]
EXIT_CLASSES = ["repaid", "refinanced", "impaired", "unresolved"]
DEBT = {"SENIOR_SECURED_1L", "UNITRANCHE", "SENIOR_SECURED_2L", "SUBORDINATED_MEZZ", "SENIOR_UNSECURED",
        "SENIOR_SECURED_UNSPEC"}      # operating debt; CLO_STRUCTURED_EQUITY and OTHER_UNKNOWN are not
EQUITY_WORDS = (r"(?i)\b(?:preferred|common|units?|stock|shares|warrants?|equity securities|equity interests?|"
                r"membership interests?|partnership interests?)\b")


def named_rows(r: pd.DataFrame) -> pd.DataFrame:
    """FINAL_DATASET rows that name a borrower, with the panels' keys added: borrower (trimmed,
    upper case) and qsort (year * 4 + quarter)."""
    r = r[r["borrower_canonical"].notna()]
    r = r.assign(borrower=r["borrower_canonical"].astype(str).str.strip().str.upper(),
                 qsort=r["cal_q"].str[:4].astype(int) * 4 + r["cal_q"].str[-1].astype(int))
    return r[r["borrower"] != ""]


def loans(t: pd.DataFrame) -> pd.DataFrame:
    """Funded loans of operating borrowers (named, not CLO tranches) from the tranche panel, each
    row joined to the same chain's previous filing (suffix _p; missing at a chain's start). mark =
    FV / par where par is what was lent; FV / cost where the reported par is a commitment rather
    than what was lent (a partly drawn revolver or delayed-draw loan: cost under 80% of par)."""
    t = t[t["lien_group"] != "CLO"].sort_values(["chain_id", "qsort"]).copy()
    mark = t["price"].where(t["price"].notna(), (t["FV"] / t["COST"]).where(t["COST"] > 0))
    t["mark"] = mark.where(mark <= MAX_MARK)
    g = t.groupby("chain_id", sort=False)
    for c in ["spread", "maturity", "COST", "PAR", "FV", "IR_Final", "PIK_Final"]:
        t[c + "_p"] = g[c].shift()
    t["linked"] = t["link_rule"] > 0
    return t.reset_index(drop=True)


def equity_holdings(rows: pl.DataFrame) -> pd.DataFrame:
    """(borrower, qsort, cik) where the BDC holds an equity row of the borrower: no par, a positive
    value, not a debt seniority, and an equity word in its description. A share count alone is not
    enough: one BDC tagged its loans' principal as shares in 2026Q1."""
    keep = ["cik", "cal_q", "borrower_canonical", "instrument_seniority", "PAR", "FV", "COST", "investment_identifier"]
    r = named_rows(rows.select([c for c in keep if c in rows.columns]).to_pandas())
    words = r.get("investment_identifier", pd.Series("", index=r.index)).astype(str).str.contains(EQUITY_WORDS)
    eq = (~(r["PAR"] > 0) & ((r["FV"] > 0) | (r["COST"] > 0))
          & ~r["instrument_seniority"].isin(DEBT | {"CLO_STRUCTURED_EQUITY"}) & words)
    out = r.loc[eq, ["borrower", "qsort", "cik"]]
    return out.assign(cik=pd.to_numeric(out["cik"]).astype("int64")).drop_duplicates()


def term_changes(L: pd.DataFrame, cut: dict) -> pd.DataFrame:
    """Per loan and filing (linked rows of `loans`): pik, spread, maturity and write_down as 0 / 1."""
    pik = (L["is_distress_amendment"].fillna(False).astype(bool) if cut["pik"] == "toggle"
           else (L["PIK_Final"].fillna(0) >= MIN_PIK_RATE) & (L["PIK_Final_p"].fillna(0) < MIN_PIK_RATE))
    step = L["spread"] - L["spread_p"]
    spread = (step >= cut["spread_step"] - 1e-9) & (L["IR_Final"] - L["IR_Final_p"] >= step - RATE_SLACK)
    maturity = (L["maturity"] - L["maturity_p"]).dt.days >= cut["extension_days"]
    w = cut["write_down"] * L["PAR_p"]
    write_down = ((L["COST_p"] - L["COST"] >= w) & (L["PAR"] >= PAR_KEPT * L["PAR_p"])
                  & (L["COST_p"] - L["FV_p"] >= w))        # already marked that far below cost: a realized loss
    ev = pd.DataFrame({"pik": pik, "spread": spread, "maturity": maturity, "write_down": write_down}).fillna(False)
    ev = ev.astype(int).mul(L["linked"].astype(int), axis=0)
    return pd.concat([L[["chain_id", "cik", "borrower", "qsort", "linked"]], ev], axis=1)


def loan_exits(L: pd.DataFrame, d: pd.DataFrame, ls: pd.DataFrame, eq: pd.DataFrame, cut: dict) -> pd.DataFrame:
    """Every loan whose chain ends at quarter q while the data go on (exit_q = q + 1), with its
    bucket and the reason. d: debt rows (borrower, qsort, cik), every BDC's, funded or not; ls: the
    BDC's non-accrual status of the borrower (model_panel.lender_status); eq: equity_holdings.
    A loan whose BDC did not file the next quarter cannot be called repaid (its absence says
    nothing); it is impaired, refinanced by another BDC, or unresolved."""
    d, ls, eq = (f.assign(cik=pd.to_numeric(f["cik"]).astype("int64")) for f in (d, ls, eq))
    last_q = L["qsort"].max()
    x = L[L["qsort"] == L.groupby("chain_id")["qsort"].transform("max")]
    x = x[x["qsort"] < last_q].assign(exit_q=lambda z: z["qsort"] + 1)
    key = ["borrower", "exit_q", "cik"]

    def has(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
        f = frame[cols].drop_duplicates().assign(_hit=True)
        return x[cols].merge(f, on=cols, how="left")["_hit"].fillna(False).astype(bool).to_numpy()

    filed = has(d.rename(columns={"qsort": "exit_q"})[["cik", "exit_q"]], ["cik", "exit_q"])
    na_next = has(ls[ls["ls"] == 1].rename(columns={"qsort": "exit_q"}), key)
    # equity: a new equity row at this BDC next quarter, at a BDC that reported equity the quarter before
    e_prev = eq.assign(qsort=eq["qsort"] + 1)
    new_eq = eq.merge(e_prev.assign(_old=1), on=["borrower", "qsort", "cik"], how="left")
    new_eq = new_eq[new_eq["_old"].isna()]
    reports = eq[["cik", "qsort"]].drop_duplicates().assign(qsort=lambda z: z["qsort"] + 1)
    new_eq = new_eq.merge(reports, on=["cik", "qsort"]).rename(columns={"qsort": "exit_q"})
    equity = has(new_eq, key)
    starts = L.loc[~L["linked"], ["borrower", "qsort", "cik"]].rename(columns={"qsort": "exit_q"})
    new_tranche = has(starts, key)

    def others(frame: pd.DataFrame) -> np.ndarray:
        """Number of other BDCs in `frame` at (borrower, exit_q)."""
        n = frame.groupby(["borrower", "exit_q"])["cik"].nunique().rename("n").reset_index()
        return x[["borrower", "exit_q"]].merge(n, how="left")["n"].fillna(0).to_numpy() - has(frame, key)

    other_bdc = others(d.rename(columns={"qsort": "exit_q"})) > 0
    other_new = others(starts) > 0                     # another BDC started a loan to the borrower that quarter

    non_accrual = (x["non_accrual"] == 1).to_numpy() | na_next
    low = (x["mark"] <= cut["well_below_par"]).to_numpy()
    near = (x["mark"] >= cut["near_par"]).to_numpy() & filed
    reason = np.select([non_accrual, low, equity, new_tranche & filed, other_bdc & other_new, other_bdc, near],
                       ["non-accrual", "marked well below par", "replaced by equity", "new tranche, same BDC",
                        "another BDC, new loan that quarter", "another BDC, already lending", "last mark near par"],
                       np.where(filed, "neither near nor well below par, or no mark", "BDC did not file next quarter"))
    cls = np.select([non_accrual | low | equity, (new_tranche & filed) | other_bdc, near],
                    ["impaired", "refinanced", "repaid"], "unresolved")
    out = x[["chain_id", "cik", "borrower", "qsort", "exit_q", "mark"]].assign(exit_class=cls, exit_reason=reason,
                                                                             equity=equity.astype(int))
    return out.reset_index(drop=True)


def borrower_exit_class(exits: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """The bucket of a borrower that left every BDC's books, from its loans that left in its exit
    quarter: impaired if any was, repaid if all were, unresolved otherwise or if none of its loans
    can be traced. Never refinanced: no BDC holds the borrower any more."""
    r = b[["borrower", "exit_q"]].reset_index().merge(exits[["borrower", "exit_q", "exit_class"]],
                                                      on=["borrower", "exit_q"])
    g = r.groupby("index")["exit_class"]
    any_imp = g.apply(lambda s: (s == "impaired").any()).reindex(b.index, fill_value=False).to_numpy()
    all_rep = g.apply(lambda s: (s == "repaid").all()).reindex(b.index, fill_value=False).to_numpy()
    return np.select([any_imp, all_rep], ["impaired", "repaid"], "unresolved")


def borrower_events(changes: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:
    """Per borrower x quarter: 1 if any BDC's loan to it had the event in that quarter (equity:
    a loan replaced by equity at that quarter), and restructured = any of the five."""
    c = changes.groupby(["borrower", "qsort"])[["pik", "spread", "maturity", "write_down"]].max()
    e = exits[exits["equity"] == 1].groupby(["borrower", "exit_q"])["equity"].max().rename_axis(["borrower", "qsort"])
    out = c.join(e, how="outer").fillna(0).astype(int)
    out["restructured"] = out[EVENT_TYPES].max(axis=1)
    return out.reset_index()


def restructuring_by_quarter(changes: dict, exits: dict, last_q: int) -> pd.DataFrame:
    """Loans with each event per quarter at the primary cutoff, a total row, and the total at the
    second cutoff. A loan with two events in a quarter counts once in each type and once in 'any'."""
    def table(ch, ex):
        c = ch[(ch["qsort"] <= last_q) & ch["linked"]].drop(columns="linked")
        e = ex[(ex["equity"] == 1) & (ex["exit_q"] <= last_q)][["chain_id", "exit_q"]].rename(columns={"exit_q": "qsort"})
        ev = pd.concat([c, e.assign(equity=1)], ignore_index=True).fillna(0)
        ev["any"] = ev[EVENT_TYPES].max(axis=1)
        n = c.groupby("qsort").size().rename("loans_observed")     # loans seen in two consecutive filings
        return ev.groupby("qsort")[EVENT_TYPES + ["any"]].sum().astype(int).join(n, how="outer").fillna(0)
    t = table(changes["primary"], exits["primary"])
    t.index = [f"{(q - 1) // 4}Q{(q - 1) % 4 + 1}" for q in t.index]
    t = t[t["loans_observed"] > 0] if len(t) else t
    t.loc["all"] = t.sum()
    t.loc["all, second cutoff"] = table(changes["second"], exits["second"]).sum()
    return t.astype(int).rename_axis("quarter").reset_index()


def exit_counts(exits: dict, last_q: int) -> pd.DataFrame:
    """Every loan exit by bucket and reason, at both cutoffs."""
    parts = []
    for name, ex in exits.items():
        e = ex[ex["exit_q"] <= last_q]
        parts.append(e.groupby(["exit_class", "exit_reason"]).size().rename(f"loans_{name}"))
    t = pd.concat(parts, axis=1).fillna(0).astype(int).reset_index()
    t["order"] = t["exit_class"].map({k: i for i, k in enumerate(EXIT_CLASSES)})
    t = t.sort_values(["order", "loans_primary"], ascending=[True, False]).drop(columns="order")
    sub = t.groupby("exit_class")[["loans_primary", "loans_second"]].sum().reindex(EXIT_CLASSES).fillna(0).astype(int)
    rows = []
    for k in EXIT_CLASSES:
        rows.append({"exit_class": k, "exit_reason": "all", **sub.loc[k].to_dict()})
        rows += t[t["exit_class"] == k].to_dict("records")
    out = pd.DataFrame(rows)
    out = pd.concat([out, pd.DataFrame([{"exit_class": "all", "exit_reason": "all", **sub.sum().to_dict()}])],
                    ignore_index=True)
    for name in exits:
        out[f"share_{name}"] = out[f"loans_{name}"] / sub[f"loans_{name}"].sum()
    return out[["exit_class", "exit_reason", "loans_primary", "share_primary", "loans_second", "share_second"]]
