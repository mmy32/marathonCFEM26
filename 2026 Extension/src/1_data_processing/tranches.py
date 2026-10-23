"""Tranche matching within a borrower on lien, spread and maturity -- never on identifier text.

A borrower can have several loans (term loan, revolver, delayed-draw, second lien). The Risk
Dimensions doc wants signals per tranche: a PIK toggle must happen "without a change of
tranche", and dispersion compares holders' prices on the same loan. investment_identifier is
free text that BDCs reword, so tranches are told apart by their terms instead:

    lien       instrument_seniority, grouped (LIEN_GROUPS): 1L (incl. unitranche), 2L, SUB,
               UNSEC, CLO, and ANY for SENIOR_SECURED_UNSPEC / OTHER_UNKNOWN
    maturity   InvestmentMaturityDate (tagged by ~110 of 190 BDCs; 47% of debt rows in 2026Q2)
    spread     InvestmentBasisSpreadVariableRate (92% of debt rows)

Two known lien groups never match. ANY matches any group, but only when maturity or spread
confirms it: one BDC often labels a loan SENIOR_SECURED_1L while another leaves it OTHER_UNKNOWN.

Steps (build_tranche_panel runs them all):

1. standardize_debt_rows: debt rows (par > 0) with borrower_key, lien, lien_group, maturity,
   spread, manager family, a DIP flag, and two flags:
     funded     FV > 0, cost > 0, cost >= 5% of par. Undrawn commitments are left out.
     price_ok   funded and cost >= 80% of par. Below that, par is not what was lent (a partly
                drawn revolver / delayed-draw line reports the whole commitment as par; a cost
                write-down or deep-discount purchase also qualifies), so FV / par is not a
                price. About 4% of funded rows; they stay in the chains but not in prices.
2. collapse_tranches: one row per BDC x quarter x borrower x lien x maturity x spread. A
   BDC's pieces of one loan with identical terms (e.g. a term loan and its funded
   delayed-draw piece) become one tranche.
3. link_tranche_chains: links each tranche to a lien-compatible tranche of the same BDC and
   borrower in its previous observed quarter (gap <= 2 quarters) -> chain_id. Rules, in order:
       1  maturity within 31 days and spread within 25 bp (or spread missing)
       2  maturity within 31 days, spread changed (repricing)
       3  maturity missing, spread within 25 bp; or neither side reports either term
       4  what is left: the only remaining pair, or pairs by closest par (within 2x)
          -- an amendment that moved both terms; the weakest link
4. match_tranches_across_bdcs: groups the same quarter's tranches of one borrower across
   BDCs -> xb_id, with a confidence tier per tranche:
       1  maturity within 31 days (spread within 25 bp when both report it)
       2  maturity missing, joined on spread within 25 bp
       3  maturity and spread missing; joined because the borrower x lien has one loan
   A cluster never holds two tranches of the same BDC through tiers 2-3. Rows with no
   validated borrower ("ID: ..." keys) are never matched across BDCs.
5. add_pik_origination: P3, whether the loan had PIK when it first appeared in any BDC.

DIP loans are matched only to DIP loans (flagged from identifier text, which is classifying
text, not matching on it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FV_RAW = "InvestmentOwnedAtFairValue_normalized"
COST_RAW = "InvestmentOwnedAtCost_normalized"
PAR_RAW = "InvestmentOwnedBalancePrincipalAmount_normalized"
SPREAD_RAW = "InvestmentBasisSpreadVariableRate_normalized"
MATURITY_RAW = "InvestmentMaturityDate"

SPREAD_TOL = 0.0025          # 25 bp
MATURITY_TOL_DAYS = 31
MAX_CHAIN_GAP = 2            # quarters a chain may skip (missed filings)
MAX_PAR_RATIO = 2.0          # rule-4 links between several leftovers need par within 2x
FUNDED_MIN_COST_SHARE = 0.05
PRICE_MIN_COST_SHARE = 0.8   # rows priced only when cost >= 80% of par (see standardize_debt_rows)
MAX_PRICE = 1.2              # family prices above 120% of par are data errors (often par in GBP/EUR/AUD, FV in USD)
MIN_PIK_RATE = 0.0025        # a PIK rate under 25 bp is rounding noise from total - cash
DIP_PATTERN = r"\bDIP\b|debtor[\s-]+in[\s-]+possession"
RATE_COLS = ["PIC_Final", "PIK_Final", "IR_Final", "rate_cash", "interest_rate", "rate_pik", "spread"]
CARRY_COLS = ["family", "sector", "check_2", "borrower_canonical", "instrument_seniority", "bdc_type"]

# instrument_seniority -> lien group. Known groups never match each other; ANY (a seniority the
# label could not pin down) matches any group when maturity or spread confirms it.
LIEN_GROUPS = {
    "SENIOR_SECURED_1L": "1L", "UNITRANCHE": "1L",
    "SENIOR_SECURED_2L": "2L",
    "SUBORDINATED_MEZZ": "SUB",
    "SENIOR_UNSECURED": "UNSEC",
    "CLO_STRUCTURED_EQUITY": "CLO",
    "SENIOR_SECURED_UNSPEC": "ANY", "OTHER_UNKNOWN": "ANY",
}

_NO_DATE = pd.Timestamp("1900-01-01")
_NO_SPREAD = -9.0


def _to_pandas(df) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else df.to_pandas()


def standardize_debt_rows(df, add_families: bool = True) -> pd.DataFrame:
    """Debt rows (par > 0) of FINAL_DATASET (raw or after load_and_prepare_investment_data)."""
    d = _to_pandas(df).copy()
    for raw, alias in [(FV_RAW, "FV"), (COST_RAW, "COST"), (PAR_RAW, "PAR")]:
        if alias not in d.columns:
            d[alias] = pd.to_numeric(d[raw], errors="coerce")
    d = d[d["PAR"] > 0].copy()
    d["cik"] = pd.to_numeric(d["cik"]).astype("int64")
    b = d["borrower_canonical"].astype("string").str.strip()
    d["borrower_key"] = b.where(b.notna() & (b != ""), "ID: " + d["investment_identifier"].astype(str))
    d["lien"] = d["instrument_seniority"].fillna("OTHER_UNKNOWN")
    d["lien_group"] = d["lien"].map(LIEN_GROUPS).fillna("ANY")
    d["maturity"] = pd.to_datetime(d[MATURITY_RAW], errors="coerce")
    d["spread"] = pd.to_numeric(d[SPREAD_RAW], errors="coerce")
    d["is_dip"] = d["investment_identifier"].astype(str).str.contains(DIP_PATTERN, case=False, regex=True)
    d["qsort"] = d["cal_q"].str[:4].astype(int) * 4 + d["cal_q"].str[-1].astype(int)
    d["funded"] = (d["FV"] > 0) & (d["COST"] > 0) & (d["COST"] >= FUNDED_MIN_COST_SHARE * d["PAR"])
    # FV / par is a price only when par is what was lent. A partly drawn revolver or delayed-draw
    # line reports the whole commitment as par but only the drawn amount as cost; a cost
    # write-down or a deep-discount purchase also puts cost well below par. Such rows stay in
    # the chains (same loan) but are left out of prices.
    d["price_ok"] = d["funded"] & (d["COST"] >= PRICE_MIN_COST_SHARE * d["PAR"])
    if add_families and "family" not in d.columns:
        from manager_family import add_family
        d = add_family(d)
    return d


def collapse_tranches(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per BDC x quarter x borrower x lien x DIP x maturity x spread, funded rows only.

    FV, COST and PAR are summed; rates are weighted by par over the rows that report them;
    other columns come from the largest-FV row. FV_px / PAR_px sum only the price_ok rows,
    and price = FV_px / PAR_px (NaN when no row can be priced). Adds n_source_rows (rows
    combined)."""
    r = rows[rows["funded"]].copy()
    r["_mat"] = r["maturity"].fillna(_NO_DATE)
    r["_spr"] = r["spread"].round(4).fillna(_NO_SPREAD)
    r["FV_px"] = r["FV"].where(r["price_ok"], 0.0)
    r["PAR_px"] = r["PAR"].where(r["price_ok"], 0.0)
    keys = ["cik", "cal_q", "qsort", "borrower_key", "lien", "lien_group", "is_dip", "_mat", "_spr"]
    rates = [c for c in RATE_COLS if c in r.columns and c != "spread"]
    for c in rates:
        has = r[c].notna()
        r["_num_" + c] = (r[c] * r["PAR"]).where(has, 0.0)
        r["_den_" + c] = r["PAR"].where(has, 0.0)
    sums = ["FV", "COST", "PAR", "FV_px", "PAR_px"] + [f"_{k}_{c}" for c in rates for k in ("num", "den")]
    g = r.groupby(keys, sort=False)
    out = g[sums].sum()
    out["n_source_rows"] = g.size()
    if "non_accrual" in r.columns:   # 1 if any piece is flagged; NaN if all pieces are unknown
        out["non_accrual"] = pd.to_numeric(r["non_accrual"], errors="coerce").groupby([r[k] for k in keys], sort=False).max()
    for c in rates:
        den = out.pop("_den_" + c)
        out[c] = (out.pop("_num_" + c) / den).where(den > 0)
    carry = [c for c in CARRY_COLS if c in r.columns]
    top = r.sort_values("FV", ascending=False).drop_duplicates(keys).set_index(keys)[carry]
    out = out.join(top).reset_index()
    out["maturity"] = out["_mat"].where(out["_mat"] != _NO_DATE)
    out["spread"] = out["_spr"].where(out["_spr"] != _NO_SPREAD)
    out["price"] = (out["FV_px"] / out["PAR_px"]).where(out["PAR_px"] > 0)
    return out.drop(columns=["_mat", "_spr"])


# ---------------------------------------------------------------- matching helpers

def _days(s: pd.Series) -> np.ndarray:
    return ((s - pd.Timestamp("1970-01-01")).dt.days).to_numpy(dtype=float)


def _close(a: float, b: float, tol: float) -> bool:
    return a == a and b == b and abs(a - b) <= tol + 1e-9


def _compatible(la: str, lb: str) -> bool:
    """Lien groups match if equal, or if either is the wildcard ANY (UNSPEC / OTHER_UNKNOWN)."""
    return la == lb or la == "ANY" or lb == "ANY"


def _pair_rule(mp: float, mc: float, sp: float, sc: float) -> int:
    """Chain rule for one prev/current pair (0 = no match under rules 1-3). NaN = not reported."""
    if _close(mp, mc, MATURITY_TOL_DAYS):
        return 1 if (sp != sp or sc != sc or _close(sp, sc, SPREAD_TOL)) else 2
    if (mp != mp or mc != mc) and _close(sp, sc, SPREAD_TOL):
        return 3
    if mp != mp and mc != mc and sp != sp and sc != sc:
        return 3                                 # neither reports maturity or spread: terms agree as far as known
    return 0


def _match_blocks(P, C, mat, spr, par, lg):
    """Greedy one-to-one links between previous-quarter tranches P and current tranches C.

    Only lien-compatible pairs link. Within a rule, the closest pair links first. A missing
    maturity or spread counts as half a tolerance away (so an exact match beats an unknown
    one), and par size breaks the rest of the ties (a term loan stays with the term loan, a new
    small rescue loan starts its own chain). Rule 4 links what is left: always for one
    compatible tranche on each side; for several, by closest par, within MAX_PAR_RATIO."""
    if len(P) == 1 and len(C) == 1:
        p, c = P[0], C[0]
        if not _compatible(lg[p], lg[c]):
            return []
        return [(p, c, _pair_rule(mat[p], mat[c], spr[p], spr[c]) or 4)]
    cands = []
    for p in P:
        for c in C:
            if not _compatible(lg[p], lg[c]):
                continue
            rule = _pair_rule(mat[p], mat[c], spr[p], spr[c])
            if rule:
                dm, ds = abs(mat[p] - mat[c]), abs(spr[p] - spr[c])
                dist = ((dm / MATURITY_TOL_DAYS if dm == dm else 0.5) + (ds / SPREAD_TOL if ds == ds else 0.5)
                        + 0.1 * abs(np.log(par[c] / par[p])) + (0.0 if lg[p] == lg[c] else 0.25))
                cands.append((rule, dist, p, c))
    cands.sort()
    freeP, freeC, links = set(P), set(C), []
    for rule, _, p, c in cands:
        if p in freeP and c in freeC:
            links.append((p, c, rule))
            freeP.discard(p)
            freeC.discard(c)
    single = len(freeP) == 1 and len(freeC) == 1
    rest = sorted((abs(np.log(par[c] / par[p])), p, c) for p in freeP for c in freeC if _compatible(lg[p], lg[c]))
    for lr, p, c in rest:
        if (single or lr <= np.log(MAX_PAR_RATIO)) and p in freeP and c in freeC:
            links.append((p, c, 4))
            freeP.discard(p)
            freeC.discard(c)
    return links


def link_tranche_chains(tr: pd.DataFrame, max_gap: int = MAX_CHAIN_GAP) -> pd.DataFrame:
    """Add chain_id (same loan, same BDC, over time), link_rule (0 = chain start) and q_gap.

    Candidates are all of the BDC's tranches in the borrower that are lien-compatible, so a
    relabel from SENIOR_SECURED_1L to OTHER_UNKNOWN does not break the chain."""
    tr = tr.sort_values(["cik", "borrower_key", "is_dip", "qsort"]).reset_index(drop=True)
    n = len(tr)
    chain = np.full(n, -1, dtype=np.int64)
    rule = np.zeros(n, dtype=np.int8)
    gap = np.full(n, np.nan)
    mat, spr, q = _days(tr["maturity"]), tr["spread"].to_numpy(dtype=float), tr["qsort"].to_numpy()
    par, lg = tr["PAR"].to_numpy(dtype=float), tr["lien_group"].to_numpy()
    next_id = 0
    for idx in tr.groupby(["cik", "borrower_key", "is_dip"], sort=False).indices.values():
        idx = np.sort(idx)
        blocks = np.split(idx, np.flatnonzero(np.diff(q[idx])) + 1)
        prev = None
        for blk in blocks:
            if prev is not None and q[blk[0]] - q[prev[0]] <= max_gap:
                for p, c, r in _match_blocks(list(prev), list(blk), mat, spr, par, lg):
                    chain[c], rule[c], gap[c] = chain[p], r, q[c] - q[p]
            for c in blk:
                if chain[c] < 0:
                    chain[c] = next_id
                    next_id += 1
            prev = blk
    tr["chain_id"], tr["link_rule"], tr["q_gap"] = chain, rule, gap
    return tr


def _single_link(order_vals: np.ndarray, tol: float) -> np.ndarray:
    """Cluster labels for already-sorted values: a new cluster wherever the gap exceeds tol."""
    return np.concatenate([[0], np.cumsum(np.diff(order_vals) > tol + 1e-9)]) if len(order_vals) else order_vals


def _cluster_items(idx, mat, spr, cik, allow_fallback: bool):
    """Cross-BDC clusters of one lien group's tranches. Returns (clusters, tiers): clusters is a
    list of member lists, tiers maps position -> tier (NaN = unplaced, alone in its cluster).

    Maturity first (tier 1), then spread for tranches without maturity (tier 2), never joining a
    cluster that already holds the same BDC. Tranches with neither join (tier 3) only when
    allow_fallback and every BDC holds a single tranche (the borrower x lien has one loan)."""
    clusters, tiers = [], {}
    one_each = len({cik[i] for i in idx}) == len(idx)
    M = sorted([i for i in idx if mat[i] == mat[i]], key=lambda i: mat[i])
    N = [i for i in idx if mat[i] != mat[i]]
    if M:
        lab = _single_link(np.array([mat[i] for i in M]), MATURITY_TOL_DAYS)
        for g in np.unique(lab):
            grp = [M[j] for j in np.flatnonzero(lab == g)]
            with_s = sorted([i for i in grp if spr[i] == spr[i]], key=lambda i: spr[i])
            no_s = [i for i in grp if spr[i] != spr[i]]
            subs = []
            if with_s:
                sl = _single_link(np.array([spr[i] for i in with_s]), SPREAD_TOL)
                subs = [[with_s[j] for j in np.flatnonzero(sl == s)] for s in np.unique(sl)]
            if len(subs) <= 1:
                clusters.append((subs[0] if subs else []) + no_s)
                tiers.update({i: 1 for i in clusters[-1]})
            else:
                for s in subs:
                    clusters.append(s)
                    tiers.update({i: 1 for i in s})
                for i in no_s:                      # can't tell which spread group they belong to
                    clusters.append([i])
                    tiers[i] = np.nan
    leftovers = []
    for i in [i for i in N if spr[i] == spr[i]]:
        hits = [ci for ci, s in enumerate(clusters)
                if all(tiers[j] == tiers[j] for j in s) and all(cik[j] != cik[i] for j in s)
                and any(_close(spr[j], spr[i], SPREAD_TOL) for j in s)]
        if len(hits) == 1:
            clusters[hits[0]].append(i)
            tiers[i] = 2
        elif not hits:
            leftovers.append(i)
        else:
            clusters.append([i])                    # ambiguous: its own, unmatched cluster
            tiers[i] = np.nan
    if leftovers:
        leftovers.sort(key=lambda i: spr[i])
        sl = _single_link(np.array([spr[i] for i in leftovers]), SPREAD_TOL)
        for s in np.unique(sl):
            grp = [leftovers[j] for j in np.flatnonzero(sl == s)]
            # one BDC with two near-equal spreads: two loans, so group on the exact spread instead
            parts = ([grp] if len({cik[i] for i in grp}) == len(grp)
                     else [[i for i in grp if spr[i] == v] for v in sorted({spr[i] for i in grp})])
            for part in parts:
                clusters.append(part)
                tiers.update({i: 2 for i in part})
    none = [i for i in N if spr[i] != spr[i]]
    if none:
        if allow_fallback and one_each and len(clusters) <= 1:
            if not clusters:
                clusters.append([])
            clusters[0].extend(none)
            tiers.update({i: 3 for i in none})
        else:
            for i in none:
                clusters.append([i])
                tiers[i] = np.nan
    return clusters, tiers


def _cluster_borrower(idx, mat, spr, cik, lg):
    """Cross-BDC clusters within one quarter x borrower x DIP. Returns {pos: (cluster, tier)}.

    Known lien groups are clustered separately and never mixed. A wildcard-lien tranche
    (UNSPEC / OTHER_UNKNOWN) joins a known-lien cluster only when its terms single it out:
    maturity within 31 days (tier 1), or spread within 25 bp with maturity missing on either
    side (tier 2). Wildcards left over are clustered among themselves; their no-terms fallback
    (tier 3) applies only when the borrower has no known-lien tranche at all."""
    groups = {}
    for i in idx:
        groups.setdefault(lg[i], []).append(i)
    wild = groups.pop("ANY", [])
    clusters, tiers = [], {}
    for items in groups.values():
        c, t = _cluster_items(items, mat, spr, cik, allow_fallback=True)
        clusters += c
        tiers.update(t)
    placed = [ci for ci, s in enumerate(clusters) if all(tiers[j] == tiers[j] for j in s)]
    left = []
    for i in wild:
        free = [ci for ci in placed if all(cik[j] != cik[i] for j in clusters[ci])]
        by_mat = [ci for ci in free if mat[i] == mat[i] and any(
            _close(mat[j], mat[i], MATURITY_TOL_DAYS)
            and (spr[j] != spr[j] or spr[i] != spr[i] or _close(spr[j], spr[i], SPREAD_TOL))
            for j in clusters[ci])]
        by_spr = [ci for ci in free if any(
            (mat[i] != mat[i] or mat[j] != mat[j]) and _close(spr[j], spr[i], SPREAD_TOL) for j in clusters[ci])]
        if len(by_mat) == 1:
            clusters[by_mat[0]].append(i)
            tiers[i] = 1
        elif not by_mat and len(by_spr) == 1:
            clusters[by_spr[0]].append(i)
            tiers[i] = 2
        else:
            left.append(i)
    if left:
        c, t = _cluster_items(left, mat, spr, cik, allow_fallback=not clusters)
        clusters += c
        tiers.update(t)
    return {i: (ci, tiers[i]) for ci, s in enumerate(clusters) for i in s}


def match_tranches_across_bdcs(tr: pd.DataFrame) -> pd.DataFrame:
    """Add xb_id (same loan across BDCs in one quarter) and xb_tier (1 best, 3 weakest; NaN =
    could not be placed). A tranche no other BDC holds still gets its own xb_id."""
    tr = tr.reset_index(drop=True)
    n = len(tr)
    xb = np.full(n, -1, dtype=np.int64)
    tier = np.full(n, np.nan)
    mat, spr = _days(tr["maturity"]), tr["spread"].to_numpy(dtype=float)
    cik, lg = tr["cik"].to_numpy(), tr["lien_group"].to_numpy()
    ok = ~tr["borrower_key"].str.startswith("ID: ")
    sub = tr[ok]
    base = 0
    for idx in sub.groupby(["cal_q", "borrower_key", "is_dip"], sort=False).indices.values():
        pos = sub.index.to_numpy()[idx]
        res = _cluster_borrower(list(pos), mat, spr, cik, lg)
        for i, (c, t) in res.items():
            xb[i], tier[i] = base + c, t
        base += max(c for c, _ in res.values()) + 1
    for i in np.flatnonzero(xb < 0):                  # "ID: ..." rows: never matched
        xb[i] = base
        base += 1
    tr["xb_id"], tr["xb_tier"] = xb, tier
    return tr


def add_pik_origination(tr: pd.DataFrame, pik_col: str = "PIK_Final", min_pik: float = MIN_PIK_RATE) -> pd.DataFrame:
    """P3: did the loan carry PIK the first time it appeared in any BDC's schedule?

    For each chain, look at the cross-BDC group of its first quarter: the loan's first
    appearance is the earliest chain start among those BDCs, and P3 is whether any chain that
    started then had PIK. One hop only: holders that left before this chain began are not seen.
    Loans first seen in the panel's first quarter are censored (P3 unknown, null)."""
    tr = tr.copy()
    tr["chain_start_q"] = tr.groupby("chain_id")["qsort"].transform("min")
    is_start = tr["qsort"] == tr["chain_start_q"]
    starts = (tr[is_start].assign(_pik=tr[pik_col].fillna(0) >= min_pik)
              .groupby("chain_id").agg(o_start=("chain_start_q", "first"), o_pik=("_pik", "max")))
    xb_at_start = tr.loc[is_start, ["chain_id", "xb_id"]].drop_duplicates()
    # every chain present in the same cross-BDC group as this chain's first quarter
    members = tr[["xb_id", "chain_id"]].drop_duplicates().rename(columns={"chain_id": "other"})
    m = xb_at_start.merge(members, on="xb_id").merge(starts, left_on="other", right_index=True)
    m = m[m["o_start"] == m.groupby("chain_id")["o_start"].transform("min")]
    orig = m.groupby("chain_id").agg(loan_first_q=("o_start", "min"), pik_at_origination=("o_pik", "max"))
    tr = tr.join(orig, on="chain_id")
    censored = tr["loan_first_q"] <= tr["qsort"].min()
    tr["origination_censored"] = censored
    tr["pik_at_origination"] = tr["pik_at_origination"].astype("boolean").mask(censored)
    return tr.drop(columns=["chain_start_q"])


def build_tranche_panel(df, with_origination: bool = True) -> pd.DataFrame:
    """standardize -> collapse -> chains -> cross-BDC match (-> P3 when PIK_Final is present)."""
    tr = collapse_tranches(standardize_debt_rows(df))
    tr = match_tranches_across_bdcs(link_tranche_chains(tr))
    if with_origination and "PIK_Final" in tr.columns:
        tr = add_pik_origination(tr)
    return tr


def family_tranche_prices(tr: pd.DataFrame, max_price: float = MAX_PRICE, min_families: int = 1) -> pd.DataFrame:
    """One row per quarter x cross-BDC tranche x manager family: the family's FV / par over its
    priceable rows (price_ok), summed over its BDCs (one family = one holder). Prices above
    max_price (data errors) are dropped. tier is the family's worst match tier in that tranche
    (tier_missing = some member could not be placed). lien_group is the tranche's known lien
    group (ANY if none known). min_families > 1 keeps only tranches held by that many
    families (much faster). FV and PAR in the output are the priced amounts."""
    t = tr[(tr["xb_id"] >= 0) & (tr["PAR_px"] > 0)]
    t = t.assign(FV=t["FV_px"], PAR=t["PAR_px"])
    if min_families > 1:
        nfam = t.drop_duplicates(["xb_id", "family"]).groupby("xb_id").size()
        t = t[t["xb_id"].isin(nfam.index[nfam >= min_families])]
    t = t.assign(_miss=t["xb_tier"].isna(), _wild=t["lien_group"] == "ANY",
                 sector=t["sector"] if "sector" in t else None)
    known = t.sort_values("_wild").drop_duplicates("xb_id").set_index("xb_id")["lien_group"]
    fam = (t.groupby(["cal_q", "xb_id", "family"])
             .agg(FV=("FV", "sum"), PAR=("PAR", "sum"), tier=("xb_tier", "max"), tier_missing=("_miss", "max"),
                  n_bdcs=("cik", "nunique"), borrower_key=("borrower_key", "first"), sector=("sector", "first"))
             .reset_index())
    fam["lien_group"] = fam["xb_id"].map(known)
    fam["price"] = fam["FV"] / fam["PAR"]
    return fam[fam["price"] <= max_price]


def tranche_dispersion(tr: pd.DataFrame, min_families: int = 2, max_tier: float = 3,
                       max_price: float = MAX_PRICE) -> pd.DataFrame:
    """Valuation dispersion per cross-BDC tranche and quarter (Risk Dimensions V1, V2, V4).

    Prices come from family_tranche_prices (FV / par, one family = one holder). Keeps
    tranches with at least min_families families whose worst member tier is <= max_tier.

      V4  consensus price: median family price
      V1  dispersion: std (ddof=1) and IQR of family prices; range for reference
      V2  low mark: lowest family price minus the median (<= 0)
    plus within-quarter percentiles V1_pct (std), V2_pct (low = most pessimistic), V4_pct."""
    fam = family_tranche_prices(tr, max_price, min_families=min_families)
    if fam.empty:
        return pd.DataFrame(columns=["cal_q", "xb_id", "borrower_key", "lien_group", "sector", "n_families",
                                     "n_bdcs", "total_par", "tier", "V4_consensus", "V1_std", "p_min", "p_max",
                                     "V1_iqr", "V1_range", "V2_low_mark", "V1_pct", "V2_pct", "V4_pct"])
    g = fam.groupby(["cal_q", "xb_id"])
    out = g.agg(borrower_key=("borrower_key", "first"), lien_group=("lien_group", "first"),
                sector=("sector", "first"), n_families=("family", "nunique"), n_bdcs=("n_bdcs", "sum"),
                total_par=("PAR", "sum"), tier=("tier", "max"), V4_consensus=("price", "median"),
                V1_std=("price", "std"), p_min=("price", "min"), p_max=("price", "max"),
                tier_missing=("tier_missing", "max"))
    out = out.join(g["price"].quantile([0.25, 0.75]).unstack().rename(columns={0.25: "p25", 0.75: "p75"})).reset_index()
    out = out[(out["n_families"] >= min_families) & ~out["tier_missing"] & (out["tier"] <= max_tier)].copy()
    out["V1_iqr"] = out["p75"] - out["p25"]
    out["V1_range"] = out["p_max"] - out["p_min"]
    out["V2_low_mark"] = out["p_min"] - out["V4_consensus"]
    for c, name in [("V1_std", "V1_pct"), ("V2_low_mark", "V2_pct"), ("V4_consensus", "V4_pct")]:
        out[name] = out.groupby("cal_q")[c].rank(pct=True) * 100
    return out.drop(columns=["p25", "p75", "tier_missing"]).sort_values(["cal_q", "V1_std"], ascending=[True, False])
