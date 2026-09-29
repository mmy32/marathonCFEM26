"""
IXBRL cleaning utilities: currency, dollar-scale and interest-rate cleaning.

The rate logic rests on two identities:

    (I1)  IR = PIC + PIK                      coupon identity (all loans)
    (I2)  IR ~= Base_q + Spread, with floors  floating-rate identity, Base_q = the loan
                                              currency's rate from SOFR_augmented.csv

Every repair is one of two things:
    1. a *role repair*  - a field holds a value that belongs in another field, detected
                          because moving it makes I1 or I2 hold;
    2. a *fill*         - one term of I1 / I2 is missing and the others are present.
A row that contradicts I1 / I2 with no move that fixes it is left as reported.

Rules live in one ordered table (RATE_RULES). Each rule is (tag, condition, assignment);
`apply_rules` handles masking, tagging and bookkeeping in one place, so a rule is ~5 lines.
Every rule is kept only because removing it breaks I1, I2 or a sign on a measurable share of
rows (see README_2.md, "Why each rule is kept").

Output columns:
    rate_config   which of spread/rate/pik/pic were present on input   (also written as check_2)
    rate_source   reported | derived | estimated | zero | none
    check_1       'resolved' iff the row ends with a usable non-zero IR
    estimate      base + spread (with floors) for floating/untyped rows with a spread
    change_tracker  pipe-joined tags of every rule that fired
    RateType      base-rate member from the XBRL enumeration (e.g. 'FixedRateMember')
    is_fixed      True / False from RateType, <NA> when the filer did not tag it
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np
import pandas as pd

# ============================================================
# Column names & constants (defined once)
# ============================================================

SPREAD = "InvestmentBasisSpreadVariableRate_normalized"
RATE = "InvestmentInterestRate_normalized"
PIC = "InvestmentInterestRatePaidInCash_normalized"
PIK = "InvestmentInterestRatePaidInKind_normalized"
FLOOR = "InvestmentInterestRateFloor_normalized"
RATE_TYPE_RAW = "InvestmentVariableInterestRateTypeExtensibleEnumeration"
FIXED_MEMBERS = ["FixedMember", "FixedRateMember"]
RATE_FIELDS = {"spread": SPREAD, "rate": RATE, "pik": PIK, "pic": PIC}  # check_2 order

FV_RAW, COST_RAW, PRIN_RAW = (
    "InvestmentOwnedAtFairValue",
    "InvestmentOwnedAtCost",
    "InvestmentOwnedBalancePrincipalAmount",
)
VALUE_COLS = [FV_RAW, COST_RAW, PRIN_RAW]
UNIT_COLS = [c + "-unitRef" for c in VALUE_COLS]

SPREAD_RANGE = (0.00, 0.20)
RATE_RANGE = (0.00, 0.50)
RATE_SCALES = (100, 10_000)             # percent / basis-point tagging errors

TOL = 1e-6                              # float-equality tolerance, used everywhere
BASE_TOL = 0.005                        # 50bp: "differs by the base rate" test
PRIME_OVER_SOFR = 0.032                 # Prime ~ SOFR + 3.2% (2023-2026 quarterly averages)
SPREAD_LIKE_RANGE = (0.02, 0.09)        # untagged spread == IR: a value in here is a spread,
                                        # above it a coupon (tagged loans: floating <= 7.5%, fixed >= 10%)
_PRIME_TEXT = r"(?i)\bprime\s*(?:rate\s*)?[+\-–]"   # "Prime + 1.35%", "Prime - 1.15%"

COUNTRY_TO_CURRENCY = {
    "United States": "USD", "Australia": "AUD", "Canada": "CAD", "Singapore": "SGD",
    "Hong Kong": "HKD", "New Zealand": "NZD", "United Kingdom": "GBP", "Euro Area": "EUR",
    "Euro Zone": "EUR", "Japan": "JPY", "Switzerland": "CHF", "China": "CNY",
    "South Korea": "KRW", "Taiwan": "TWD", "India": "INR", "Mexico": "MXN",
    "Brazil": "BRL", "Argentina": "ARS", "Chile": "CLP", "Colombia": "COP",
    "Peru": "PEN", "South Africa": "ZAR", "Norway": "NOK", "Sweden": "SEK",
    "Denmark": "DKK", "Poland": "PLN", "Czech Republic": "CZK", "Hungary": "HUF",
    "Turkey": "TRY", "Israel": "ILS", "Saudi Arabia": "SAR", "United Arab Emirates": "AED",
    "Thailand": "THB", "Malaysia": "MYR", "Philippines": "PHP", "Indonesia": "IDR",
    "Vietnam": "VND", "Pakistan": "PKR", "Bangladesh": "BDT", "Russia": "RUB",
    "Nigeria": "NGN", "Egypt": "EGP",
}
CURRENCY_CODES = sorted(set(COUNTRY_TO_CURRENCY.values()))


# ============================================================
# Generic helpers
# ============================================================

def close(a, b, tol: float = TOL) -> pd.Series:
    """NaN-safe |a - b| <= tol. The only equality test used in this module."""
    return (a - b).abs() <= tol


def append_tag(df: pd.DataFrame, mask: pd.Series, tag: str, col: str = "change_tracker") -> None:
    """In-place: append `tag` to a pipe-delimited column on `mask` rows."""
    if not mask.any():
        return
    prev = df.loc[mask, col].astype("string").fillna("")
    df.loc[mask, col] = np.where(prev.eq(""), tag, prev + "|" + tag)


def rescale_to_range(x: pd.Series, lo: float, hi: float,
                     scales: Iterable[float]) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Vectorised power-of-ten rescaling shared by rate and dollar-value normalisation.
    In-range values pass through; otherwise the candidate scale whose result lands
    closest to the band midpoint wins; nothing lands -> 'unresolved'.
    Returns (rescaled values, flag, divisor applied or NaN).
    """
    a = x.abs()
    ok = a.between(lo, hi)
    mid = (lo + hi) / 2
    best_scale = pd.Series(np.nan, index=x.index)
    best_dist = pd.Series(np.inf, index=x.index)
    for s in scales:
        y = a / s
        cand = ~ok & y.between(lo, hi) & ((y - mid).abs() < best_dist)
        best_scale[cand] = s
        best_dist[cand] = (y - mid).abs()[cand]
    out = x.where(ok | best_scale.isna(), x / best_scale)
    flag = pd.Series("unresolved", index=x.index, dtype=object)
    flag[ok] = "unchanged"
    flag[best_scale.notna()] = "div" + best_scale[best_scale.notna()].astype(int).astype(str)
    flag[x.isna()] = "na"
    return out, flag, best_scale


# ============================================================
# 1. Rate normalisation
# ============================================================

RATE_COLS = {
    "InvestmentBasisSpreadVariableRate": SPREAD_RANGE,
    "InvestmentInterestRateFloor": SPREAD_RANGE,
    "InvestmentInterestRate": RATE_RANGE,
    "InvestmentInterestRatePaidInCash": RATE_RANGE,
    "InvestmentInterestRatePaidInKind": RATE_RANGE,
}
SIGN_FIX_COLS = ["InvestmentInterestRatePaidInCash", "InvestmentInterestRatePaidInKind",
                 "InvestmentInterestRateFloor"]


def normalize_interest_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, (lo, hi) in RATE_COLS.items():
        if col not in out:
            continue
        v, f, _ = rescale_to_range(out[col], lo, hi, RATE_SCALES)
        if col in SIGN_FIX_COLS:
            neg = v < 0
            v = v.abs()
            f = f.where(~neg, f + "|sign_error")
        out[f"{col}_normalized"], out[f"{col}_scale_flag"] = v, f
    return fix_component_scale(out)


COMPONENT_EXCESS = 0.10    # a component this far above the row's coupon is a percent-tagging error


def fix_component_scale(out: pd.DataFrame) -> pd.DataFrame:
    """
    Row-level scale check for PIC / PIK. Each column is rescaled on its own above, so a raw
    0.50 meaning 0.50% ("(including 50 PIK)") passes through as 50%, and a row can mix a
    percent-tagged IR (8.96 -> 8.96%) with a PIK left at 0.48. A component that exceeds the
    row's coupon (IR, or spread + 6% when IR is missing) by more than COMPONENT_EXCESS, and
    fits under it once divided by 100, was tagged in percent.

    "50/50 Cash/PIK" tagged as PIC = PIK = 0.5 is a split, not two 50% rates: each half of IR.
    """
    if not {RATE, PIC, PIK} <= set(out.columns):
        return out
    rate = out[RATE]
    split = close(out[PIC], 0.5) & close(out[PIK], 0.5) & rate.between(0, RATE_RANGE[1] / 2)
    for c in (PIC, PIK):
        out.loc[split, c] = rate[split] / 2
        fc = c.replace("_normalized", "_scale_flag")
        out.loc[split, fc] = out.loc[split, fc].astype(str) + "|split_50_50"
    ref = rate.fillna(out[SPREAD] + 0.06) if SPREAD in out else rate
    for c in (PIC, PIK):
        bad = (out[c] > 0.2) & (out[c] > ref + COMPONENT_EXCESS) & (out[c] / 100 <= ref)
        out.loc[bad, c] = out.loc[bad, c] / 100
        fc = c.replace("_normalized", "_scale_flag")
        out.loc[bad, fc] = out.loc[bad, fc].astype(str) + "|row_div100"
    return out


# ============================================================
# 2. Currency
# ============================================================

_CCY_RE = re.compile(r"(?<![A-Z])(" + "|".join(CURRENCY_CODES) + r")(?![A-Z])")


def normalize_currency(s: pd.Series) -> pd.Series:
    """Vectorised: first ISO code appearing as a standalone token (so 'Unit_Standard_USD_x'
    matches USD, but a random hash like '...ZNOKq...' does not match NOK)."""
    return s.astype("string").str.upper().str.extract(_CCY_RE, expand=False)


def determine_currency(df: pd.DataFrame) -> pd.Series:
    """Principal > FairValue > Cost priority, USD if all missing. (If the three agree,
    priority order returns the same value, so no separate 'consistent' branch is needed.)"""
    prio = [PRIN_RAW + "-unitRef_normalized", FV_RAW + "-unitRef_normalized", COST_RAW + "-unitRef_normalized"]
    return df[prio].bfill(axis=1).iloc[:, 0].fillna("USD")


def prepare_fx_data(fx_path: str) -> pd.DataFrame:
    fx = pd.read_csv(fx_path)
    fx["cal_q"] = pd.to_datetime(fx["Effective Date"]).dt.to_period("Q").astype(str)
    country = fx["Country - Currency Description"].str.split("-", n=1).str[0].str.strip()
    fx["currency_norm"] = country.map(COUNTRY_TO_CURRENCY)
    fx["fx_to_usd"] = pd.to_numeric(fx["Exchange Rate"], errors="coerce")
    return fx.groupby(["cal_q", "currency_norm"], as_index=False)["fx_to_usd"].mean()


def convert_currencies(df: pd.DataFrame, fx_path: str) -> pd.DataFrame:
    out = df.copy()
    out["cal_q"] = out["cal_q"].astype(str)
    fx = prepare_fx_data(fx_path).set_index(["cal_q", "currency_norm"])["fx_to_usd"]
    to_usd = lambda x, ccy: x.where(ccy.isna() | ccy.eq("USD"),
                                    x / pd.Series(pd.MultiIndex.from_arrays([out["cal_q"], ccy]).map(fx),
                                                  index=out.index))
    for v, u in zip(VALUE_COLS, UNIT_COLS):
        ccy = normalize_currency(out[u])
        out[u + "_normalized"] = ccy
        out[v + "_normalized"] = to_usd(out[v], ccy)

    # Principal in a different unit from FV and cost is normal (a EUR loan reported in USD).
    # It is a filer unit error only when converting it is what breaks it: read in the FV / cost
    # currency the principal sits within 3x of FV, converted it does not (CIK 1976336 tags USD
    # principal as U_EGP; left alone, the loan is labelled EGP and the value-scale check then
    # divides FV and cost by 1000).
    pu = out[PRIN_RAW + "-unitRef_normalized"]
    fu, cu = out[FV_RAW + "-unitRef_normalized"], out[COST_RAW + "-unitRef_normalized"]
    pv, fvn = out[PRIN_RAW + "_normalized"], out[FV_RAW + "_normalized"]
    alt = to_usd(out[PRIN_RAW], fu)
    # Only where the two currencies are at least 2x apart (EGP, NOK, SEK, DKK, CNY, JPY vs USD);
    # for EUR / GBP the FX move is too small to tell a unit error from a marked-down loan.
    within = lambda r: r.abs().between(1 / 3, 3)
    far_fx = ~(pv / alt).abs().between(0.5, 2)
    odd = (fu.notna() & fu.eq(cu) & pu.notna() & pu.ne(fu) & fvn.ne(0) & far_fx
           & within(alt / fvn) & ~within(pv / fvn)).fillna(False).astype(bool)
    out.loc[odd, PRIN_RAW + "_normalized"] = alt[odd]
    out.loc[odd, PRIN_RAW + "-unitRef_normalized"] = fu[odd]
    out["unit_mismatch_fixed"] = odd
    out["currency"] = determine_currency(out)
    return out


# ============================================================
# 3. Dollar-value scale correction
# ============================================================

VALUE_SCALE_CANDIDATES = (1_000, 1_000_000)
VALUE_RATIO_RANGE = (0.0, 3.0)
VALUE_RESCALE_TARGET_RANGE = (0.0, 3.0)   # see README "Known limitation: the rescale floor"
PRINCIPAL_COST_RANGE = (0.5, 2.0)         # principal / cost must land here after a 1e3 / 1e6 fix
PRINCIPAL_CHECK_MIN_COST = 1_000_000      # below this, principal >> cost is usually real: cost
                                          # written down (CLO equity) or an unfunded commitment


def normalize_value_scale(df, value_cols=(FV_RAW + "_normalized", COST_RAW + "_normalized"),
                          anchor_col=PRIN_RAW + "_normalized", position_key=("cik", "investment_identifier")):
    """
    Rescale FV / cost that are off by 1e3 / 1e6 relative to the row's own principal.

    Exception: when FV and cost both need the *same* rescale and the raw FV matches the same
    position's FV in its other rows (within 3x), the principal is the mis-scaled field. The
    principal is multiplied instead and FV / cost are left as reported.

    Principal too large (a unit count, or units vs thousands): principal is ~1e3 / 1e6 x cost,
    and dividing lands it within PRINCIPAL_COST_RANGE of cost. The check is anchored on cost,
    never FV, so real markdowns (FV far below cost, e.g. First Brands 2025Q3+) are untouched,
    and only the principal changes. Positions with cost under PRINCIPAL_CHECK_MIN_COST are
    skipped, since a written-down cost basis or an unfunded commitment can make that ratio real.
    """
    out = df.copy()
    if anchor_col not in out:
        return out
    a = out[anchor_col]
    has_anchor = a.notna() & a.ne(0)
    pflag = pd.Series("unchanged", index=out.index, dtype=object).where(a.notna(), "na")
    scales, flags = {}, {}
    for col in value_cols:
        if col not in out:
            continue
        v = out[col]
        ratio = (v / a).abs()
        ok = ratio.between(*VALUE_RATIO_RANGE)
        _, flag, scale = rescale_to_range(ratio.where(~ok), *VALUE_RESCALE_TARGET_RANGE,
                                          VALUE_SCALE_CANDIDATES)
        scales[col] = scale.where(has_anchor & ~ok)
        flags[col] = flag.where(~ok, "unchanged").where(has_anchor & v.notna(), "na")

    fv, cost = value_cols
    if fv in scales and cost in scales and all(k in out for k in position_key):
        same = scales[fv].notna() & scales[fv].eq(scales[cost])
        keys = [out[k] for k in position_key]
        ref = out[fv].abs().where(scales[fv].isna() & out[fv].ne(0)).groupby(keys).transform("median")
        # ...and multiplying the principal by that scale actually brings FV / principal to ~1
        # (a 10x currency-unit mismatch must not be "fixed" with a 1000x multiplier).
        prin_bad = (same & (out[fv].abs() / ref).between(1 / 3, 3)
                    & (out[fv] / (a * scales[fv])).abs().between(1 / 3, 3))
        out.loc[prin_bad, anchor_col] = a[prin_bad] * scales[fv][prin_bad]
        pflag[prin_bad] = "mul" + scales[fv][prin_bad].astype(int).astype(str)
        for col in (fv, cost):
            scales[col] = scales[col].mask(prin_bad)
            flags[col] = flags[col].mask(prin_bad, "unchanged")

    for col, scale in scales.items():
        out[col] = out[col].where(scale.isna(), out[col] / scale)
        out[f"{col}_scale_flag"] = flags[col]

    if cost in out:
        p, c = out[anchor_col], out[cost]
        big = c.abs() >= PRINCIPAL_CHECK_MIN_COST
        _, cflag, s = rescale_to_range((p / c).where(big), *PRINCIPAL_COST_RANGE, VALUE_SCALE_CANDIDATES)
        out[anchor_col] = p.where(s.isna(), p / s)
        pflag = pflag.where(s.isna(), cflag)
    out[f"{anchor_col}_scale_flag"] = pflag
    return out


# ============================================================
# 4. Filer-quarter outliers (leave-one-out median, vectorised)
# ============================================================

FILER_QUARTER_OUTLIER_RATIO = 20.0


def flag_filer_quarter_outliers(df, value_col=FV_RAW + "_normalized", filer_col="cik",
                                quarter_col="cal_q", ratio_threshold=FILER_QUARTER_OUTLIER_RATIO,
                                min_other_quarters=2):
    out = df.copy()
    tot = out[value_col].abs().groupby([out[filer_col], out[quarter_col]]).sum().rename("q").reset_index()

    def loo_median(q: pd.Series) -> pd.Series:
        v = q.to_numpy()
        if len(v) < 2:
            return pd.Series(np.nan, index=q.index)
        return pd.Series([np.median(np.delete(v, i)) for i in range(len(v))], index=q.index)

    tot["n"] = tot.groupby(filer_col)["q"].transform("size")
    tot["med_others"] = tot.groupby(filer_col)["q"].transform(loo_median)
    tot["filer_quarter_outlier"] = (tot["n"] > min_other_quarters) & (tot["med_others"] > 0) & \
                                   (tot["q"] / tot["med_others"] > ratio_threshold)
    key = tot.set_index([filer_col, quarter_col])["filer_quarter_outlier"]
    idx = pd.MultiIndex.from_frame(out[[filer_col, quarter_col]])
    out["filer_quarter_outlier"] = key.reindex(idx).fillna(False).astype(bool).to_numpy()
    return out


OUTLIER_POSITION_SHARE = 0.25   # one position above this share of the fund's typical total
                                # FV, inside an outlier quarter, is tagged in the wrong units


def rescale_filer_quarter_outliers(df, value_col=FV_RAW + "_normalized", filer_col="cik",
                                   quarter_col="cal_q", share=OUTLIER_POSITION_SHARE,
                                   scale=1_000):
    """
    Row-level repair for filer-quarter outliers. In every case on this data only a few
    positions (2-14) carry FV, cost AND principal 1000x too large together, so the ratio
    checks in normalize_value_scale cannot see them, and dropping the whole quarter also
    drops the correctly scaled rows. Here, inside a flagged quarter, a position larger than
    `share` of the fund's median quarterly total in its other quarters has all three value
    fields divided by `scale`. flag_filer_quarter_outliers() is still run afterwards as a
    fallback, so a quarter that is still an outlier after this repair is dropped.
    """
    out = flag_filer_quarter_outliers(df, value_col, filer_col, quarter_col)
    flagged = out["filer_quarter_outlier"]
    if not flagged.any():
        return out.drop(columns="filer_quarter_outlier")
    tot = out[value_col].abs().groupby([out[filer_col], out[quarter_col]]).sum()
    ref = pd.Series(np.nan, index=out.index)
    for (f, q) in out.loc[flagged, [filer_col, quarter_col]].drop_duplicates().itertuples(index=False):
        others = tot.loc[f].drop(q)
        m = flagged & out[filer_col].eq(f) & out[quarter_col].eq(q)
        ref[m] = others.median()
    big = flagged & (out[value_col].abs() > share * ref)
    cols = [c + "_normalized" for c in VALUE_COLS]
    for c in cols:
        out.loc[big, c] = out.loc[big, c] / scale
        fc = c + "_scale_flag"
        if fc in out:
            out.loc[big, fc] = out.loc[big, fc].astype(str) + f"|filer_quarter_div{scale}"
    return out.drop(columns="filer_quarter_outlier")


_CHILD_SEP = re.compile(r"\s*[-–|,:;(]")


def drop_subtotal_rows(df, value_col=FV_RAW + "_normalized", filing_cols=("cik", "accession"),
                       id_col="investment_identifier", tol=0.05, min_children=20):
    """
    Drop subtotal / heading rows: an identifier that `min_children`+ other identifiers in the
    same filing extend with a separator ("Portfolio Company Debt Securities" -> "Portfolio
    Company Debt Securities- United States ... Torus Inc. ..."), and whose FV equals the sum
    of those rows within `tol`. Left in, they double-count FV.

    Issuer names that head only 2-3 rows ("Vision Solutions, Inc." + "Vision Solutions, Inc.,
    Emerald JV LP") are usually separate holdings (direct vs. JV, different terms) whose FVs
    happen to be close, so they are not treated as subtotals.
    """
    drop = []
    for _, g in df.groupby(list(filing_cols), sort=False):
        ids = g[id_col].dropna().astype(str)
        uniq = sorted(ids.unique())
        for i, x in enumerate(uniq):
            kids = []
            for y in uniq[i + 1:]:                      # sorted: all extensions of x follow it
                if not y.startswith(x):
                    break
                if _CHILD_SEP.match(y, len(x)):
                    kids.append(y)
            if len(kids) < min_children:
                continue
            parent = g.loc[ids.index[ids.eq(x)], value_col].sum()
            child = g.loc[ids.index[ids.isin(kids)], value_col].sum()
            if child and abs(parent / child - 1) <= tol:
                drop.extend(ids.index[ids.eq(x)])
    return df.drop(index=drop)


# ============================================================
# 5. Rate resolution: one rule table, two identities
# ============================================================

def load_base_rates(path: str) -> pd.DataFrame:
    """
    Base-rate file -> DataFrame indexed by cal_q, one column per currency, in decimal.
    Columns after DATE / TIME PERIOD are named by the ISO code they start with
    ('USD Federal Reserve ... SOFR ...' -> USD, 'EUR EURIBOR 3-month ...' -> EUR); see
    augment_rates.py. A plain SOFR.csv yields a USD-only table.
    """
    raw = pd.read_csv(path)
    rates = raw.iloc[:, 2:] / 100
    rates.columns = [c.split()[0].upper() for c in rates.columns]
    return rates.set_axis(raw["TIME PERIOD"].astype(str).str.strip())


@dataclass
class Rule:
    tag: str
    when: Callable[[pd.DataFrame], pd.Series]                  # row mask
    then: Callable[[pd.DataFrame, pd.Series], None]            # in-place assignment on masked rows
    moves_ir: bool = False     # the reported IR value is moved to another field, so the
                               # final IR is derived rather than reported


def _set(**assign):
    """Build a `then` that assigns columns from expressions evaluated *before* any write."""
    def then(d, m):
        vals = {col: (f(d)[m] if callable(f) else f) for col, f in assign.items()}
        for col, v in vals.items():
            d.loc[m, col] = v
    return then


# Shorthands used inside the rule table
isna = lambda d, c: d[c].isna()
has = lambda d, c: d[c].notna()
comp = lambda d: d[PIC].fillna(0) + d[PIK].fillna(0)            # observed PIC + PIK
live = lambda d: ~d["_zero"]                                       # skip unfunded / 0% rows
fixed = lambda d: d["is_fixed"].eq(True).fillna(False).astype(bool)      # untagged -> neither
floating = lambda d: d["is_fixed"].eq(False).fillna(False).astype(bool)
untagged = lambda d: d["is_fixed"].isna()


def est(d: pd.DataFrame) -> pd.Series:
    """All-in floating coupon. A floor below the spread is a base-rate floor (0.5-2%):
    max(base, floor) + spread. A floor at or above it is an all-in floor ("Floor rate 9.85%"):
    max(base + spread, floor). Base-rate floors only bind where the base is near zero (SARON
    from 2025Q3: CHF loans with a 0.75% floor report exactly spread + 0.75%)."""
    s = d["_base"] + d[SPREAD]
    if FLOOR not in d:
        return s
    base_floor = d[FLOOR] < d[SPREAD]
    s = s.where(~(base_floor & (d[FLOOR] > d["_base"])), d[FLOOR] + d[SPREAD])
    return s.where(~(~base_floor & (d[FLOOR] > s)), d[FLOOR])


def spread_like(d: pd.DataFrame) -> pd.Series:
    lo, hi = SPREAD_LIKE_RANGE
    return (d[RATE] > lo) & (d[RATE] <= hi)


RATE_RULES = [
    # ---------- rate-type repairs (need the XBRL rate-type tag) ----------
    # Fixed loan with only a "spread": it is the coupon (no base rate on a fixed loan).
    Rule("fixed_spread_is_rate",
         lambda d: fixed(d) & isna(d, RATE) & has(d, SPREAD) & isna(d, PIC) & isna(d, PIK),
         _set(**{RATE: lambda d: d[SPREAD], SPREAD: np.nan}), moves_ir=True),

    # Floating loan with spread == IR: the value is the spread (median 6.25%), IR = base + spread.
    Rule("rate_is_spread_floating",
         lambda d: floating(d) & live(d) & close(d[RATE], d[SPREAD]) & d["_base"].notna(),
         _set(**{RATE: est, "_est": True}), moves_ir=True),

    # Untagged loan with spread == IR: decide by size, using the split the tagged loans show.
    Rule("rate_is_spread_untagged",
         lambda d: untagged(d) & live(d) & close(d[RATE], d[SPREAD]) & spread_like(d)
                   & d["_base"].notna(),
         _set(**{RATE: est, "_est": True}), moves_ir=True),
    Rule("clear_spread_untagged_coupon",
         lambda d: untagged(d) & live(d) & close(d[RATE], d[SPREAD]) & (d[RATE] > SPREAD_LIKE_RANGE[1]),
         _set(**{SPREAD: np.nan})),

    # ---------- role repairs ----------
    # Spread field actually holds the all-in coupon: Spread == PIC + PIK, IR missing.
    Rule("spread_is_allin_rate",
         lambda d: live(d) & isna(d, RATE) & has(d, SPREAD) & (has(d, PIC) | has(d, PIK))
                   & close(d[SPREAD], comp(d)),
         _set(**{RATE: lambda d: d[SPREAD], SPREAD: np.nan}), moves_ir=True),

    # IR field actually holds the spread: IR < PIC. Keep an existing spread; IR = PIC + PIK.
    # (An IR of 0 next to a real PIC is just empty, not a spread.)
    Rule("rate_is_spread",
         lambda d: live(d) & has(d, RATE) & has(d, PIC) & (d[RATE] < d[PIC] - TOL)
                   & (comp(d) <= RATE_RANGE[1]),
         _set(**{SPREAD: lambda d: d[SPREAD].fillna(d[RATE].where(d[RATE] > 0)), RATE: comp}),
         moves_ir=True),

    # PIC field actually holds the spread: IR - PIC == base rate (no spread, no PIK reported).
    Rule("pic_is_spread",
         lambda d: live(d) & has(d, RATE) & has(d, PIC) & isna(d, SPREAD) & isna(d, PIK)
                   & close(d[RATE] - d[PIC], d["_base"], BASE_TOL),
         _set(**{SPREAD: lambda d: d[PIC], PIC: lambda d: d[RATE]})),

    # IR present, PIC + PIK exceed it, no spread: old IR was the spread
    # ("1M SOFR + 16.00% (0.00% Cash + 20.65% PIK)" tagged IR=16%, PIC=0%, PIK=20.65%).
    Rule("rate_is_spread_components_exceed",
         lambda d: live(d) & has(d, RATE) & has(d, PIC) & has(d, PIK) & isna(d, SPREAD)
                   & (comp(d) > d[RATE] + TOL),
         _set(**{SPREAD: lambda d: d[RATE], RATE: comp}), moves_ir=True),

    # PIK > IR with no PIC: IR holds the cash part ("2% cash + 9% PIK").
    # Without this the downstream PIC = IR - PIK goes negative.
    Rule("rate_is_cash_component",
         lambda d: live(d) & has(d, RATE) & has(d, PIK) & isna(d, PIC) & (d[PIK] > d[RATE] + TOL)
                   & (d[RATE] + d[PIK] <= RATE_RANGE[1]),
         _set(**{PIC: lambda d: d[RATE], RATE: lambda d: d[RATE] + d[PIK]}), moves_ir=True),

    # IR < spread with no components is impossible for a floating loan (base >= 0);
    # the IR field holds a floor or the base rate (median 0.5-1%) -> IR = base + spread.
    Rule("rate_below_spread_replaced",
         lambda d: live(d) & has(d, RATE) & has(d, SPREAD) & isna(d, PIC) & isna(d, PIK)
                   & (d[RATE] < d[SPREAD] - TOL) & d["_base"].notna(),
         _set(**{RATE: est, "_est": True})),

    # ---------- fills: IR missing ----------
    # Spread + PIK where PIK exceeds the spread but is below the base rate: PIK cannot be the
    # all-in coupon, so it is paid on top ("Prime + 1.35%, Floor 9.85%, PIK 2.50%").
    Rule("rate_from_base_plus_spread_plus_pik",
         lambda d: live(d) & isna(d, RATE) & has(d, SPREAD) & has(d, PIK) & isna(d, PIC)
                   & (d[PIK] > d[SPREAD] + TOL) & (d[PIK] < d["_base"] - TOL),
         _set(**{RATE: lambda d: est(d) + d[PIK], PIC: est, "_est": True})),

    # Observed components exceed the spread (or there is no spread) -> IR = PIC + PIK.
    # Observed coupons beat an estimate, so this runs before the SOFR fill.
    Rule("rate_from_components",
         lambda d: live(d) & isna(d, RATE) & (has(d, PIC) | has(d, PIK))
                   & (isna(d, SPREAD) | (comp(d) > d[SPREAD] + TOL))
                   & (comp(d) <= RATE_RANGE[1]),
         _set(**{RATE: comp})),

    # Only spread, or components that are a slice of it (PIC/PIK <= spread) -> IR = base + spread.
    # A PIK slice is then split out by pic_from_rate_minus_pik below.
    Rule("rate_from_base_plus_spread",
         lambda d: live(d) & isna(d, RATE) & has(d, SPREAD) & d["_base"].notna(),
         _set(**{RATE: est, "_est": True})),

    # IR, PIK and spread reported, no PIC: if IR + PIK = base + spread (within BASE_TOL) while IR
    # alone does not, the PIK is carved out of the margin and IR holds only the cash part
    # ("SOFR + 6.00% (3.25% PIK)" tagged IR=6.42%, PIK=3.25%) -> PIC = old IR, IR = IR + PIK.
    Rule("rate_is_cash_part_of_base_plus_spread",
         lambda d: live(d) & has(d, RATE) & has(d, PIK) & has(d, SPREAD) & isna(d, PIC)
                   & d["_base"].notna() & d[PIK].gt(0)
                   & close(d[RATE] + d[PIK], est(d), BASE_TOL) & ~close(d[RATE], est(d), BASE_TOL),
         _set(**{PIC: lambda d: d[RATE], RATE: lambda d: d[RATE] + d[PIK]}), moves_ir=True),

    # ---------- fills: one I1 component missing ----------
    Rule("pic_from_rate_minus_pik",
         lambda d: live(d) & has(d, RATE) & has(d, PIK) & isna(d, PIC) & (d[PIK] <= d[RATE] + TOL),
         _set(**{PIC: lambda d: (d[RATE] - d[PIK]).clip(lower=0)})),

    # ---------- reconcile I1 when all three present ----------
    Rule("pic_from_rate_minus_pik_reconcile",
         lambda d: live(d) & has(d, RATE) & has(d, PIC) & has(d, PIK) & ~close(comp(d), d[RATE])
                   & (d[PIK] <= d[RATE] + TOL),
         _set(**{PIC: lambda d: d[RATE] - d[PIK]})),
]


def classify_rate_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    RateType = member name after '#' in the XBRL enumeration; is_fixed from it (NA if untagged).
    is_prime from the tag or, since many Prime loans are untagged, from "Prime +/- x%" in the
    identifier.
    """
    out = df.copy()
    rt = out[RATE_TYPE_RAW].astype("string").str.split("#").str[-1] if RATE_TYPE_RAW in out \
        else pd.Series(pd.NA, index=out.index, dtype="string")
    out["RateType"] = rt
    out["is_fixed"] = rt.isin(FIXED_MEMBERS).astype("boolean").mask(rt.isna())
    ident = out["investment_identifier"].astype("string") if "investment_identifier" in out \
        else pd.Series(pd.NA, index=out.index, dtype="string")
    out["is_prime"] = (rt.str.contains("Prime", case=False).fillna(False)
                       | ident.str.contains(_PRIME_TEXT, regex=True).fillna(False)).astype(bool)
    return out


def rate_config(df: pd.DataFrame) -> pd.Series:
    """Vectorised check_2: 'spread, rate, pik, pic' style label of present fields."""
    parts = [np.where(df[c].notna(), name, "") for name, c in RATE_FIELDS.items()]
    lab = pd.Series([", ".join(p for p in row if p) for row in zip(*parts)], index=df.index)
    return lab.replace("", "none")


def apply_rules(df: pd.DataFrame, rules=RATE_RULES) -> pd.DataFrame:
    for r in rules:
        m = r.when(df).fillna(False).astype(bool)
        if not m.any():
            continue
        r.then(df, m)
        append_tag(df, m, r.tag)
        if r.moves_ir:
            df.loc[m, "_moved_ir"] = True
    return df


def resolve_rates(df: pd.DataFrame, base_rates, currency_col: str = "currency",
                  drop_empty: bool = True) -> pd.DataFrame:
    """
    base_rates: DataFrame indexed by cal_q (e.g. '2024Q3') with one column per currency, in
    decimal (see load_base_rates).
    Each floating/untagged row uses its own currency's base rate; rows in a currency without
    one, and fixed rows, can still be resolved via I1.
    """
    out = classify_rate_type(df)
    if "change_tracker" not in out:
        out["change_tracker"] = pd.NA
    out["rate_config"] = rate_config(out)
    if drop_empty:
        out = out[out["rate_config"].ne("none")].copy()

    ccy = out[currency_col] if currency_col in out else pd.Series("USD", index=out.index)
    base = pd.Series(np.nan, index=out.index)
    for c in base_rates.columns:
        m = ccy.eq(c)
        base[m] = out.loc[m, "cal_q"].map(base_rates[c])
    # Negative SARON / TONA are real, but loans floor the benchmark at 0% (market standard;
    # no reported CHF coupon passes a negative SARON through), so the loan's base is >= 0.
    base = base.clip(lower=0)
    prime = out["is_prime"] & ccy.eq("USD")            # Prime ~ SOFR + 3.2% is a USD relation
    out["_base"] = base.where(~prime, base + PRIME_OVER_SOFR).where(~fixed(out))
    out["_est"] = False
    out["_moved_ir"] = False
    # Unfunded revolvers / '—%' rows: IR is 0 and no component carries a coupon either.
    out["_zero"] = out[RATE].eq(0) & comp(out).eq(0)
    reported = out[RATE].notna() & ~out["_zero"]

    out = apply_rules(out)

    # Rows whose IR ends at exactly 0 after the rules (IR missing, PIC / PIK reported as 0:
    # non-accrual loans, equity units) carry no coupon either.
    final_zero = out["_zero"] | out[RATE].eq(0)

    # A rule that moved the IR value to another field makes the final IR derived, not reported.
    conds = [final_zero, out["_est"], reported & ~out["_moved_ir"], out[RATE].notna()]
    out["rate_source"] = np.select(
        [c.fillna(False).to_numpy(bool) for c in conds],
        ["zero", "estimated", "reported", "derived"], default="none")
    out["estimate"] = est(out)                     # base + spread with floors, rows with a spread
    # Status columns
    out["check_2"] = out["rate_config"]
    out["check_1"] = np.where(out[RATE].gt(0) & ~final_zero, "resolved", "unresolved")
    return out.drop(columns=["_base", "_zero", "_est", "_moved_ir"])
