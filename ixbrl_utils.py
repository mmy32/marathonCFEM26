# Reusable utilities for IXBLR cleaning pipeline

"""
IXBLR rate and currency cleaning utilities.

This module contains pure, reusable functions for:
- Interest rate normalization
- FX normalization
- Fixed/variable rate inference
- Multi-pass cleaning with audit tags
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple

# =====================
# Constants / metadata
# =====================

TERM_COLS = [
    "InvestmentBasisSpreadVariableRate",
    "InvestmentInterestRate",
    "InvestmentInterestRatePaidInCash",
    "InvestmentInterestRatePaidInKind",
    "InvestmentVariableInterestRateTypeExtensibleEnumeration",
    "InvestmentMaturityDate",
]

AMOUNT_COLS = [
    "InvestmentOwnedAtCost",
    "InvestmentOwnedAtFairValue",
    "InvestmentOwnedBalancePrincipalAmount",
]

SPREAD_RANGE = (0.00, 0.20)
RATE_RANGE = (0.00, 0.50)

CURRENCY_CODES = [
    "USD","EUR","GBP","JPY","CHF","CAD","AUD","NZD","CNY","HKD","SGD",
    "SEK","NOK","DKK","KRW","INR","RUB","BRL","MXN","ZAR","TRY",
    "PLN","CZK","HUF","ILS","SAR","AED","CLP","COP","THB","MYR",
    "IDR","PHP","TWD","ARS","PEN","EGP","NGN","VND","PKR","BDT",
]

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

UNIT_COLS = [
    "InvestmentOwnedAtFairValue-unitRef",
    "InvestmentOwnedAtCost-unitRef",
    "InvestmentOwnedBalancePrincipalAmount-unitRef",
]

VALUE_COLS = [
    "InvestmentOwnedAtFairValue",
    "InvestmentOwnedAtCost",
    "InvestmentOwnedBalancePrincipalAmount",
]

# =====================
# Context / helpers
# =====================

def classify_context(row: pd.Series) -> str:
    has_terms = row[TERM_COLS].notna().any()
    has_amounts = row[AMOUNT_COLS].notna().any()
    if has_terms and not has_amounts:
        return "terms_only"
    elif has_amounts and not has_terms:
        return "amounts_only"
    elif has_terms and has_amounts:
        return "mixed"
    else:
        return "empty"


def _choose_scale(x_abs: float, valid_range: Tuple[float, float]):
    lo, hi = valid_range
    candidates = []
    for scale in (1, 100, 10000):
        y = x_abs / scale
        if lo <= y <= hi:
            candidates.append((scale, y))
    if not candidates:
        return None
    for scale, _ in candidates:
        if scale == 1:
            return 1
    mid = (lo + hi) / 2
    return min(candidates, key=lambda t: abs(t[1] - mid))[0]


def _normalize_series(s: pd.Series, valid_range, kind_name: str) -> pd.DataFrame:
    vals, flags = [], []
    for v in s.values:
        if pd.isna(v):
            vals.append(np.nan); flags.append("na"); continue
        x = float(v); sign = -1 if x < 0 else 1
        x_abs = abs(x)
        lo, hi = valid_range
        if lo <= x_abs <= hi:
            vals.append(x); flags.append("unchanged"); continue
        scale = _choose_scale(x_abs, valid_range)
        if scale is None:
            vals.append(x); flags.append("unresolved")
        else:
            vals.append(sign * (x_abs / scale))
            flags.append(f"div{scale}" if scale != 1 else "unchanged")
    return pd.DataFrame({
        f"{s.name}_normalized": vals,
        f"{s.name}_scale_flag": flags,
    }, index=s.index)


def _append_sign_flag(flags: pd.Series) -> pd.Series:
    def f(v):
        if pd.isna(v) or v == "na":
            return v
        return v if "sign_error" in v else f"{v}|sign_error"
    return flags.map(f)

# =====================
# Rate normalization
# =====================

def normalize_interest_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        ("InvestmentBasisSpreadVariableRate", SPREAD_RANGE),
        ("InvestmentInterestRateFloor", SPREAD_RANGE),
        ("InvestmentInterestRate", RATE_RANGE),
        ("InvestmentInterestRatePaidInCash", RATE_RANGE),
        ("InvestmentInterestRatePaidInKind", RATE_RANGE),
    ]
    out = df.copy()
    for col, rng in cols:
        if col in out.columns:
            new_cols = _normalize_series(out[col], rng, col)
            out = out.drop(columns=new_cols.columns, errors="ignore")
            #out = pd.concat([out, _normalize_series(out[col], rng, col)], axis=1)
            out = pd.concat([out, new_cols], axis=1)
    for base in ["InvestmentInterestRatePaidInCash", "InvestmentInterestRatePaidInKind", "InvestmentInterestRateFloor"]:
        v, f = f"{base}_normalized", f"{base}_scale_flag"
        #print(v, out.columns[out.columns == v])
        if v in out.columns:
            neg = out[v] < 0
            out.loc[neg, v] = out.loc[neg, v].abs()
            if f in out.columns:
                out.loc[neg, f] = _append_sign_flag(out.loc[neg, f])
    
    return out

# =====================
# Currency handling
# =====================

def normalize_currency(x):
    if pd.isna(x):
        return np.nan
    s = str(x).upper()
    for cur in CURRENCY_CODES:
        if re.search(rf"\b{cur}\b", s) or cur in s:
            return cur
    return np.nan


def prepare_fx_data(fx_path: str) -> pd.DataFrame:
    fx = pd.read_csv(fx_path)
    fx["Effective Date"] = pd.to_datetime(fx["Effective Date"])
    fx["cal_q"] = fx["Effective Date"].dt.to_period("Q").astype(str)
    fx[["Country", "CurrencyName"]] = fx["Country - Currency Description"].str.split("-", n=1, expand=True)
    fx["Country"] = fx["Country"].str.strip()
    fx["currency_norm"] = fx["Country"].map(COUNTRY_TO_CURRENCY)
    fx["Exchange Rate"] = pd.to_numeric(fx["Exchange Rate"], errors="coerce")
    fx_q = fx.groupby(["cal_q", "currency_norm"], as_index=False)["Exchange Rate"].mean()
    fx_q = fx_q.rename(columns={"Exchange Rate": "fx_to_usd"})
    usd = fx_q[["cal_q"]].drop_duplicates()
    usd["currency_norm"] = "USD"; usd["fx_to_usd"] = 1.0
    return pd.concat([fx_q, usd], ignore_index=True).drop_duplicates(["cal_q", "currency_norm"])


def convert_currencies(df: pd.DataFrame, fx_path: str) -> pd.DataFrame:
    fx_q = prepare_fx_data(fx_path)
    out = df.copy()
    out["cal_q"] = out["cal_q"].astype(str)
    for u in UNIT_COLS:
        out[u + "_normalized"] = out[u].apply(normalize_currency)
    for v, u in zip(VALUE_COLS, UNIT_COLS):
        u_n = u + "_normalized"; v_usd = v + "_normalized"
        m = out.merge(fx_q.rename(columns={"currency_norm": u_n}), on=[u_n, "cal_q"], how="left")
        out[v_usd] = np.where(m[u_n].eq("USD") | m[u_n].isna(), m[v], m[v] / m["fx_to_usd"])
    return out

# =====================
# Cleaning helpers
# =====================

def add_tag_bulk(df, mask, tag, mark_resolved=False):
    idx = df.index[mask]
    if len(idx) == 0:
        return df
    prev = df.loc[idx, "change_tracker"].astype("string").fillna("")
    df.loc[idx, "change_tracker"] = np.where(prev.eq(""), tag, prev + "|" + tag)
    if "check_1" not in df.columns:
        df["check_1"] = "unresolved"
    if mark_resolved:
        df.loc[idx, "check_1"] = "resolved"
    return df


def initialize_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Initialize dataframe with required columns for cleaning."""
    df_clean = df.copy()
    
    if "change_tracker" not in df_clean.columns:
        df_clean["change_tracker"] = pd.NA
    if "is_fixed" not in df_clean.columns:
        df_clean["is_fixed"] = pd.NA
        
    return df_clean


def classify_rate_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.loc[out['RateType'].isin(['FixedMember','FixedRateMember']), 'is_fixed'] = True
    out.loc[out['RateType'].notna() & ~out['RateType'].isin(['FixedMember','FixedRateMember']), 'is_fixed'] = False
    return out

def perform_initial_swap(df: pd.DataFrame) -> pd.DataFrame:
    """Perform initial spread/rate swap for fixed rate investments."""
    TOL = 1e-6
    spread = 'InvestmentBasisSpreadVariableRate_normalized'
    cash = 'InvestmentInterestRatePaidInCash_normalized'
    pik = 'InvestmentInterestRatePaidInKind_normalized'
    rate = 'InvestmentInterestRate_normalized'
    
    df_clean = df.copy()
    
    mask = (
        df_clean[spread].notna() &
        df_clean[cash].notna() &
        df_clean[pik].notna() &
        df_clean[rate].isna()
    )
    
    # check if cash + pik == spread, then spread is actually rate
    key = np.isclose((df_clean[cash] + df_clean[pik]).to_numpy(), df_clean[spread].to_numpy(), atol=TOL)
    
    # Swap spread <-> rate
    # find rows where mask and key are true, rate = spread, spread = NaN
    tmp = df_clean.loc[mask&key, spread].copy()
    df_clean.loc[mask&key, spread] = df_clean.loc[mask&key, rate]
    df_clean.loc[mask&key, rate] = tmp

    # Update tracker
    # normal fixed rate swap, suspicious swap of variable rate loan, no swap needed
    df_clean.loc[mask&key&(df_clean['is_fixed'].isna()|(df_clean['is_fixed']==True)), "change_tracker"] = "swap_spread_rate_fixed"
    df_clean.loc[mask&key&(df_clean['is_fixed']==False), "change_tracker"] = "swap_spread_rate_fixed_sus"
    df_clean.loc[mask&~key, "change_tracker"] = "checked"
    df_clean.loc[mask&key, "is_fixed"] = True # if spread = cash + PIK, that's a fixed-rate loan
    
    return df_clean

def clean_fixed_rate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and impute fixed rate investment data."""
    TOL = 1e-6
    spread = 'InvestmentBasisSpreadVariableRate_normalized'
    cash = 'InvestmentInterestRatePaidInCash_normalized'
    pik = 'InvestmentInterestRatePaidInKind_normalized'
    rate = 'InvestmentInterestRate_normalized'
    
    df_clean = df.copy()
    fixed_mask = df_clean["RateType"].isin(["FixedMember", "FixedRateMember"])

    # Stage 1: Clear exact duplicate spread == rate
    # When spread=rate, its possible base rate is 0
    # Set spread to NaN
    m_clear_sr = (
        fixed_mask &
        df_clean[spread].notna() & df_clean[rate].notna() &
        np.isclose(df_clean[spread].to_numpy(), df_clean[rate].to_numpy(), atol=TOL)
    )
    if m_clear_sr.any():
        df_clean.loc[m_clear_sr, spread] = np.nan
        df_clean = add_tag_bulk(df_clean, m_clear_sr, "clear_spread_equals_rate", mark_resolved=True)

    # Stage 2: Fill components from rate
    # For fixed-rate loans, rate = cash + pik
    # A) rate & PIK present, cash NaN: find cash = rate - 
    # if rate == PIK also fine (mezzanine debt)
    # if spread == cash, delete spread
    m_A = (
        fixed_mask &
        df_clean[rate].notna() & df_clean[pik].notna() & df_clean[cash].isna() &
        (df_clean[pik] <= df_clean[rate] + TOL)
    )
    if m_A.any():
        new_cash = (df_clean[rate] - df_clean[pik]).clip(lower=0)
        df_clean.loc[m_A, cash] = new_cash[m_A]
        df_clean = add_tag_bulk(df_clean, m_A, "cash_from_rate_minus_pik", mark_resolved=True)

        m_A_clear_spread = m_A & df_clean[spread].notna() & np.isclose(
            df_clean[spread].to_numpy(), df_clean[cash].to_numpy(), atol=TOL
        )
        if m_A_clear_spread.any():
            df_clean.loc[m_A_clear_spread, spread] = np.nan
            df_clean = add_tag_bulk(df_clean, m_A_clear_spread, "clear_spread_equals_cash_after_rate_minus_pik", mark_resolved=True)

    # B) rate & cash present, PIK NaN
    # if rate == cash, then anyways PIK will be 0
    # if not then find PIK; anyways its a fixed rate loan
    m_B = (
        fixed_mask &
        df_clean[rate].notna() & df_clean[cash].notna() & df_clean[pik].isna() &
        (df_clean[cash] <= df_clean[rate] + TOL)
    )
    if m_B.any():
        new_pik = (df_clean[rate] - df_clean[cash]).clip(lower=0)
        df_clean.loc[m_B, pik] = new_pik[m_B]
        df_clean = add_tag_bulk(df_clean, m_B, "pik_from_rate_minus_cash", mark_resolved=True)

    # Stage 3: Fill missing rates; cash and PIK present, but no rate 
    m2_base = (
        fixed_mask &
        df_clean[cash].notna() & df_clean[pik].notna() & df_clean[rate].isna()
    )

    # 3a) Use spread when it matches cash+PIK exactly
    m2a = (
        m2_base &
        df_clean[spread].notna() &
        np.isclose(df_clean[spread].to_numpy(),
                   (df_clean[cash] + df_clean[pik]).to_numpy(), atol=TOL)
    )
    if m2a.any():
        df_clean.loc[m2a, rate] = df_clean.loc[m2a, spread]
        df_clean.loc[m2a, spread] = np.nan
        df_clean = add_tag_bulk(df_clean, m2a, "rate_from_spread_cash_plus_pik_and_clear_spread", mark_resolved=True)

    # 3b) Fill rate when PIK > cash
    m2b = (
        m2_base &
        ~m2a &
        df_clean[pik].notna() &
        df_clean[cash].notna() &
        (df_clean[pik] > df_clean[cash])
    )
    if m2b.any():
        df_clean.loc[m2b, rate] = df_clean.loc[m2b, cash] + df_clean.loc[m2b, pik]
        df_clean = add_tag_bulk(df_clean, m2b, "rate_from_cash_plus_pik_pik_gt_cash", mark_resolved=True)
    
    return df_clean

def perform_additional_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Perform additional data corrections and imputations."""
    spread = 'InvestmentBasisSpreadVariableRate_normalized'
    cash = 'InvestmentInterestRatePaidInCash_normalized'
    pik = 'InvestmentInterestRatePaidInKind_normalized'
    rate = 'InvestmentInterestRate_normalized'
    
    df_clean = df.copy()
    TOL = 1e-6

    # Correction 1: Swap when cash + pik > rate
    # spread = rate, rate = cash + PIK
    m_swap1 = (
        df_clean[cash].notna() & df_clean[pik].notna() & df_clean[rate].notna() &
        df_clean[spread].isna() &
        ((df_clean[cash] + df_clean[pik]) > (df_clean[rate] + TOL))
    )
    if m_swap1.any():
        old_rate = df_clean.loc[m_swap1, rate].copy()
        df_clean.loc[m_swap1, spread] = old_rate
        df_clean.loc[m_swap1, rate] = (df_clean.loc[m_swap1, cash] + df_clean.loc[m_swap1, pik])
        df_clean = add_tag_bulk(df_clean, m_swap1, "set_spread_from_old_rate_and_rate_from_cash_plus_pik", mark_resolved=True)

    # Correction 2: Set rate from cash + pik when both missing
    # rate = cash + PIK
    key = (df_clean[cash] + df_clean[pik] != df_clean[rate])
    m_swap2 = (
        df_clean[cash].notna() & df_clean[pik].notna() &
        df_clean[rate].isna() & df_clean[spread].isna() & key
    )
    if m_swap2.any():
        df_clean.loc[m_swap2, rate] = (df_clean.loc[m_swap2, cash] + df_clean.loc[m_swap2, pik])
        df_clean = add_tag_bulk(df_clean, m_swap2, "set_rate_from_cash_plus_pik_both_missing", mark_resolved=True)

    # Correction 3: Set rate from cash + pik when spread exists but cash+pik > spread
    # rate = cash + PIK
    key = ((df_clean[cash] + df_clean[pik]) > df_clean[spread])
    m_swap3 = (
        df_clean[cash].notna() & df_clean[pik].notna() &
        df_clean[rate].isna() & df_clean[spread].notna() & key
    )
    if m_swap3.any():
        df_clean.loc[m_swap3, rate] = (df_clean.loc[m_swap3, cash] + df_clean.loc[m_swap3, pik])
        df_clean = add_tag_bulk(df_clean, m_swap3, "set_rate_from_cash_plus_pik_spread_there", mark_resolved=True)

    # Correction 4: Set rate from cash when cash > spread
    # rate = cash
    key = ((df_clean[cash]) > df_clean[spread])
    m_swap4 = (
        df_clean[cash].notna() & df_clean[rate].isna() &
        df_clean[spread].notna() & df_clean[pik].isna() & key
    )
    if m_swap4.any():
        df_clean.loc[m_swap4, rate] = df_clean.loc[m_swap4, cash]
        df_clean = add_tag_bulk(df_clean, m_swap4, "set_rate_from_cash", mark_resolved=True)
    
    return df_clean

def check_data(df, idx):
    """Utility function to inspect specific rows."""
    pd.set_option('display.max_colwidth', None)
    print(df.loc[idx,][['cik','period','investment_identifier']])
    return df.loc[idx,]

# spread > IR but no PIC/PIK. then, IR = spread, PIC = IR
def clean_additional_rate_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle edge cases:
    1. spread > rate but no PIC/PIK
    Adds appropriate tags in 'change_tracker'.
    """
    df_clean = df.copy()
    
    spread = 'InvestmentBasisSpreadVariableRate_normalized'
    rate = 'InvestmentInterestRate_normalized'
    pik = 'InvestmentInterestRatePaidInKind_normalized'
    cash = 'InvestmentInterestRatePaidInCash_normalized'
    pic = cash  # assuming PIC corresponds to cash component

    # Case 1: spread > rate but no PIC/PIK
    mask1 = (
        df_clean[spread].notna() & df_clean[rate].notna() &
        ((df_clean[pik].isna()) & (df_clean[pic].isna())) &
        (df_clean[spread] > df_clean[rate])
    )
    if mask1.any():
        # Optionally, swap spread -> rate, or just tag
        df_clean.loc[mask1, rate] = df_clean.loc[mask1, spread]
        df_clean.loc[mask1, spread] = np.nan
        df_clean = add_tag_bulk(df_clean, mask1, "spread_gt_rate_no_pik_pic", mark_resolved=True)

    return df_clean

def fix_pic_from_ir_minus_pik(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix rows where PIC + PIK != IR by using PIK as truth and recomputing:
        PIC = IR - PIK
    Adds tag: 'pic_from_ir_minus_pik'
    """
    df_clean = df.copy()
    
    rate = 'InvestmentInterestRate_normalized'
    cash = 'InvestmentInterestRatePaidInCash_normalized'   # PIC
    pik = 'InvestmentInterestRatePaidInKind_normalized'
    TOL = 1e-6
    
    mask_base = (
        df_clean[rate].notna() &
        df_clean[pik].notna() &
        df_clean[cash].notna()
    )
    
    mask_bad = (
        mask_base &
        ((df_clean[cash] + df_clean[pik]) < df_clean[rate]) & ## NEW CONDITION
        (~np.isclose(
            (df_clean[cash] + df_clean[pik]).to_numpy(),
            df_clean[rate].to_numpy(),
            atol=TOL
        ))
    )
    
    if mask_bad.any():
        new_pic = (df_clean[rate] - df_clean[pik])

        df_clean.loc[mask_bad, cash] = new_pic[mask_bad]
        df_clean = add_tag_bulk(df_clean, mask_bad, "pic_from_ir_minus_pik", mark_resolved=True)
    
    return df_clean

# # IR < PIC -> spread = rate, rate = PIC, ignore PIK
def fix_ir_lt_pic(df: pd.DataFrame) -> pd.DataFrame:
    """Fix cases where IR < PIC by treating IR as mis-filled.
       Rule: spread = old_rate, rate = PIC, ignore PIK."""
    
    spread = 'InvestmentBasisSpreadVariableRate_normalized'
    rate = 'InvestmentInterestRate_normalized'
    pic = 'InvestmentInterestRatePaidInCash_normalized'
    pik = 'InvestmentInterestRatePaidInKind_normalized'
    
    df_clean = df.copy()
    
    # Mask: IR and PIC exist, PIK may or may not exist
    mask = (
        df_clean[rate].notna() &
        df_clean[pic].notna() &
        (df_clean[rate] < df_clean[pic])
    )
    
    if mask.any():
        # Save old rate
        old_rate = df_clean.loc[mask, rate].copy()
        
        # Apply corrections
        df_clean.loc[mask, spread] = old_rate
        df_clean.loc[mask, rate] = df_clean.loc[mask, pic]
        
        # Tag
        df_clean = add_tag_bulk(df_clean, mask, "fix_ir_lt_pic", mark_resolved=True)
        
    return df_clean

####################################################################################
def percentile_range(values, pct):
    vals = pd.Series(values).dropna()
    lower = vals.quantile(pct)
    upper = vals.quantile(1 - pct)
    return lower, upper


def in_range(series, range_):
    return series.between(range_[0], range_[1], inclusive="both")

SPREAD = 'InvestmentBasisSpreadVariableRate_normalized'
RATE = 'InvestmentInterestRate_normalized'

def apply_spread_rate_rules(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    spread = 'InvestmentBasisSpreadVariableRate_normalized'
    rate   = 'InvestmentInterestRate_normalized'
    pic    = 'InvestmentInterestRatePaidInCash_normalized'
    pik    = 'InvestmentInterestRatePaidInKind_normalized'

    for col in [
        'spread_rate_issue',
        'pik_given_pic_equal_rate',
        'ir_equals_pik_tag',
        'pik_gt_ir_pic_not_equal_rate'
    ]:
        if col not in df_clean.columns:
            df_clean[col] = False

    # Rule 1
    mask = df_clean[rate].notna() & df_clean[spread].isna()
    df_clean.loc[mask, pic] = df_clean.loc[mask, rate]

    # Rule 2
    mask = (
        df_clean[spread].notna() &
        df_clean[rate].isna() &
        (df_clean[pic].notna() | df_clean[pik].notna())
    )
    df_clean.loc[mask, 'spread_rate_issue'] = True

    # Rule 4
    mask = (
        df_clean[rate].notna() &
        df_clean[spread].notna() &
        (df_clean[rate] < df_clean[spread])
    )
    df_clean.loc[mask, 'spread_rate_issue'] = True

    # Rule 5
    mask = (
        df_clean[pik].notna() &
        df_clean[rate].notna() &
        df_clean[pic].notna() &
        (df_clean[pik] > df_clean[rate]) &
        (df_clean[pic] == df_clean[rate])
    )
    df_clean.loc[mask, 'pik_given_pic_equal_rate'] = True

    # Rule 6
    TOL = 1e-6
    mask = (
        df_clean[rate].notna() &
        df_clean[pik].notna() &
        (np.abs(df_clean[rate] - df_clean[pik]) <= TOL)
    )
    df_clean.loc[mask, 'ir_equals_pik_tag'] = True

    # Rule 7
    mask = (
        df_clean[pik].notna() &
        df_clean[rate].notna() &
        (df_clean[pik] > df_clean[rate]) &
        (df_clean[pic] != df_clean[rate])
    )
    df_clean.loc[mask, 'pik_gt_ir_pic_not_equal_rate'] = True

    return df_clean

# dropping rows that are not debt investments
def drop_fully_missing_rate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where ALL rate-related fields are missing.
    These cases are not debt instruments or have unusable data.

    Returns:
        df_cleaned  = dataframe with rows dropped
    """
    spread = 'InvestmentBasisSpreadVariableRate_normalized'
    rate = 'InvestmentInterestRate_normalized'
    cash = 'InvestmentInterestRatePaidInCash_normalized'
    pik = 'InvestmentInterestRatePaidInKind_normalized'

    mask_all_missing = (
        df[spread].isna() &
        df[rate].isna() &
        df[cash].isna() &
        df[pik].isna()
    )

    # rows to drop
    df_dropped = df.loc[mask_all_missing].copy()

    # add tag
    if mask_all_missing.any():
        df_dropped['drop_reason'] = "all_rate_fields_missing"
    
    # keep only the rows NOT in mask
    df_cleaned = df.loc[~mask_all_missing].copy()

    return df_cleaned


def add_check_2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column 'check_2' that lists which of the four rate-related columns
    are present for each row. Non-NaN values are included in the string.
    
    Order: spread, rate, pik, pic
    """
    df_clean = df.copy()
    
    cols = [
        ('InvestmentBasisSpreadVariableRate_normalized', 'spread'),
        ('InvestmentInterestRate_normalized', 'rate'),
        ('InvestmentInterestRatePaidInKind_normalized', 'pik'),
        ('InvestmentInterestRatePaidInCash_normalized', 'pic')
    ]
    
    def present_cols(row):
        present = [alias for col, alias in cols if pd.notna(row[col])]
        return ", ".join(present) if present else pd.NA
    
    df_clean['check_2'] = df_clean.apply(present_cols, axis=1)
    
    return df_clean

# rate, pic case
# if rate == pic, resolved
# else, have to be fixed
def unresolved_rate_pic(df):
    """
    For rows where check_2 == 'rate, pic':
      1. Count matches and mismatches
      2. Mark rows where pic == rate as resolved (check_1 = 'resolved')
      3. Return the FULL updated dataframe
      4. Also returns df_mismatch separately if needed
    """
    df = df.copy()  # so we don't modify external df accidentally
    
    rate = 'InvestmentInterestRate_normalized'
    pic = 'InvestmentInterestRatePaidInCash_normalized'
    
    # Rows with exactly these two present
    mask_rate_pic = df['check_2'] == 'rate, pic'
    df_subset = df.loc[mask_rate_pic, [rate, pic, 'check_2', 'check_1']]
    
    total = len(df_subset)
    if total == 0:
        #print("No rows with check_2 == 'rate, pic'")
        return df
    
    # Compare rate vs PIC
    mask_equal = df_subset[pic] == df_subset[rate]
    
    count_equal = mask_equal.sum()
    count_unequal = total - count_equal
    
    # print(f"Total rows with rate + pic: {total}")
    # print(f"Rows where pic == rate:     {count_equal}")
    # print(f"Rows where pic != rate:     {count_unequal}")
    
    # ---------------------------------------------
    # ✔ Mark equal cases as resolved
    # ---------------------------------------------
    df.loc[mask_rate_pic & mask_equal, 'check_1'] = 'resolved'
    
    # Mismatched rows (optional output)
    df_mismatch = df.loc[mask_rate_pic & (~mask_equal), :]
    
    # Return full cleaned df AND mismatches
    return df


# rate, pik, pic given
def unresolved_rate_pik_pic(df):
    """
    For rows where check_2 == 'rate, pik, pic':
      1. If pik + pic == rate, mark resolved.
      2. If pik + pic != rate:
            set pic = rate - pik
            mark resolved.
      Returns full df + mismatch rows before fixing.
    """
    df = df.copy()
    
    rate = 'InvestmentInterestRate_normalized'
    pic  = 'InvestmentInterestRatePaidInCash_normalized'
    pik  = 'InvestmentInterestRatePaidInKind_normalized'
    
    # Filter matching rows
    mask_group = df['check_2'] == 'rate, pik, pic'
    df_subset = df.loc[mask_group, [rate, pic, pik, 'check_2', 'check_1']]
    
    total = len(df_subset)
    if total == 0:
        #print("No rows with check_2 == 'rate, pik, pic'")
        return df
    
    # Compute sum
    sum_pik_pic = df_subset[pik] + df_subset[pic]
    
    # Cases where sum matches
    mask_equal = sum_pik_pic == df_subset[rate]
    
    count_equal = mask_equal.sum()
    count_unequal = total - count_equal
    
    # print(f"Total 'rate, pik, pic' rows:   {total}")
    # print(f"pik + pic == rate:            {count_equal}")
    # print(f"pik + pic != rate (fixed):    {count_unequal}")
    
    # ---------------------------------------------
    # 1Equal → mark resolved
    # ---------------------------------------------
    df.loc[mask_group & mask_equal, 'check_1'] = 'resolved'
    
    # ---------------------------------------------
    # 2Not equal → fix pic = rate - pik
    # ---------------------------------------------
    mask_fix = mask_group & (~mask_equal)
    df.loc[mask_fix, pic] = df.loc[mask_fix, rate] - df.loc[mask_fix, pik]
    
    # Mark fixed rows as resolved
    df.loc[mask_fix, 'check_1'] = 'resolved'
    
    # Mismatched rows BEFORE correction
    df_mismatch_before = df_subset[~mask_equal]
    
    return df

def unresolved_pik(df):
    """
    For rows where check_2 == 'pik':
      - Do NOT change PIC
      - Simply mark them as resolved
      - Return updated df + affected rows
    """
    df = df.copy()
    
    mask_pik_only = df['check_2'] == 'pik'
    
    df_subset = df.loc[mask_pik_only, [
        'InvestmentInterestRate_normalized',
        'InvestmentInterestRatePaidInCash_normalized',
        'InvestmentInterestRatePaidInKind_normalized',
        'check_2', 'check_1'
    ]]
    
    total = len(df_subset)
    #print(f"Total 'pik' only rows: {total}")
    
    if total == 0:
        return df
    
    # ---------------------------------------------------
    # Mark only-PIL rows as resolved
    # ---------------------------------------------------
    df.loc[mask_pik_only, 'check_1'] = 'resolved'
    
    df_fixed = df.loc[mask_pik_only]
    
    return df

def unresolved_spread_rate_pic_pik(df):
    # Work on a copy
    df = df.copy()

    # mask for rows that have all four
    mask = df["check_2"] == "spread, rate, pik, pic"

    # pull relevant columns
    rate = df.loc[mask, "InvestmentInterestRate_normalized"]
    pic = df.loc[mask, "InvestmentInterestRatePaidInCash_normalized"]
    pik = df.loc[mask, "InvestmentInterestRatePaidInKind_normalized"]

    # condition: already matches
    ok_condition = (pic + pik).round(8) == rate.round(8)
    n_total = mask.sum()
    n_ok = ok_condition.sum()
    n_adjust = n_total - n_ok

    # 1. rows that match → resolved
    df.loc[mask & ok_condition, "resolved"] = True

    # 2. rows that don't match → adjust pic = rate - pik
    df.loc[mask & ~ok_condition, "InvestmentInterestRatePaidInCash_normalized"] = (
        rate - pik
    )

    # mark resolved
    df.loc[mask & ~ok_condition, "resolved"] = True

    return df


def unresolved_pic(df):
    """
    For rows where check_2 == 'pic':
      - Do NOT change PIK
      - Simply mark them as resolved
      - Return updated df + affected rows
    """
    df = df.copy()
    
    mask_pic_only = df['check_2'] == 'pic'
    
    df_subset = df.loc[mask_pic_only, [
        'InvestmentInterestRate_normalized',
        'InvestmentInterestRatePaidInCash_normalized',
        'InvestmentInterestRatePaidInKind_normalized',
        'check_2', 'check_1'
    ]]
    
    total = len(df_subset)
    #print(f"Total 'pic' only rows: {total}")
    
    if total == 0:
        return df
    
    # ---------------------------------------------------
    # Mark only-PIL rows as resolved
    # ---------------------------------------------------
    df.loc[mask_pic_only, 'check_1'] = 'resolved'
    
    df_fixed = df.loc[mask_pic_only]
    
    return df

def unresolved_spread_pik_pic(df):
    df = df.copy()

    mask = df["check_2"] == "spread, pik, pic"

    pik = df.loc[mask, "InvestmentInterestRatePaidInKind_normalized"]
    pic = df.loc[mask, "InvestmentInterestRatePaidInCash_normalized"]

    # Condition: pik > pic
    mask_pik_gt_pic = pik > pic

    # Print counts
    #print("Total rows with (spread, pik, pic):", mask.sum())
    #print("Rows where PIK > PIC:", mask_pik_gt_pic.sum())

    # Mark resolved
    df.loc[mask, "resolved"] = True

    return df

def unresolved_spread_rate_pik(df, sample_n=10):
    df = df.copy()

    rate_col = "InvestmentInterestRate_normalized"
    pik_col  = "InvestmentInterestRatePaidInKind_normalized"

    # Filter relevant rows
    mask = df["check_2"] == "spread, rate, pik"
    df_sub = df.loc[mask, ['cik', 'accession','investment_identifier', rate_col, pik_col]]

    if df_sub.empty:
        #print("No rows with check_2 = 'spread, rate, pik'")
        return df

    # Conditions
    mask_equal = df_sub[pik_col] == df_sub[rate_col]
    mask_gt    = df_sub[pik_col] >  df_sub[rate_col]
    mask_lt    = df_sub[pik_col] <  df_sub[rate_col]

    # Print stats
    # print("Total rows with (spread, rate, pik):", mask.sum())
    # print("PIK == RATE:", mask_equal.sum())

    # print("PIK > RATE :", mask_gt.sum())
    # print("PIK < RATE :", mask_lt.sum())
    

    # Mark resolved where pik == rate
    df.loc[mask & mask_equal, "resolved"] = True

    # Print sample rows
    # print("\nSample rows where PIK > RATE:")
    # print(df_sub[mask_gt].head(sample_n))
    
    # print("\nSample rows where PIK < RATE:")
    # print(df_sub[mask_lt].head(sample_n))

    return df

def unresolved_spread_rate_pic(df):
    df = df.copy()
    
    rate_col = "InvestmentInterestRate_normalized"
    pic_col  = "InvestmentInterestRatePaidInCash_normalized"

    # Filter relevant rows
    mask = df["check_2"] == "spread, rate, pic"
    df_sub = df.loc[mask, [rate_col, pic_col, "check_2", "change_tracker"]]

    if df_sub.empty:
        #print("No rows with check_2 = 'spread, rate, pic'")
        return df

    # Condition where PIC == RATE
    mask_equal = df_sub[pic_col] == df_sub[rate_col]

    # Print counts
    # print("Total rows with (spread, rate, pic):", mask.sum())
    # print("PIC == RATE:", mask_equal.sum())
    # print("PIC != RATE (will be dropped):", (~mask_equal).sum())

    # Mark resolved
    df.loc[mask & mask_equal, "check_1"] = "resolved"

    # Drop rows where PIC != RATE
    df = df.drop(df.index[mask & ~mask_equal])

    # print("\nSample resolved rows (PIC == RATE):")
    # print(df.loc[mask & mask_equal].head(5))

    return df


UNIT_COLS = [
    "InvestmentOwnedAtFairValue-unitRef",
    "InvestmentOwnedAtCost-unitRef", 
    "InvestmentOwnedBalancePrincipalAmount-unitRef",
]

# Columns that store numeric values tied to the unit columns above
VALUE_COLS = [
    "InvestmentOwnedAtFairValue",
    "InvestmentOwnedAtCost",
    "InvestmentOwnedBalancePrincipalAmount",
]

# Columns representing terms, rates, or descriptive fields tied to investment conditions
TERM_COLS = [
    "InvestmentBasisSpreadVariableRate",
    'InvestmentInterestRate',
    "InvestmentInterestRatePaidInCash",
    "InvestmentInterestRatePaidInKind",
    'InvestmentVariableInterestRateTypeExtensibleEnumeration',
    "InvestmentMaturityDate",
]

# Columns representing monetary amounts tied to valuation or principal
AMOUNT_COLS = [
    "InvestmentOwnedAtCost",
    "InvestmentOwnedAtFairValue",
    "InvestmentOwnedBalancePrincipalAmount",
]
UNIT_NORMALIZED_COLS = [col + "_normalized" for col in UNIT_COLS]
TERM_NORMALIZED_COLS = [col + "_normalized" for col in TERM_COLS]
SPREAD = "InvestmentBasisSpreadVariableRate_normalized"
RATE = 'InvestmentInterestRate_normalized'
CASH = "InvestmentInterestRatePaidInCash_normalized"
PIK = "InvestmentInterestRatePaidInKind_normalized"

def determine_currency(row):
    """
    Determines the currency for each row by comparing three normalized currency fields,
    returning the common value if they match, 'USD' if all are missing, or, in case of
    conflicts, choosing the currency in the priority order:
    PrincipalAmount > FairValue > Cost.
    
    NOTE: if missing, we assume it's USD
    """
    UNIT_NORMALIZED_COLS = [col + "_normalized" for col in UNIT_COLS]
    values = [v for v in row[UNIT_NORMALIZED_COLS] if pd.notna(v)]

    # all missing
    if len(values) == 0:
        return 'USD'

    unique_vals = set(values)

    # concistent currency information
    if len(unique_vals) == 1:
        return values[0]

    # inconsistent currency info：Principal > FairValue > Cost
    principal_col = "InvestmentOwnedBalancePrincipalAmount-unitRef_normalized"
    fair_col = "InvestmentOwnedAtFairValue-unitRef_normalized"
    cost_col = "InvestmentOwnedAtCost-unitRef_normalized"

    for col in [principal_col, fair_col, cost_col]:
        val = row.get(col)
        if pd.notna(val):
            return val

    return 'USD'

def add_estimate_with_sofr(
    df: pd.DataFrame,
    sofr_df: pd.DataFrame,
    rate_range: Tuple[float, float],
    spread_range: Tuple[float, float],
    currency_col: str = "currency",
    spread_col: str = SPREAD,
    rate_col: str = RATE,
    quarter_col: str = "cal_q",   # e.g. "2023Q1"
    sofr_q_col: str = "TIME PERIOD",    # in sofr_df, e.g. "2023Q1"
    sofr_value_col: str = "sofr"
) -> pd.DataFrame:

    out = df.copy()

    sofr_trim = (
        sofr_df[[sofr_q_col, sofr_value_col]]
        .drop_duplicates(subset=[sofr_q_col])
    )

    out = out.merge(
        sofr_trim.rename(columns={sofr_q_col: quarter_col}),
        on=quarter_col,
        how="left"
    )

    out["estimate"] = pd.NA
    out["check_3"] = pd.NA

    usd_mask = out[currency_col] == "USD"
    spread_notna = out[spread_col].notna()
    rate_notna = out[rate_col].notna()
    sofr_notna = out[sofr_value_col].notna()

    spread_in_spread_range = out[spread_col].between(spread_range[0], spread_range[1])
    rate_in_rate_range    = out[rate_col].between(rate_range[0], rate_range[1])
    rate_in_spread_range  = out[rate_col].between(spread_range[0], spread_range[1])

    # ---------------- Case 1 ----------------
    # 1a) spread notna, spread has reasonable range -> spread + base
    cond1_ok = usd_mask & spread_notna & sofr_notna & spread_in_spread_range
    out.loc[cond1_ok, "estimate"] = (
        out.loc[cond1_ok, spread_col] + out.loc[cond1_ok, sofr_value_col]
    )
    out.loc[cond1_ok, "check_3"] = "resolved"

    # 1b base: spread notna, BUT spread is out of spread_range
    cond1_susp_base = usd_mask & spread_notna & sofr_notna & (~spread_in_spread_range)

    # estimate is always spread + base in 1b
    out.loc[cond1_susp_base, "estimate"] = (
        out.loc[cond1_susp_base, spread_col] + out.loc[cond1_susp_base, sofr_value_col]
    )

    # 1b-1) spread out-of-range, BUT rate is in spread_range -> "sus_rate_better"
    cond1_susp_rate_better = cond1_susp_base & rate_notna & rate_in_spread_range
    out.loc[cond1_susp_rate_better, "check_3"] = "sus_rate_better"

    # 1b-2) spread out-of-range AND (rate not in spread_range OR rate is NA) -> "sus_spread"
    cond1_susp_spread = cond1_susp_base & (~(rate_notna & rate_in_spread_range))
    out.loc[cond1_susp_spread, "check_3"] = "sus_spread"

    # ---------------- Case 2 ----------------
    # spread isna, rate notna, rate in spread_range -> rate + base
    cond2 = usd_mask & (~spread_notna) & rate_notna & rate_in_spread_range & sofr_notna
    out.loc[cond2, "estimate"] = (
        out.loc[cond2, rate_col] + out.loc[cond2, sofr_value_col]
    )
    out.loc[cond2, "check_3"] = "sus_rate_is_spread"

    # ---------------- Case 3 ----------------
    # spread isna, rate notna, rate in rate_range -> rate
    cond3 = usd_mask & (~spread_notna) & rate_notna & rate_in_rate_range
    out.loc[cond3, "estimate"] = out.loc[cond3, rate_col]
    out.loc[cond3, "check_3"] = "can_use_rate"

    return out


def unresolved_spread(df):
    """
    For rows where only spread is present (check_2 == 'spread'),
    set rate = estimate and mark as resolved.
    """
    df = df.copy()

    spread_mask = df["check_2"] == "spread"

    rate_col = "InvestmentInterestRate_normalized"
    est_col  = "estimate"

    if est_col not in df.columns:
        raise ValueError(f"Column '{est_col}' not found in df.")

    #print("Total rows with only spread:", spread_mask.sum())

    # Set rate = estimate
    df.loc[spread_mask, rate_col] = df.loc[spread_mask, est_col]

    # Mark resolved
    df.loc[spread_mask, "check_1"] = "resolved"

    # Show sample
    # print("\nSample updated rows:")
    # print(df.loc[spread_mask, ["check_2", rate_col, est_col]].head(5))

    return df


def unresolved_spread_rate(df, tol=0.04):
    """
    For rows where check_2 == 'spread, rate':
    - If |rate - estimate| <= tol → treat as equal
    - If outside tolerance → still keep the row and mark resolved
    - No dropping
    - rate is NOT modified
    """
    df = df.copy()

    rate_col = "InvestmentInterestRate_normalized"
    est_col  = "estimate"

    mask = df["check_2"] == "spread, rate"
    df_sub = df.loc[mask]

    #print("Total rows with spread, rate:", len(df_sub))

    # Absolute difference
    diff = (df_sub[rate_col] - df_sub[est_col]).abs()

    mask_equal_tol = diff <= tol
    mask_outside_tol = diff > tol

    # print("Rows within 4% absolute tolerance:", mask_equal_tol.sum())
    # print("Rows outside 4% absolute tolerance:", mask_outside_tol.sum())

    # Show mismatches
    df_mismatch = df_sub.loc[mask_outside_tol, [rate_col, est_col, "check_2"]]
    # print("\n--- Mismatched rows (outside 4% tolerance) — up to 10 shown ---")
    # print(df_mismatch.head(10))

    # Mark ALL as resolved
    df.loc[mask, "check_1"] = "resolved"

    return df


def unresolved_spread_pik(df):
    """
    Case: check_2 == 'spread, pik'
    - Count pik == estimate, pik > estimate, pik < estimate
    - If pik < estimate: set PIC = estimate - pik
    - If pik > estimate: leave as is
    - Mark ALL 'spread, pik' cases as resolved
    """
    df = df.copy()

    pik_col = "InvestmentInterestRatePaidInKind_normalized"
    pic_col = "InvestmentInterestRatePaidInCash_normalized"
    est_col = "estimate"

    mask = df["check_2"] == "spread, pik"
    df_sub = df.loc[mask]

    #print("Total rows (spread, pik):", len(df_sub))

    # First check for NA values
    # print("Rows with NA in pik_col:", df_sub[pik_col].isna().sum())
    # print("Rows with NA in est_col:", df_sub[est_col].isna().sum())
    
    # Create masks for valid comparisons (both columns not NA)
    valid_mask = df_sub[pik_col].notna() & df_sub[est_col].notna()
    df_valid = df_sub[valid_mask]
    
    #print("Rows with valid non-NA values for comparison:", len(df_valid))

    # Comparisons only on non-NA values
    mask_eq = df_valid[pik_col] == df_valid[est_col]
    mask_gt = df_valid[pik_col] > df_valid[est_col]
    mask_lt = df_valid[pik_col] < df_valid[est_col]

    # print("pik == estimate:", mask_eq.sum())
    # print("pik >  estimate:", mask_gt.sum())
    # print("pik <  estimate:", mask_lt.sum())

    # -----------------------------
    # Fix pik < estimate: create PIC = estimate - pik
    # -----------------------------
    # Get indices where pik < estimate (safely handling NA)
    to_fix_indices = []
    adjusted_count = 0
    
    # Iterate through ALL 'spread, pik' rows
    for idx in df_sub.index:
        pik_val = df.at[idx, pik_col]
        est_val = df.at[idx, est_col]
        
        # Only process if both values are available and pik < estimate
        if pd.notna(pik_val) and pd.notna(est_val) and pik_val < est_val:
            to_fix_indices.append(idx)
            df.at[idx, pic_col] = est_val - pik_val
            adjusted_count += 1

    # Mark ALL 'spread, pik' rows as resolved
    df.loc[mask, "check_1"] = "resolved"
    #(f"\nMarked ALL {len(df_sub)} 'spread, pik' rows as resolved")

    # Show a preview of adjustments
    #if adjusted_count > 0:
        #print(f"\nAdjusted {adjusted_count} rows where pik < estimate:")
        #print("\nSample adjusted rows:")
        #sample_indices = to_fix_indices[:min(10, len(to_fix_indices))]
        #print(df.loc[sample_indices, [pik_col, est_col, pic_col]])
    #else:
        #print("\nNo rows needed adjustment (pik < estimate)")

    # Also show what happens with rows where pik > estimate or pik == estimate
    #if mask_gt.sum() > 0:
        #print(f"\n{pik_col} > {est_col} for {mask_gt.sum()} rows - left as is")
    #if mask_eq.sum() > 0:
        #print(f"\n{pik_col} == {est_col} for {mask_eq.sum()} rows - no adjustment needed")

    return df

def unresolved_spread_pic(df):
    """
    Case: check_2 == 'spread, pic'
    
    - Print counts of pic == estimate, pic > estimate, pic < estimate
    - Do NOT modify pic or estimate
    - Mark all rows as resolved
    """
    df = df.copy()

    pic_col = "InvestmentInterestRatePaidInCash_normalized"
    est_col = "estimate"

    mask = df["check_2"] == "spread, pic"
    df_sub = df.loc[mask]

    #print("Total rows with spread, pic:", len(df_sub))

    if len(df_sub) == 0:
        return df

    mask_eq = df_sub[pic_col] == df_sub[est_col]
    mask_gt = df_sub[pic_col] > df_sub[est_col]
    mask_lt = df_sub[pic_col] < df_sub[est_col]

    # print("pic == estimate:", mask_eq.sum())
    # print("pic >  estimate:", mask_gt.sum())
    # print("pic <  estimate:", mask_lt.sum())

    # Mark all as resolved
    df.loc[mask, "check_1"] = "resolved"

    return df

def unresolved2_spread_rate_pik(df):
    df = df.copy()

    rate = "InvestmentInterestRate_normalized"
    pik  = "InvestmentInterestRatePaidInKind_normalized"
    pic  = "InvestmentInterestRatePaidInCash_normalized"
    est  = "estimate"

    mask = df["check_2"] == "spread, rate, pik"
    df_sub = df.loc[mask]

    #print("Total rows: ", len(df_sub))

    # Main comparisons
    mask_eq = df_sub[pik] == df_sub[rate]
    mask_lt = df_sub[pik] < df_sub[rate]
    mask_gt = df_sub[pik] > df_sub[rate]

    # print("pik == rate:", mask_eq.sum())
    # print("pik < rate:", mask_lt.sum())
    # print("pik > rate:", mask_gt.sum())

    # -----------------------------
    # Case 1: pik == rate
    # -----------------------------
    df.loc[mask & mask_eq, "check_1"] = "resolved"

    # -----------------------------
    # Case 2: pik < rate → compute PIC
    # -----------------------------
    df.loc[mask & mask_lt, pic] = (
        df.loc[mask & mask_lt, rate] - df.loc[mask & mask_lt, pik]
    )
    df.loc[mask & mask_lt, "check_1"] = "resolved"

    # -----------------------------
    # Case 3: pik > rate → leave unchanged
    # -----------------------------
    df_gt = df_sub[mask_gt]

    # Among pik > rate, check pik > estimate
    mask_gt_est = df_gt[pik] > df_gt[est]

    # print("\nAmong pik > rate:")
    # print("pik > estimate:", mask_gt_est.sum())
    # print("pik <= estimate:", (~mask_gt_est).sum())

    # mark resolved but do NOT modify
    df.loc[mask & mask_gt, "check_1"] = "resolved"

    return df

