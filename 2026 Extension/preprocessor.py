from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Robust import: prefer src.utils, fallback to utils
try:
    from helpers import add_calendar_quarter_columns
except Exception:  # noqa
    from helpers import add_calendar_quarter_columns  # type: ignore

from paths import IXBRL_WIDE_ALL_CSV, PROCESSED_DIR

logger = logging.getLogger("preprocessor")


# ----------------------------
# Columns / Ranges
# ----------------------------

TERM_COLS = [
    "InvestmentBasisSpreadVariableRate",
    "InvestmentInterestRate",
    "InvestmentInterestRatePaidInCash",
    "InvestmentInterestRatePaidInKind",
    "InvestmentInterestRateFloor",
    "InvestmentVariableInterestRateTypeExtensibleEnumeration",
    "InvestmentMaturityDate",
]

AMOUNT_COLS = [
    "InvestmentOwnedAtCost",
    "InvestmentOwnedAtFairValue",
    "InvestmentOwnedBalancePrincipalAmount",
]

SPREAD_RANGE = (0.00, 0.20)  # 0% ~ 20%
RATE_RANGE = (0.02, 0.50)    # 2% ~ 50%


# ----------------------------
# Config
# ----------------------------

@dataclass
class PreprocessConfig:
    quarter_from: str = "2023Q1"
    quarter_to: str = "2025Q3"
    keep_share_col: str = "InvestmentOwnedBalanceShares"

    # Filters
    only_no_shares: bool = True
    drop_amounts_only: bool = True

    # Diagnostics
    write_stats_csv: bool = False


# ----------------------------
# Helpers
# ----------------------------

def prefix_merge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge columns that differ only by namespace prefix:
      us-gaap:Foo, dei:Foo, Foo  -> Foo (combine_first)
    """
    original_cols = df.columns.tolist()
    cleaned_cols = [c.split(":", 1)[-1] if ":" in c else c for c in original_cols]

    prefix_map: dict[str, list[str]] = defaultdict(list)
    for orig, cleaned in zip(original_cols, cleaned_cols):
        prefix_map[cleaned].append(orig)

    merged_dict = {}
    for clean_name, col_group in prefix_map.items():
        if len(col_group) == 1:
            merged_dict[clean_name] = df[col_group[0]]
        else:
            merged = df[col_group[0]].copy()
            for col in col_group[1:]:
                merged = merged.combine_first(df[col])
            merged_dict[clean_name] = merged

    return pd.concat(merged_dict, axis=1)


def nonnull_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if not c.endswith("-unitRef")]
    out = (
        df[cols].notna().sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "non_null_count"})
    )
    out["percentage"] = (out["non_null_count"] / len(df) * 100.0) if len(df) else 0.0
    return out.sort_values("non_null_count", ascending=False).reset_index(drop=True)


def classify_context(df: pd.DataFrame) -> pd.Series:
    has_terms = df.reindex(columns=TERM_COLS, fill_value=np.nan).notna().any(axis=1)
    has_amounts = df.reindex(columns=AMOUNT_COLS, fill_value=np.nan).notna().any(axis=1)

    labels = np.where(
        has_terms & ~has_amounts, "terms_only",
        np.where(
            has_amounts & ~has_terms, "amounts_only",
            np.where(has_terms & has_amounts, "mixed", "empty")
        )
    )
    return pd.Series(labels, index=df.index, name="context_type")


def _choose_scale(x_abs: float, valid_range: Tuple[float, float]) -> Optional[int]:
    lo, hi = valid_range
    candidates = []
    for scale in (1, 100, 10000):
        y = x_abs / scale
        if lo <= y <= hi:
            candidates.append((scale, y))
    if not candidates:
        return None
    if any(scale == 1 for scale, _ in candidates):
        return 1
    mid = (lo + hi) / 2.0
    return min(candidates, key=lambda t: abs(t[1] - mid))[0]


def _normalize_series(s: pd.Series, valid_range: Tuple[float, float]) -> pd.DataFrame:
    norm_vals = []
    flags = []
    lo, hi = valid_range

    for v in s.values:
        if pd.isna(v):
            norm_vals.append(np.nan)
            flags.append("na")
            continue

        x = float(v)
        sign = -1.0 if x < 0 else 1.0
        x_abs = abs(x)

        if lo <= x_abs <= hi:
            norm_vals.append(x)
            flags.append("unchanged")
            continue

        scale = _choose_scale(x_abs, valid_range)
        if scale is None:
            norm_vals.append(x)
            flags.append("unresolved")
            continue

        y = sign * (x_abs / scale)
        norm_vals.append(y)
        flags.append("div100" if scale == 100 else ("div10000" if scale == 10000 else "unchanged"))

    return pd.DataFrame(
        {f"{s.name}_normalized": norm_vals, f"{s.name}_scale_flag": flags},
        index=s.index,
    )


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
        if col not in out.columns:
            continue
        out = pd.concat([out, _normalize_series(out[col], rng)], axis=1)

    # Make PIC/PIK/Floor non-negative after normalization
    for base in [
        "InvestmentInterestRatePaidInCash",
        "InvestmentInterestRatePaidInKind",
        "InvestmentInterestRateFloor",
    ]:
        val_col = f"{base}_normalized"
        flag_col = f"{base}_scale_flag"
        if val_col not in out.columns or flag_col not in out.columns:
            continue

        neg = out[val_col] < 0
        if neg.any():
            out.loc[neg, val_col] = out.loc[neg, val_col].abs()
            # append sign_error without overwriting existing flags
            flags = out.loc[neg, flag_col].astype(str)
            out.loc[neg, flag_col] = np.where(
                flags.str.contains("sign_error", na=False),
                flags,
                flags + "|sign_error",
            )

    return out


def select_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        ["cik", "accession", "investment_identifier", "context_id", "period", "cal_qe", "cal_q"]
        + AMOUNT_COLS
        + TERM_COLS
        + [f"{c}-unitRef" for c in AMOUNT_COLS]
        + ["InvestmentOwnedBalanceShares"]
    )
    cols = [c for c in keep if c in df.columns]
    return df[cols].copy()


# ----------------------------
# Public API
# ----------------------------

def run_preprocess(
    in_csv: str | Path = IXBRL_WIDE_ALL_CSV,
    out_csv: str | Path = PROCESSED_DIR / "ixbrl_clean.csv",
    cfg: Optional[PreprocessConfig] = None,
    start_q: Optional[str] = None,
    end_q: Optional[str] = None,
    save_diagnostics: bool = False,
    chunksize: int = 5_000,
    reload_max_bytes: int = 500_000_000,
) -> pd.DataFrame:
    """
    Input: combined wide CSV (default: PROCESSED_DIR/ixbrl_wide_all.csv)
    Output: one clean analysis-ready CSV (default: PROCESSED_DIR/ixbrl_clean.csv)

    - prefix merge
    - calendar quarter columns
    - optional filtering
    - interest normalization (adds *_normalized and *_scale_flag)

    save_diagnostics:
      - if True, also writes preprocess_nonnull_stats.csv (and respects cfg.write_stats_csv)

    The combined wide table can be very large: hundreds of BDC filers each
    contribute their own XBRL extension-taxonomy columns (no common schema until
    the prefix-merge step below runs), so it's wide *and* long at once - observed
    at ~745k rows x 1,746 cols, ~1.5GB on disk. Loading that as one DataFrame (even
    via the faster pyarrow engine, which was the previous behavior here) is not
    safe on a memory-constrained machine - this processes the file in row chunks
    and appends each cleaned chunk directly to out_csv instead. Every transform
    below (calendar-quarter tagging, prefix-merge, column selection, the quarter/
    share/context-type filters, rate normalization) is purely row-local, so
    chunking gives identical results to doing it on the whole file at once.
    Diagnostics (non-null counts) are the one aggregate step; those are summed
    incrementally across chunks instead.
    """
    cfg = cfg or PreprocessConfig()

    if start_q is not None:
        cfg.quarter_from = start_q
    if end_q is not None:
        cfg.quarter_to = end_q
    if save_diagnostics:
        cfg.write_stats_csv = True

    in_csv = Path(in_csv)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # cik/accession are digit-only strings (no leading zeros/dashes); without an explicit
    # dtype they're silently inferred as int64, which risks breaking any downstream join
    # against zero-padded CIK identifiers elsewhere in the pipeline (e.g. BDC_intervals.csv).
    # engine="pyarrow" doesn't support chunksize, so this uses the default C engine.
    reader = pd.read_csv(
        in_csv, dtype={"cik": str, "accession": str}, chunksize=chunksize, low_memory=False
    )

    stats_totals: dict[str, int] = {}
    stats_rows = 0
    total_in = 0
    total_out = 0
    first_write = True

    for raw_chunk in reader:
        total_in += len(raw_chunk)

        chunk = add_calendar_quarter_columns(raw_chunk, period_col="period")
        chunk = prefix_merge_columns(chunk)

        if cfg.write_stats_csv:
            ns = nonnull_stats(chunk)
            for col, cnt in zip(ns["column"], ns["non_null_count"]):
                stats_totals[col] = stats_totals.get(col, 0) + int(cnt)
            stats_rows += len(chunk)

        chunk = select_analysis_columns(chunk)

        if "cal_q" in chunk.columns:
            chunk["cal_q"] = chunk["cal_q"].astype(str)
            chunk = chunk[chunk["cal_q"].between(cfg.quarter_from, cfg.quarter_to)].copy()

        chunk["context_type"] = classify_context(chunk)

        if cfg.only_no_shares and cfg.keep_share_col in chunk.columns:
            chunk = chunk[chunk[cfg.keep_share_col].isna()].copy()

        if cfg.drop_amounts_only:
            chunk = chunk[chunk["context_type"] != "amounts_only"].copy()

        chunk = normalize_interest_columns(chunk)

        total_out += len(chunk)
        chunk.to_csv(out_csv, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False

    if first_write:
        pd.DataFrame().to_csv(out_csv, index=False)

    logger.info(f"Loaded: {in_csv} (rows={total_in})")
    logger.info(f"Wrote clean output -> {out_csv} (rows={total_out})")

    if cfg.write_stats_csv:
        stats = pd.DataFrame(
            {"column": list(stats_totals.keys()), "non_null_count": list(stats_totals.values())}
        )
        stats["percentage"] = (stats["non_null_count"] / stats_rows * 100.0) if stats_rows else 0.0
        stats = stats.sort_values("non_null_count", ascending=False).reset_index(drop=True)
        stats_path = out_csv.with_name("preprocess_nonnull_stats.csv")
        stats.to_csv(stats_path, index=False)
        logger.info(f"Wrote stats -> {stats_path}")

    # Callers/notebooks expect a DataFrame back, but re-loading a huge cleaned
    # output would recreate the exact problem this function exists to avoid.
    if out_csv.exists() and out_csv.stat().st_size <= reload_max_bytes:
        return pd.read_csv(out_csv, dtype={"cik": str, "accession": str}, low_memory=False)
    logger.warning(
        f"Clean output too large to reload into memory ({out_csv.stat().st_size} bytes) - "
        "returning an empty DataFrame; read the CSV directly (e.g. in chunks) if you need the data."
    )
    return pd.DataFrame()


def preprocess_combined_wide(
    combined_csv: str | Path,
    out_csv: Optional[str | Path] = None,
    cfg: Optional[PreprocessConfig] = None,
) -> pd.DataFrame:

    if out_csv is None:
        out_csv = PROCESSED_DIR / "ixbrl_clean.csv"
    return run_preprocess(in_csv=combined_csv, out_csv=out_csv, cfg=cfg)
