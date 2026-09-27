import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # 2026 Extension/
sys.path.append(str(ROOT / "src"))

from ixbrl_utils import (
    normalize_interest_columns,
    convert_currencies,
    normalize_value_scale,
    flag_filer_quarter_outliers,
    drop_subtotal_rows,
    load_base_rates,
    resolve_rates,
)

warnings.filterwarnings("ignore")


def run_pipeline(data_path: str, fx_path: str, sofr_path: str = "SOFR_augmented.csv") -> pd.DataFrame:
    # -------------------------
    # Load & filter
    # -------------------------
    df = pd.read_csv(data_path, low_memory=False)
    df["cal_qe"] = pd.to_datetime(df["cal_qe"], errors="coerce")
    df["cal_q"] = df["cal_qe"].dt.to_period("Q").astype(str)

    # Shares are normally an equity signal, but HPS Corporate Lending Fund's 2026Q1 10-Q
    # tagged shares on 63.7% of positions (mostly term loans with real rate/FV data), so
    # rows whose context_type is "mixed" are kept even when shares are present.
    has_shares = df["InvestmentOwnedBalanceShares"].notna()
    df = df.loc[~has_shares | df["context_type"].eq("mixed")]
    df = df.loc[~df["context_type"].isin(["amounts_only", "empty"])]

    # -------------------------
    # Normalise units, currency, dollar scale
    # -------------------------
    df = normalize_interest_columns(df)
    df = convert_currencies(df, fx_path)            # also sets df["currency"]
    df = normalize_value_scale(df)

    # Filings that mis-tag FV, cost and principal together (e.g. TCW Direct Lending VIII,
    # 2023Q1) -- see flag_filer_quarter_outliers() docstring.
    df = flag_filer_quarter_outliers(df)
    df = df.loc[~df["filer_quarter_outlier"]].drop(columns="filer_quarter_outlier")

    # Subtotal / heading rows that would double-count FV (see drop_subtotal_rows()).
    df = drop_subtotal_rows(df)

    # -------------------------
    # Rates: role repairs -> fills -> base-rate estimate -> flags
    # -------------------------
    return resolve_rates(df, load_base_rates(sofr_path))


if __name__ == "__main__":
    run_pipeline("ixbrl_clean.csv", "FX.csv", "SOFR_augmented.csv").to_csv("ixbrl_cleaned_out.csv", index=False)
