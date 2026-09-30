"""Notebook-friendly exploratory views for the private-credit pipeline.

The functions here are read-only: they load existing pipeline outputs, display
tables/charts, and return the underlying results for further analysis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from IPython.display import display


DEFAULT_RAW_PANEL_PATH = Path("data/processed/ixbrl_clean.csv")
DEFAULT_ENRICHED_PANEL_PATH = Path("data/processed/data_private_credit_FINAL_enriched.csv")
DEFAULT_BDC_INTERVALS_PATH = Path("data/processed/BDC_intervals.csv")


def visualize_single_filing(
    cik: str,
    accession: str,
    raw_panel_path: str | Path = DEFAULT_RAW_PANEL_PATH,
) -> tuple[pl.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Display the investment-level fields extracted from one filing.

    Parameters
    ----------
    cik, accession
        The filing identifiers to inspect. Both values are treated as strings.
    raw_panel_path
        Preprocessed parser output, normally ``data/processed/ixbrl_clean.csv``.

    Returns
    -------
    single_filing, filing_summary, completeness
        The extracted contexts and the two displayed summary tables.
    """
    display_columns = [
        "investment_identifier", "context_id", "period", "context_type",
        "InvestmentOwnedAtFairValue", "InvestmentOwnedAtCost",
        "InvestmentOwnedBalancePrincipalAmount", "InvestmentBasisSpreadVariableRate",
        "InvestmentInterestRate", "InvestmentInterestRatePaidInCash",
        "InvestmentInterestRatePaidInKind", "InvestmentInterestRateFloor",
        "InvestmentMaturityDate", "InvestmentOwnedAtFairValue-unitRef",
    ]

    single_filing = (
        pl.scan_csv(raw_panel_path)
        .with_columns([
            pl.col("cik").cast(pl.Utf8),
            pl.col("accession").cast(pl.Utf8),
        ])
        .filter((pl.col("cik") == str(cik)) & (pl.col("accession") == str(accession)))
        .select(["cik", "accession"] + display_columns)
        .collect()
    )

    if single_filing.is_empty():
        raise ValueError(
            "No rows found. Choose a CIK/accession pair present in "
            f"{raw_panel_path}."
        )

    filing_summary = single_filing.select([
        pl.col("cik").first().alias("CIK"),
        pl.col("accession").first().alias("Accession"),
        pl.col("period").min().alias("Reporting period"),
        pl.len().alias("Extracted contexts"),
        pl.col("investment_identifier").n_unique().alias("Unique investment labels"),
    ]).to_pandas()

    display(filing_summary)
    display(single_filing.to_pandas())

    fields_to_check = [
        column for column in display_columns
        if column not in {"investment_identifier", "context_id", "period", "context_type"}
    ]
    nonnull = single_filing.select([
        pl.col(column).is_not_null().sum().alias(column)
        for column in fields_to_check
    ]).to_dicts()[0]
    completeness = pd.DataFrame({
        "Field": list(nonnull),
        "Contexts with a value": list(nonnull.values()),
    }).sort_values("Contexts with a value")

    ax = completeness.plot.barh(
        x="Field",
        y="Contexts with a value",
        legend=False,
        figsize=(9, 5),
        color="#2c7fb8",
    )
    ax.set_title("Field coverage within the selected filing")
    ax.set_xlabel("Number of extracted investment contexts")
    plt.tight_layout()
    plt.show()

    return single_filing, filing_summary, completeness


def visualize_borrower_valuation_dispersion(
    borrower_query: str = "First Brands",
    enriched_panel_path: str | Path = DEFAULT_ENRICHED_PANEL_PATH,
    bdc_intervals_path: str | Path = DEFAULT_BDC_INTERVALS_PATH,
    denominator: str = "principal"  # Added parameter: defaults to "principal", accepts "cost"
) -> tuple[pl.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare a borrower's reported valuation across BDCs and quarters.

    The charted measure is a BDC-quarter's fair value divided by either
    principal or cost, depending on the 'denominator' argument.
    """
    borrower_rows = (
        pl.scan_csv(enriched_panel_path)
        .with_columns([
            pl.col("cik").cast(pl.Utf8),
            pl.col("borrower_name").cast(pl.Utf8),
            pl.col("InvestmentOwnedAtFairValue_normalized")
            .cast(pl.Float64, strict=False)
            .alias("fair_value"),
            pl.col("InvestmentOwnedBalancePrincipalAmount_normalized")
            .cast(pl.Float64, strict=False)
            .alias("principal"),
            # Added Cost extraction
            pl.col("InvestmentOwnedAtCost_normalized")
            .cast(pl.Float64, strict=False)
            .alias("cost"),
        ])
        .filter(
            pl.col("borrower_name")
            .str.to_lowercase()
            .str.contains(borrower_query.lower(), literal=True)
        )
        .select([
            "cik", "accession", "cal_q", "borrower_name", "investment_identifier",
            "instrument_seniority", "fair_value", "principal", "cost",
        ])
        .collect()
    )

    if borrower_rows.is_empty():
        raise ValueError(f"No borrower labels matched: {borrower_query!r}")

    matched_labels = borrower_rows.select("borrower_name").unique().sort("borrower_name")
    print(
        f"Matched {borrower_rows.select('cik').n_unique()} BDCs and "
        f"{borrower_rows.height} extracted rows."
    )
    display(matched_labels.to_pandas())

    bdc_names = (
        pl.scan_csv(bdc_intervals_path)
        .select([
            pl.col("CIK").cast(pl.Utf8).alias("cik"),
            pl.col("Company").alias("bdc_name"),
        ])
        .drop_nulls()
        .unique(subset=["cik"], keep="first")
        .collect()
    )

    # Dynamic aggregation based on the chosen denominator
    case_bdc = (
        borrower_rows
        .filter(
            pl.col("fair_value").is_not_null()
            & pl.col(denominator).is_not_null()
            & (pl.col(denominator) > 0)
        )
        .group_by(["cik", "cal_q"])
        .agg([
            pl.col("fair_value").sum().alias("total_fair_value"),
            pl.col(denominator).sum().alias(f"total_{denominator}"),
            pl.len().alias("matching_positions"),
        ])
        .with_columns(
            (pl.col("total_fair_value") / pl.col(f"total_{denominator}"))
            .alias("valuation_ratio")
        )
        .join(bdc_names, on="cik", how="left")
        .with_columns(
            pl.when(pl.col("bdc_name").is_null())
            .then(pl.concat_str([pl.lit("CIK "), pl.col("cik")]))
            .otherwise(pl.col("bdc_name"))
            .alias("BDC")
        )
        .sort(["cal_q", "BDC"])
        .to_pandas()
    )
    
    case_bdc["valuation_pct"] = 100 * case_bdc["valuation_ratio"]
    case_bdc["quarter_order"] = (
        case_bdc["cal_q"].str.slice(0, 4).astype(int) * 4
        + case_bdc["cal_q"].str[-1].astype(int)
    )
    case_bdc = case_bdc.sort_values(["quarter_order", "BDC"])

    # Dynamically display the correct total column
    display(case_bdc[[
        "cal_q", "BDC", "matching_positions", "total_fair_value",
        f"total_{denominator}", "valuation_pct",
    ]])

    dispersion_summary = (
        case_bdc.groupby("cal_q", as_index=False)["valuation_pct"]
        .agg(
            BDCs="count",
            minimum="min",
            median="median",
            maximum="max",
            standard_deviation="std",
        )
    )
    display(dispersion_summary)

    quarters = (
        case_bdc[["cal_q", "quarter_order"]]
        .drop_duplicates()
        .sort_values("quarter_order")
    )
    quarter_labels = quarters["cal_q"].tolist()
    quarter_positions = quarters["quarter_order"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    for bdc, group in case_bdc.groupby("BDC"):
        group = group.sort_values("quarter_order")
        axes[0].plot(
            group["quarter_order"],
            group["valuation_pct"],
            marker="o",
            linewidth=1.5,
            label=bdc,
        )
    axes[0].axhline(100, color="black", linestyle="--", linewidth=1, label="Par/Cost")
    axes[0].set_title(f"{borrower_query}: reported valuation by BDC")
    
    # Dynamic Y-axis label
    y_label = f"Fair value / {denominator.capitalize()} (%)"
    axes[0].set_ylabel(y_label)
    axes[0].set_xticks(quarter_positions, quarter_labels, rotation=45)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)

    box_data = [
        case_bdc.loc[case_bdc["cal_q"] == quarter, "valuation_pct"].dropna()
        for quarter in quarter_labels
    ]
    axes[1].boxplot(box_data, labels=quarter_labels, showfliers=False)
    for position, quarter in enumerate(quarter_labels, start=1):
        values = case_bdc.loc[
            case_bdc["cal_q"] == quarter, "valuation_pct"
        ].dropna().to_numpy()
        if len(values) == 0:
            continue
        jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.array([0.0])
        axes[1].scatter(position + jitter, values, color="#d95f02", alpha=0.75, s=25)
    axes[1].axhline(100, color="black", linestyle="--", linewidth=1)
    axes[1].set_title(f"{borrower_query}: cross-BDC valuation dispersion")
    axes[1].set_ylabel(y_label)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    return borrower_rows, case_bdc, dispersion_summary



def screen_valuation_dispersion(
    enriched_panel_path: str | Path = DEFAULT_ENRICHED_PANEL_PATH,
    min_bdcs: int = 4,
    min_total_amount: float = 30_000_000.0,
    min_spread_pct: float = 15.0,
    quarter: str | None = None,
    denominator: str = "cost",
    top_n: int = 20,
) -> pd.DataFrame:
    """Screen for multi-lender borrowers showing significant valuation dispersion.

    Parameters
    ----------
    enriched_panel_path
        Path to the enriched private credit panel.
    min_bdcs
        Minimum number of distinct BDCs holding the borrower in the same quarter.
    min_total_amount
        Minimum aggregate cost or par across all lenders (in dollars).
    min_spread_pct
        Minimum difference between highest and lowest BDC mark (percentage points).
    quarter
        Optional quarter string (e.g. '2024Q4') to filter, or None for all quarters.
    denominator
        'cost' to measure Fair Value / Cost, or 'principal' for Fair Value / Par.
    top_n
        Number of top candidates to return.

    Returns
    -------
    pd.DataFrame
        Ranked table of borrowers with the widest cross-BDC valuation divergence.
    """
    denom_col = (
        "InvestmentOwnedAtCost_normalized"
        if denominator.lower() == "cost"
        else "InvestmentOwnedBalancePrincipalAmount_normalized"
    )

    query = (
        pl.scan_csv(enriched_panel_path)
        .with_columns([
            pl.col("cik").cast(pl.Utf8),
            pl.col("borrower_name").cast(pl.Utf8),
            pl.col("cal_q").cast(pl.Utf8),
            pl.col("InvestmentOwnedAtFairValue_normalized")
            .cast(pl.Float64, strict=False)
            .alias("fair_value"),
            pl.col(denom_col)
            .cast(pl.Float64, strict=False)
            .alias("denom_val"),
        ])
        .filter(
            pl.col("borrower_name").is_not_null()
            & (pl.col("borrower_name") != "")
            & (pl.col("borrower_name") != "Other / Unknown / Unresolved")
            & pl.col("fair_value").is_not_null()
            & pl.col("denom_val").is_not_null()
            & (pl.col("denom_val") > 0)
        )
    )

    if quarter is not None:
        query = query.filter(pl.col("cal_q") == quarter)

    # Step 1: Aggregate per borrower, quarter, and BDC (sums multiple tranches/positions)
    bdc_level = (
        query.group_by(["borrower_name", "cal_q", "cik"])
        .agg([
            pl.col("fair_value").sum().alias("bdc_fv"),
            pl.col("denom_val").sum().alias("bdc_denom"),
        ])
        .with_columns(
            (pl.col("bdc_fv") / pl.col("bdc_denom") * 100.0).alias("bdc_val_pct")
        )
    )

    # Step 2: Cross-BDC dispersion metrics per borrower and quarter
    dispersion = (
        bdc_level.group_by(["borrower_name", "cal_q"])
        .agg([
            pl.col("cik").n_unique().alias("num_bdcs"),
            (pl.col("bdc_denom").sum() / 1e6).alias(f"total_{denominator}_m"),
            (pl.col("bdc_fv").sum() / 1e6).alias("total_fv_m"),
            pl.col("bdc_val_pct").min().alias("min_mark_pct"),
            pl.col("bdc_val_pct").median().alias("median_mark_pct"),
            pl.col("bdc_val_pct").max().alias("max_mark_pct"),
            (pl.col("bdc_val_pct").max() - pl.col("bdc_val_pct").min()).alias("spread_pct"),
            pl.col("bdc_val_pct").std().alias("std_dev_pct"),
        ])
        .filter(
            (pl.col("num_bdcs") >= min_bdcs)
            & (pl.col(f"total_{denominator}_m") >= (min_total_amount / 1e6))
            & (pl.col("spread_pct") >= min_spread_pct)
        )
        .sort(by=["spread_pct", f"total_{denominator}_m"], descending=[True, True])
        .head(top_n)
        .collect()
        .to_pandas()
    )

    if dispersion.empty:
        print("No borrowers matched the screening criteria. Consider lowering min_bdcs or min_spread_pct.")
        return pd.DataFrame()

    # Round numeric columns for clean notebook rendering
    numeric_cols = [
        f"total_{denominator}_m", "total_fv_m",
        "min_mark_pct", "median_mark_pct", "max_mark_pct",
        "spread_pct", "std_dev_pct"
    ]
    dispersion[numeric_cols] = dispersion[numeric_cols].round(2)

    return dispersion


import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_borrower_universe(enriched_panel_path="data/processed/data_private_credit_FINAL_enriched.csv"):
    df = pl.read_csv(enriched_panel_path, ignore_errors=True)

    # Basic text cleanup to group obvious variations together
    df = df.with_columns(
        pl.col("borrower_name").str.to_lowercase().str.split("-").list.first().str.strip_chars().alias("borrower_name")
    )

    agg_df = (
        df.group_by(["borrower_name", "cal_q"])
        .agg([
            pl.col("cik").n_unique().alias("num_lenders"),
            pl.col("InvestmentOwnedBalancePrincipalAmount").sum().alias("total_principal")
        ])
    )

    peak_df = (
        agg_df.group_by("borrower_name")
        .agg([
            pl.col("num_lenders").max(),
            pl.col("total_principal").max()
        ])
        .to_pandas()
    )

    peak_df = peak_df[peak_df['total_principal'] > 0].copy()
    peak_df['total_principal_millions'] = peak_df['total_principal'] / 1_000_000

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Chart 1: Lenders (Linear Scale)
    sns.countplot(data=peak_df, x="num_lenders", ax=axes[0], color="steelblue")
    axes[0].set_title("Distribution: Lenders per Borrower", fontweight="bold")
    axes[0].set_xlabel("Number of BDCs in Syndicate")
    axes[0].set_ylabel("Count of Unique Loan Identifiers")
    axes[0].set_xlim(-0.5, 9.5) 

    # Chart 2: Loan Sizes (Log Scale)
    sns.histplot(data=peak_df, x="total_principal_millions", bins=40, ax=axes[1], color="darkorange", log_scale=(True, False))
    axes[1].set_title("Distribution: Loan Sizes ($ Millions)", fontweight="bold")
    axes[1].set_xlabel("Total Principal Amount ($M) - Log Scale")
    axes[1].set_ylabel("Count of Unique Loan Identifiers")

    plt.tight_layout()
    plt.show()

    # --- COMPLETE QUANTILE BREAKDOWN ---
    print("--- UNIVERSE DISTRIBUTION STATS ---")
    print(f"Total Unique Loan Identifiers: {len(peak_df):,}\n")
    
    # Calculate pandas quantiles for a clean printout
    lender_q = peak_df['num_lenders'].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    prin_q = peak_df['total_principal_millions'].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])

    print("LENDER SYNDICATION (Percentiles):")
    print(f"  Min (0%):    {lender_q[0.0]:.0f} BDCs")
    print(f"  25th (Q1):   {lender_q[0.25]:.0f} BDCs")
    print(f"  50th (Med):  {lender_q[0.5]:.0f} BDCs")
    print(f"  75th (Q3):   {lender_q[0.75]:.0f} BDCs")
    print(f"  90th:        {lender_q[0.9]:.0f} BDCs")
    print(f"  95th:        {lender_q[0.95]:.0f} BDCs")
    print(f"  99th:        {lender_q[0.99]:.0f} BDCs")
    print(f"  Max (100%):  {lender_q[1.0]:.0f} BDCs")

    print("\nTOTAL PRINCIPAL AMOUNT (Percentiles):")
    print(f"  Min (0%):    ${prin_q[0.0]:,.1f}M")
    print(f"  25th (Q1):   ${prin_q[0.25]:,.1f}M")
    print(f"  50th (Med):  ${prin_q[0.5]:,.1f}M")
    print(f"  75th (Q3):   ${prin_q[0.75]:,.1f}M")
    print(f"  90th:        ${prin_q[0.9]:,.1f}M")
    print(f"  95th:        ${prin_q[0.95]:,.1f}M")
    print(f"  99th:        ${prin_q[0.99]:,.1f}M")
    print(f"  Max (100%):  ${prin_q[1.0]:,.1f}M")
    
    p90_lenders = int(lender_q[0.9])
    p90_principal = float(prin_q[0.9] * 1_000_000) # Convert back to raw dollars for the screener
    
    return p90_lenders, p90_principal


import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def analyze_dispersion_statistics(enriched_panel_path="data/processed/data_private_credit_FINAL_enriched.csv"):
    """
    Generates advanced statistical charts for BDC valuation dispersion:
    1. The Distribution of Valuation Spreads
    2. Dispersion vs. Distress (Mean Mark vs Spread)
    3. Systematic BDC Marking Bias (Aggressive vs Conservative)
    """
    df = pl.read_csv(enriched_panel_path, ignore_errors=True)

    # 1. Basic text cleanup for borrower names
    df = df.with_columns(
        pl.col("borrower_name").str.to_lowercase().str.split("-").list.first().str.strip_chars().alias("borrower_name")
    )
    
    # Calculate Valuation Ratio for each row
    df = df.with_columns((pl.col("FV") / pl.col("PAR")).alias("val_ratio"))

    # 2. Filter for Syndicated Loans Only (2+ BDCs lending to the same borrower in the same quarter)
    syndicate_df = (
        df.group_by(["borrower_name", "cal_q"])
        .agg([
            pl.col("cik").n_unique().alias("num_lenders"),
            pl.col("val_ratio").max().alias("max_mark"),
            pl.col("val_ratio").min().alias("min_mark"),
            pl.col("val_ratio").mean().alias("mean_mark")
        ])
        .filter(pl.col("num_lenders") >= 2)
        # Drop crazy outliers (data errors where ratio > 2.0 or < 0.0)
        .filter((pl.col("max_mark") <= 2.0) & (pl.col("min_mark") >= 0.0))
    ).to_pandas()

    # Calculate the Spread (Max - Min)
    syndicate_df['valuation_spread'] = syndicate_df['max_mark'] - syndicate_df['min_mark']
    
    # Convert ratios to percentages for cleaner charts (e.g., 0.15 -> 15%)
    syndicate_df['valuation_spread_pct'] = syndicate_df['valuation_spread'] * 100
    syndicate_df['mean_mark_pct'] = syndicate_df['mean_mark'] * 100

    # 3. Setup the 1x3 Figure layout
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- CHART 1: The Disagreement Distribution ---
    sns.histplot(data=syndicate_df, x="valuation_spread_pct", bins=50, ax=axes[0], color="purple")
    axes[0].set_title("Syndicate Disagreement Rate", fontweight="bold")
    axes[0].set_xlabel("Valuation Spread (Max Mark - Min Mark) %")
    axes[0].set_ylabel("Count of Syndicated Borrower-Quarters")
    axes[0].set_xlim(0, 30) # Cap at 30% spread for readability

    # --- CHART 2: The Distress Multiplier ---
    # Bin the mean marks into categories to show how spread jumps as distress increases
    bins = [0, 60, 80, 95, 105]
    labels = ['Severe Distress (<60)', 'Stressed (60-80)', 'Watchlist (80-95)', 'Performing (>95)']
    syndicate_df['distress_tier'] = pd.cut(syndicate_df['mean_mark_pct'], bins=bins, labels=labels)
    
    sns.boxplot(data=syndicate_df, x="valuation_spread_pct", y="distress_tier", ax=axes[1], palette="Reds_r", showfliers=False)
    axes[1].set_title("Dispersion Explodes During Distress", fontweight="bold")
    axes[1].set_xlabel("Average Valuation Spread (%)")
    axes[1].set_ylabel("")
# --- CHART 3: Aggressive vs Conservative BDCs (FIXED) ---
    pd_df = df.to_pandas()
    pd_df['val_ratio'] = pd_df['FV'] / pd_df['PAR']
    
    merged_df = pd.merge(pd_df, syndicate_df[['borrower_name', 'cal_q', 'mean_mark']], on=['borrower_name', 'cal_q'], how='inner')
    merged_df['bdc_deviation_pct'] = (merged_df['val_ratio'] - merged_df['mean_mark']) * 100
    
    top_bdcs = merged_df['cik'].value_counts().head(10).index
    bias_df = merged_df[merged_df['cik'].isin(top_bdcs)].copy()
    
    # Force CIK to string so seaborn treats it as a discrete label
    bias_df['cik_str'] = bias_df['cik'].astype(str)
    
    bdc_bias = (
        bias_df.groupby('cik_str')['bdc_deviation_pct']
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    
    # Use orient="y" with explicit x and y to draw clean horizontal bars
    sns.barplot(
        data=bdc_bias, 
        x="bdc_deviation_pct", 
        y="cik_str", 
        ax=axes[2], 
        palette="coolwarm"
    )
    axes[2].set_title("Systematic Marking Bias (Top 10 BDCs)", fontweight="bold")
    axes[2].set_xlabel("Average Deviation from Syndicate Mean (%)")
    axes[2].set_ylabel("Lender CIK")
    axes[2].axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("--- SYNDICATE DISPERSION HIGHLIGHTS ---")
    print(f"Total Syndicated Borrower-Quarters Analyzed: {len(syndicate_df):,}")
    spread_gt_5 = len(syndicate_df[syndicate_df['valuation_spread_pct'] >= 5]) / len(syndicate_df)
    spread_gt_10 = len(syndicate_df[syndicate_df['valuation_spread_pct'] >= 10]) / len(syndicate_df)
    print(f"Loans with >5% Valuation Disagreement: {spread_gt_5:.1%}")
    print(f"Loans with >10% Valuation Disagreement: {spread_gt_10:.1%}")



import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_macro_dispersion_trend(enriched_panel_path="data/processed/data_private_credit_FINAL_enriched.csv"):
    """
    Plots the median and 90th percentile valuation spread across calendar quarters 
    to track systemic disagreement over the macro cycle.
    """
    df = pl.read_csv(enriched_panel_path, ignore_errors=True)

    df = df.with_columns(
        pl.col("borrower_name").str.to_lowercase().str.split("-").list.first().str.strip_chars().alias("borrower_name"),
        (pl.col("FV") / pl.col("PAR")).alias("val_ratio")
    )

    # Filter for Syndicated Loans Only
    syndicate_df = (
        df.group_by(["borrower_name", "cal_q"])
        .agg([
            pl.col("cik").n_unique().alias("num_lenders"),
            pl.col("val_ratio").max().alias("max_mark"),
            pl.col("val_ratio").min().alias("min_mark")
        ])
        .filter(pl.col("num_lenders") >= 2)
        .filter((pl.col("max_mark") <= 2.0) & (pl.col("min_mark") >= 0.0))
    ).to_pandas()

    syndicate_df['valuation_spread_pct'] = (syndicate_df['max_mark'] - syndicate_df['min_mark']) * 100

    # Aggregate by Quarter
    trend_df = syndicate_df.groupby('cal_q')['valuation_spread_pct'].agg(
        median_spread='median',
        p90_spread=lambda x: np.percentile(x, 90)
    ).reset_index().sort_values('cal_q')

    # Plotting
    plt.figure(figsize=(12, 6))
    
    sns.lineplot(data=trend_df, x='cal_q', y='median_spread', marker='o', label='Median Spread (%)', linewidth=2, color='steelblue')
    sns.lineplot(data=trend_df, x='cal_q', y='p90_spread', marker='o', label='90th Percentile Spread (Extreme Disagreement)', linewidth=2, color='darkred')
    
    # Fill the area between to highlight the widening gap in distress
    plt.fill_between(trend_df['cal_q'], trend_df['median_spread'], trend_df['p90_spread'], color='red', alpha=0.1)

    plt.title("Macro Time-Series: BDC Valuation Dispersion Over Time", fontweight="bold")
    plt.xlabel("Calendar Quarter")
    plt.ylabel("Valuation Spread (%)")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("--- QUARTERLY DISPERSION TRENDS ---")
    print(trend_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))