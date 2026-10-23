"""Notebook-friendly exploratory views for the private-credit pipeline.

The functions here are read-only: they load existing pipeline outputs, display
tables/charts, and return the underlying results for further analysis.

Valuation dispersion follows the Risk Dimensions doc (section 3): loans are matched
across BDCs at the tranche level on lien, spread and maturity (src/1_data_processing/
tranches.py, never identifier text), each manager family is one holder, price is fair
value / par, and unfunded commitments are left out.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from IPython.display import display

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "1_data_processing"))  # tranches.py, manager_family.py
import tranches as T  # noqa: E402

_DATA = _ROOT / "data"   # works from any working directory
DEFAULT_RAW_PANEL_PATH = _DATA / "parsed" / "ixbrl_clean.csv"
DEFAULT_BDC_INTERVALS_PATH = _DATA / "universe" / "BDC_intervals.csv"
DEFAULT_FINAL_PANEL_PATH = _DATA / "final_dataset" / "FINAL_DATASET.csv"

_TRANCHE_CACHE: dict = {}


def load_final_panel(path: str | Path = DEFAULT_FINAL_PANEL_PATH) -> pl.DataFrame:
    """Positions from FINAL_DATASET.csv: ``cik``, ``cal_q``, ``borrower_name`` (the
    validated ``borrower_canonical``), ``FV`` and ``PAR``. Rows with no borrower or no
    fair value are dropped. The dispersion views below use load_tranche_panel instead."""
    return (
        pl.read_csv(
            path,
            columns=["cik", "cal_q", "borrower_canonical",
                     "InvestmentOwnedAtFairValue_normalized",
                     "InvestmentOwnedBalancePrincipalAmount_normalized"],
            schema_overrides={"InvestmentOwnedAtFairValue_normalized": pl.Float64,
                              "InvestmentOwnedBalancePrincipalAmount_normalized": pl.Float64},
        )
        .rename({"borrower_canonical": "borrower_name",
                 "InvestmentOwnedAtFairValue_normalized": "FV",
                 "InvestmentOwnedBalancePrincipalAmount_normalized": "PAR"})
        .filter(pl.col("borrower_name").is_not_null() & (pl.col("borrower_name").str.strip_chars() != "")
                & pl.col("FV").is_not_null())
    )


def load_tranche_panel(path: str | Path = DEFAULT_FINAL_PANEL_PATH) -> pd.DataFrame:
    """Funded debt tranches with chains and cross-BDC matches (tranches.build_tranche_panel).
    Built once per path per session (about a minute) and cached."""
    key = str(path)
    if key not in _TRANCHE_CACHE:
        raw = pd.read_csv(path, low_memory=False)
        _TRANCHE_CACHE[key] = T.build_tranche_panel(raw, with_origination=False)
    return _TRANCHE_CACHE[key]


def compute_tranche_dispersion(
    final_panel_path: str | Path = DEFAULT_FINAL_PANEL_PATH,
    min_families: int = 2,
    max_tier: int = 2,
) -> pd.DataFrame:
    """V1, V2 and V4 per cross-BDC tranche and quarter (tranches.tranche_dispersion).

    max_tier: 1 = maturity matched on every holder (strictest, small sample); 2 adds spread
    matches where maturity is missing (the default); 3 adds borrower x lien matches with no
    terms at all."""
    return T.tranche_dispersion(load_tranche_panel(final_panel_path), min_families=min_families, max_tier=max_tier)


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
        Preprocessed parser output, normally ``data/parsed/ixbrl_clean.csv``.

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
    final_panel_path: str | Path = DEFAULT_FINAL_PANEL_PATH,
    lien_group: str | None = None,
    max_tier: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Each manager family's price (FV / par) on one borrower's tranches, by quarter.

    The borrower is a case-insensitive substring of the validated name. Each quarter the
    borrower's largest cross-BDC tranche in `lien_group` is shown (default: the borrower's
    biggest lien group by par), so the lines compare holders on the same loan.

    Returns the family prices of every tranche, the charted tranche per quarter, and a
    per-quarter summary (families, min, median, max, std).
    """
    tr = load_tranche_panel(final_panel_path)
    hit = tr[tr["borrower_key"].str.contains(borrower_query, case=False, regex=False)]
    if hit.empty:
        raise ValueError(f"No borrower labels matched: {borrower_query!r}")
    print(f"Matched borrowers: {sorted(hit['borrower_key'].unique())}")
    fam = T.family_tranche_prices(hit[hit["xb_tier"].fillna(9) <= max_tier])
    if lien_group is None:
        lien_group = fam.groupby("lien_group")["PAR"].sum().idxmax()
    sel = fam[fam["lien_group"] == lien_group]
    main = sel.groupby(["cal_q", "xb_id"])["PAR"].sum().reset_index().sort_values("PAR").drop_duplicates("cal_q", keep="last")
    shown = sel.merge(main[["cal_q", "xb_id"]], on=["cal_q", "xb_id"])
    shown = shown.assign(price_pct=100 * shown["price"]).sort_values(["cal_q", "family"])
    display(shown[["cal_q", "family", "n_bdcs", "PAR", "FV", "price_pct", "tier"]])

    summary = shown.groupby("cal_q")["price_pct"].agg(families="count", minimum="min", median="median",
                                                    maximum="max", standard_deviation="std").reset_index()
    display(summary)

    quarters = sorted(shown["cal_q"].unique())
    xpos = {q: i for i, q in enumerate(quarters)}
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    for family, g in shown.groupby("family"):
        axes[0].plot([xpos[q] for q in g["cal_q"]], g["price_pct"], marker="o", linewidth=1.5, label=family)
    axes[0].axhline(100, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"{borrower_query} ({lien_group}): price by manager family")
    axes[0].set_ylabel("Fair value / par (%)")
    axes[0].set_xticks(range(len(quarters)), quarters, rotation=45)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
    axes[1].boxplot([shown.loc[shown["cal_q"] == q, "price_pct"] for q in quarters], labels=quarters, showfliers=False)
    axes[1].axhline(100, color="black", linestyle="--", linewidth=1)
    axes[1].set_title(f"{borrower_query} ({lien_group}): dispersion across families")
    axes[1].set_ylabel("Fair value / par (%)")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()
    return fam, shown, summary


def screen_valuation_dispersion(
    final_panel_path: str | Path = DEFAULT_FINAL_PANEL_PATH,
    min_families: int = 2,
    min_total_par: float = 0.0,
    min_range_pct: float = 15.0,
    quarter: str | None = None,
    max_tier: int = 2,
    top_n: int = 20,
) -> pd.DataFrame:
    """Tranches whose holders disagree most: range of family prices (max - min, points of
    par) >= min_range_pct, held by >= min_families manager families, in one quarter or all.
    Sorted by range, then par. Price is FV / par; unfunded commitments are excluded."""
    d = compute_tranche_dispersion(final_panel_path, min_families=min_families, max_tier=max_tier)
    if quarter is not None:
        d = d[d["cal_q"] == quarter]
    d = d[(d["total_par"] >= min_total_par) & (d["V1_range"] * 100 >= min_range_pct)]
    if d.empty:
        print("No tranches matched the screening criteria. Consider lowering min_range_pct.")
        return pd.DataFrame()
    out = d.sort_values(["V1_range", "total_par"], ascending=False).head(top_n)
    out = out.assign(total_par_m=out["total_par"] / 1e6,
                     **{c + "_pct": out[c] * 100 for c in ["V4_consensus", "p_min", "p_max", "V1_range", "V1_std", "V2_low_mark"]})
    cols = ["cal_q", "borrower_key", "lien_group", "n_families", "n_bdcs", "tier", "total_par_m",
            "V4_consensus_pct", "p_min_pct", "p_max_pct", "V1_range_pct", "V1_std_pct", "V2_low_mark_pct"]
    return out[cols].round(2).reset_index(drop=True)


def analyze_borrower_universe(final_panel_path=DEFAULT_FINAL_PANEL_PATH):
    """Lenders per borrower (manager families) and loan sizes, using each borrower's peak
    quarter, on funded debt. Returns the 90th-percentile family count and principal."""
    tr = load_tranche_panel(final_panel_path)
    tr = tr[~tr["borrower_key"].str.startswith("ID: ")]
    agg = tr.groupby(["borrower_key", "cal_q"]).agg(num_lenders=("family", "nunique"), total_principal=("PAR", "sum"))
    peak_df = agg.groupby("borrower_key").max().reset_index()
    peak_df["total_principal_millions"] = peak_df["total_principal"] / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(data=peak_df, x="num_lenders", ax=axes[0], color="steelblue")
    axes[0].set_title("Distribution: Manager Families per Borrower", fontweight="bold")
    axes[0].set_xlabel("Manager families lending (peak quarter)")
    axes[0].set_ylabel("Borrowers")
    axes[0].set_xlim(-0.5, 9.5)
    sns.histplot(data=peak_df, x="total_principal_millions", bins=40, ax=axes[1], color="darkorange", log_scale=(True, False))
    axes[1].set_title("Distribution: Funded Principal per Borrower ($M)", fontweight="bold")
    axes[1].set_xlabel("Total principal held by BDCs ($M), log scale")
    axes[1].set_ylabel("Borrowers")
    plt.tight_layout()
    plt.show()

    qs = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    lender_q = peak_df["num_lenders"].quantile(qs)
    prin_q = peak_df["total_principal_millions"].quantile(qs)
    print(f"--- UNIVERSE DISTRIBUTION STATS ({len(peak_df):,} borrowers) ---")
    print(pd.DataFrame({"families": lender_q.round(0), "principal_$M": prin_q.round(1)}).to_string())
    return int(lender_q[0.9]), float(prin_q[0.9] * 1e6)


def analyze_dispersion_statistics(final_panel_path=DEFAULT_FINAL_PANEL_PATH, max_tier: int = 2):
    """Three views of cross-family valuation dispersion on matched tranches:

    1. Distribution of the range of family prices (points of par).
    2. Median dispersion (V1 std) by consensus-price decile. Dispersion and price level are
       mechanically linked (a loan everyone marks at 100 cannot disperse), so dispersion is
       only informative relative to the consensus (V4). Deciles replace the old fixed
       <60 / 60-80 / 80-95 / >95 buckets.
    3. Lender optimism (V3, descriptive): each family's average price minus the median of
       the families on the same tranche, for the 15 families with the most shared tranches.
    """
    tr = load_tranche_panel(final_panel_path)
    disp = T.tranche_dispersion(tr, min_families=2, max_tier=max_tier)
    disp = disp.assign(range_pct=disp["V1_range"] * 100, std_pct=disp["V1_std"] * 100)

    fam = T.family_tranche_prices(tr, min_families=2)
    fam = fam[fam["xb_id"].isin(disp["xb_id"])]
    fam = fam.assign(dev_pct=100 * (fam["price"] - fam.groupby(["cal_q", "xb_id"])["price"].transform("median")))
    top = fam["family"].value_counts().head(15).index
    bias = fam[fam["family"].isin(top)].groupby("family")["dev_pct"].mean().sort_values(ascending=False).reset_index()

    disp["consensus_decile"] = pd.qcut(disp["V4_consensus"].rank(method="first"), 10, labels=range(1, 11))
    by_dec = disp.groupby("consensus_decile", observed=True).agg(
        consensus_pct=("V4_consensus", lambda s: 100 * s.median()), median_std_pct=("std_pct", "median"),
        p90_std_pct=("std_pct", lambda s: s.quantile(0.9)), n=("std_pct", "size")).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.histplot(data=disp, x="range_pct", bins=60, ax=axes[0], color="purple")
    axes[0].set_title("Range of family prices per tranche", fontweight="bold")
    axes[0].set_xlabel("Max - min price (points of par)")
    axes[0].set_xlim(0, 30)
    axes[1].plot(by_dec["consensus_pct"], by_dec["median_std_pct"], marker="o", label="median")
    axes[1].plot(by_dec["consensus_pct"], by_dec["p90_std_pct"], marker="s", linestyle="--", label="90th pct")
    axes[1].set_title("Dispersion by consensus-price decile", fontweight="bold")
    axes[1].set_xlabel("Median consensus price in decile (% of par)")
    axes[1].set_ylabel("Std of family prices (points)")
    axes[1].legend()
    sns.barplot(data=bias, x="dev_pct", y="family", ax=axes[2], color="steelblue")
    axes[2].axvline(0, color="black", linestyle="--")
    axes[2].set_title("Lender optimism: average deviation from tranche median", fontweight="bold")
    axes[2].set_xlabel("Points of par (positive = marks above peers)")
    axes[2].set_ylabel("")
    plt.tight_layout()
    plt.show()

    print("--- TRANCHE DISPERSION HIGHLIGHTS ---")
    print(f"Matched tranche-quarters (2+ families, tier <= {max_tier}): {len(disp):,}")
    print(f"Range >= 5 points: {(disp['range_pct'] >= 5).mean():.1%}   >= 10 points: {(disp['range_pct'] >= 10).mean():.1%}")
    print(by_dec.round(2).to_string(index=False))
    return disp, by_dec, bias


def analyze_macro_dispersion_trend(final_panel_path=DEFAULT_FINAL_PANEL_PATH, max_tier: int = 2):
    """Median and 90th-percentile range of family prices per quarter, on matched tranches."""
    disp = compute_tranche_dispersion(final_panel_path, max_tier=max_tier)
    trend_df = (disp.assign(range_pct=disp["V1_range"] * 100)
                .groupby("cal_q")["range_pct"]
                .agg(median_spread="median", p90_spread=lambda x: np.percentile(x, 90), n_tranches="size")
                .reset_index().sort_values("cal_q"))

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=trend_df, x="cal_q", y="median_spread", marker="o", label="Median range (points)", linewidth=2, color="steelblue")
    sns.lineplot(data=trend_df, x="cal_q", y="p90_spread", marker="o", label="90th percentile range", linewidth=2, color="darkred")
    plt.fill_between(trend_df["cal_q"], trend_df["median_spread"], trend_df["p90_spread"], color="red", alpha=0.1)
    plt.title("Valuation dispersion across manager families, matched tranches", fontweight="bold")
    plt.xlabel("Calendar Quarter")
    plt.ylabel("Max - min family price (points of par)")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("--- QUARTERLY DISPERSION TRENDS ---")
    print(trend_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return trend_df
