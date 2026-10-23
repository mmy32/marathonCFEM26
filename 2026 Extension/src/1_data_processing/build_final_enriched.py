"""Build data_private_credit_FINAL_enriched_final.csv from three inputs
(ixbrl_cleaned_out.csv in data/cleaned, the other two in data/enriched); written to data/enriched:

- ixbrl_cleaned_out.csv: the cleaned panel. Every column is read and written back as text, so
  values are exactly what run_ixbrl_pipeline.py produced; DROP_COLUMNS are left out.
- borrower_name_cleaned.csv: teammate's borrower lookup (investment_identifier -> borrower_canonical).
  Despite its name it is an .xlsx workbook. Exact join; identifiers it does not cover stay blank.
  Names are put through normalize_borrower_name() (upper case, no punctuation, no legal-form
  words such as LLC or Inc) so spelling variants of one borrower share one name.
- _seniority_sector_llm_checkpoint.csv: the cached LLM sector answers from get_seniority_debttype.py.

instrument_seniority and sector follow get_seniority_debttype.py exactly. Its rules are nested in
one function and cannot be imported, so they are copied verbatim below:
  - instrument_seniority: keyword rules on investment_identifier.
  - sector: keyword rules; if those give "Other / Unknown", the cached LLM answer when its
    confidence is >= 0.70; otherwise "Other / Unknown / Unresolved".
The LLM is never called: unresolved identifiers missing from the checkpoint get
"Other / Unknown / Unresolved".

The output is written to a .partial file, checked against the panel, and only then renamed.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py

import re
import unicodedata

import numpy as np
import pandas as pd
import polars as pl

from paths import CLEANED_DIR, ENRICHED_DIR

BASE_CSV = CLEANED_DIR / "ixbrl_cleaned_out.csv"
BORROWER_XLSX = ENRICHED_DIR / "borrower_name_cleaned.csv"
LLM_CHECKPOINT_CSV = ENRICHED_DIR / "_seniority_sector_llm_checkpoint.csv"
OUTPUT_CSV = ENRICHED_DIR / "data_private_credit_FINAL_enriched_final.csv"

CONF_THRESH = 0.70  # what the Sep 2026 data_private_credit_FINAL_enriched.csv (now in data/_archive/enriched_unused/) was built with
UNRESOLVED = "Other / Unknown / Unresolved"
CHUNK_ROWS = 50_000

DROP_COLUMNS = [
    # As-reported values, superseded by their *_normalized columns
    "InvestmentOwnedAtCost", "InvestmentOwnedAtFairValue", "InvestmentOwnedBalancePrincipalAmount",
    "InvestmentOwnedAtCost-unitRef", "InvestmentOwnedAtFairValue-unitRef", "InvestmentOwnedBalancePrincipalAmount-unitRef",
    "InvestmentBasisSpreadVariableRate", "InvestmentInterestRate", "InvestmentInterestRatePaidInCash",
    "InvestmentInterestRatePaidInKind", "InvestmentInterestRateFloor",
    "InvestmentVariableInterestRateTypeExtensibleEnumeration",  # RateType is this minus the URL prefix
    # Cleaning-step bookkeeping (rate_source, check_1 and principal_cost_check are kept)
    "InvestmentBasisSpreadVariableRate_scale_flag", "InvestmentInterestRateFloor_scale_flag",
    "InvestmentInterestRate_scale_flag", "InvestmentInterestRatePaidInCash_scale_flag",
    "InvestmentInterestRatePaidInKind_scale_flag", "InvestmentOwnedAtFairValue_normalized_scale_flag",
    "InvestmentOwnedAtCost_normalized_scale_flag", "InvestmentOwnedBalancePrincipalAmount_normalized_scale_flag",
    "unit_mismatch_fixed", "change_tracker", "rate_config", "estimate", "context_type",
    # check_2 (= rate_config: which rate fields the filing tagged) is kept: 5_pik uses it for PIK provenance
]

# Legal-form words dropped from borrower names. ANYWHERE_FORMS go wherever they follow the first
# word; END_FORMS only as trailing words, because elsewhere they are ordinary words or initials
# ("CO", "SA", "AB"). Spaced forms like "L L C" occur in the lookup as written.
ANYWHERE_FORMS = ["LLC", "INC", "INCORPORATED", "CORP", "CORPORATION", "LTD", "LP", "LLP", "LLLP", "PLC", "GMBH", "SARL", "ULC"]
END_FORMS = [
    "L L C", "L P", "L L P", "SA RL", "S A R L",  # S.à r.l. -> SA RL
    "LIMITED PARTNERSHIP", "LIMITED LIABILITY", "UNLIMITED LIABILITY", "UNLTD LIABILITY",
    "CO", "COMPANY", "LIMITED", "SA", "SAS", "SPA", "SRL", "AG", "KG", "AB", "BV", "NV", "DAC", "PTY", "PTE", "OY", "ASA",
]
_ANYWHERE_RE = re.compile(r"(?<=\S) (?:" + "|".join(ANYWHERE_FORMS) + r")\b")
_END_RE = re.compile(r"(?<=\S) (?:" + "|".join(sorted(END_FORMS + ANYWHERE_FORMS, key=len, reverse=True)) + r")$")


def normalize_borrower_name(name: str) -> str:
    """'Project Alpha Intermediate Holding, Inc.' -> 'PROJECT ALPHA INTERMEDIATE HOLDING'. Keeps the first word."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"[.'`]", "", s)  # J.P. -> JP, Wings 'N More -> WINGS N MORE
    s = " ".join(re.sub(r"[^A-Z0-9]+", " ", s).split())
    s = _ANYWHERE_RE.sub("", s)
    while (shorter := _END_RE.sub("", s)) != s:  # "GMBH & CO KG", "PTY LTD"
        s = shorter
    return s


# ---------------------------------------------------------------------------
# Copied verbatim from get_seniority_debttype.py
# ---------------------------------------------------------------------------
def classify_seniority_v2(text: str) -> str:
    if text is None:
        return "OTHER_UNKNOWN"

    t = text.lower()

    # CLO / structured equity
    if "collateralized loan obligation" in t or "clo subordinated" in t or "clo equity" in t:
        return "CLO_STRUCTURED_EQUITY"
    if "structured finance" in t and "membership interests" in t:
        return "CLO_STRUCTURED_EQUITY"

    # Subordinated / mezz / junior / holdco
    if "mezz" in t or "subordinated" in t or "junior debt" in t or "junior notes" in t:
        return "SUBORDINATED_MEZZ"
    if "holdco notes" in t or "holdco loan" in t or "holdco debt" in t:
        return "SUBORDINATED_MEZZ"
    if "pik toggle notes" in t:
        return "SUBORDINATED_MEZZ"

    # Unitranche / FO-LO
    if "unitranche" in t or "first out last out" in t or "first out / last out" in t or "fo/lo" in t:
        return "UNITRANCHE"

    # Second lien
    if "second lien" in t or "2nd lien" in t or "second-lien" in t or "junior secured" in t:
        return "SENIOR_SECURED_2L"

    # >>> UNSECURED BEFORE FIRST-LIEN/SECURED <<<
    if "unsecured" in t:
        return "SENIOR_UNSECURED"
    if ("senior notes" in t or "senior debt" in t) and "secured" not in t:
        return "SENIOR_UNSECURED"

    # First lien explicitly
    if "first lien" in t or "1st lien" in t:
        return "SENIOR_SECURED_1L"

    # Secured but lien not specified → separate bucket
    if (
        "secured" in t
        and "first lien" not in t
        and "1st lien" not in t
        and "second lien" not in t
        and "2nd lien" not in t
        and "junior" not in t
        and "mezz" not in t
    ):
        return "SENIOR_SECURED_UNSPEC"

    return "OTHER_UNKNOWN"


def inv_clean_expr() -> pl.Expr:
    """inv_clean: lowercased, alphanumeric + spaces, for regex sector rules"""
    return (
        pl.col("investment_identifier")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.strip_chars()
        .alias("inv_clean")
    )


def sector_rule_expr() -> pl.Expr:
    """
    Rules-based classifier using inv_clean with a big set of keyword patterns.
    Returns a Polars expression that produces a 'sector_rule' column.
    """
    c = pl.col("inv_clean")

    # Structured / CLO etc. (will go into Financials & Insurance)
    structured_finance_pat = r"(collateralized loan obligation|clo subordinated notes|structured finance)"

    # Healthcare / Pharma
    healthcare_pat = (
        r"\b(healthcare|health care|clinic|clinics?|medical|hospital|hospitals|"
        r"dental|vision|pharma|pharmaceuticals?|biotech|biosciences?|"
        r"imaging|fertility|rx\b|pharmacy)\b"
    )

    # Technology / Software
    tech_pat = (
        r"\b(software|saas|technology|technologies|it services?|cyber|"
        r"cloud|data center|digital)\b"
    )

    # Financials / lenders / funds / banks / insurance / structured
    financials_ins_pat = (
        r"\b(bank|banking|financial|finance|funding|fund\b|funds\b|"
        r"capital partners|capital management|asset management|"
        r"insurance|brokerage)"
    )

    # Consumer staples (food, groceries ...)
    consumer_staples_pat = (
        r"\b(food|foods|beverage|beverages|brew|bakery|baking|"
        r"tobacco|grocery|grocer|supermarket|convenience store)\b"
    )

    # Consumer discretionary (restaurants, leisure, auto, retail ...)
    consumer_disc_pat = (
        r"\b(restaurant|restaurants|cafe|caf[eé]|hotel|resort|casino|gaming|"
        r"leisure|fitness|gym|auto dealer|car wash|cinema|theater|theatre|"
        r"retail|retailer|mall|shopping)\b"
    )

    # Extra catch-all for retail / wholesale
    retail_wholesale_pat = (
        r"\b(retail|retailer|wholesale|wholesaler|distributor)\b"
    )

    # Education (we fold into Consumer)
    education_pat = (
        r"\b(education|educational|school|university|college|learning|training)\b"
    )

    # Industrials / manufacturing / logistics
    industrials_pat = (
        r"\b(industrial|manufactur|factory|plant|engineering|equipment|"
        r"logistics|distribution|warehouse|trucking|freight|railroad|railway)\b"
    )

    # Transportation (airlines, shipping, etc.)
    transportation_pat = (
        r"\b(airline|airlines|airways|shipping|shipper|marine transport|"
        r"fleet services|logistics carrier)\b"
    )

    # Energy + Utilities
    energy_utils_pat = (
        r"\b(energy|power|electricity|solar|wind|renewable|"
        r"pipeline|oil\b|gas\b|utilities|water utility|electric utility|gas utility)\b"
    )

    # Materials / chemicals / packaging
    materials_pat = (
        r"\b(materials?|chemicals?|plastics?|rubber|packaging)\b"
    )

    # Real Estate
    real_estate_pat = (
        r"\b(real estate|properties|property|reit|apartment|apartments|"
        r"office park|logistics park|industrial park)\b"
    )

    # Communication + Media
    comms_media_pat = (
        r"\b(telecom|telecommunications?|cable tv|wireless carrier|"
        r"media|advertising|marketing agency|"
        r"entertainment|streaming|content studio|film studio|music label)\b"
    )

    # NOTE: order matters → more specific patterns first, then broader ones
    return (
        # Structured goes into Financials & Insurance
        pl.when(c.str.contains(structured_finance_pat))
        .then(pl.lit("Financials & Insurance"))

        .when(c.str.contains(healthcare_pat))
        .then(pl.lit("Healthcare"))

        .when(c.str.contains(tech_pat))
        .then(pl.lit("Technology"))

        .when(c.str.contains(financials_ins_pat))
        .then(pl.lit("Financials & Insurance"))

        .when(c.str.contains(transportation_pat))
        .then(pl.lit("Transportation"))

        # All consumer-like stuff in a single bucket
        .when(
            c.str.contains(consumer_staples_pat)
            | c.str.contains(consumer_disc_pat)
            | c.str.contains(retail_wholesale_pat)
            | c.str.contains(education_pat)
        )
        .then(pl.lit("Consumer"))

        .when(c.str.contains(industrials_pat))
        .then(pl.lit("Industrials"))

        .when(c.str.contains(energy_utils_pat))
        .then(pl.lit("Energy & Utilities"))

        .when(c.str.contains(materials_pat))
        .then(pl.lit("Materials"))

        .when(c.str.contains(real_estate_pat))
        .then(pl.lit("Real Estate"))

        .when(c.str.contains(comms_media_pat))
        .then(pl.lit("Communication & Media"))

        # Fallback: goes directly to Other / Unknown
        .otherwise(pl.lit("Other / Unknown"))
        .alias("sector_rule")
    )
# ---------------------------------------------------------------------------


def read_text_csv(path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kw)


def classify_identifiers(identifiers: pd.Series) -> pd.DataFrame:
    """instrument_seniority, sector and sector_source for each unique investment_identifier."""
    labels = (
        pl.DataFrame({"investment_identifier": identifiers.drop_duplicates().tolist()})
        .with_columns(
            pl.col("investment_identifier")
            .map_elements(classify_seniority_v2, return_dtype=pl.String)
            .alias("instrument_seniority"),
            inv_clean_expr(),
        )
        .with_columns(sector_rule_expr())
        .to_pandas()
        .set_index("investment_identifier")
    )

    llm = read_text_csv(LLM_CHECKPOINT_CSV)
    assert list(llm.columns) == ["borrower_name", "sector_llm", "sector_llm_conf"], list(llm.columns)
    assert not llm["borrower_name"].duplicated().any(), "LLM checkpoint has duplicate names"
    llm = llm.set_index("borrower_name")
    sector_llm = labels.index.map(llm["sector_llm"])
    conf = labels.index.map(llm["sector_llm_conf"].astype(float))

    by_rules = labels["sector_rule"].ne("Other / Unknown").to_numpy()
    by_llm = ~by_rules & pd.notna(sector_llm) & (conf >= CONF_THRESH)
    labels["sector"] = np.where(by_rules, labels["sector_rule"], np.where(by_llm, sector_llm, UNRESOLVED))
    labels["sector_source"] = np.select(
        [by_rules, by_llm, pd.isna(sector_llm)], ["rules", "llm", "not_in_checkpoint"], "llm_low_conf"
    )
    return labels[["instrument_seniority", "sector", "sector_source"]]


def load_borrower_lookup() -> pd.Series:
    # openpyxl refuses a .csv filename, so hand it the open file instead.
    with open(BORROWER_XLSX, "rb") as f:
        lookup = pd.read_excel(f, engine="openpyxl", dtype=str, keep_default_na=False)
    assert list(lookup.columns) == ["borrower_name_original", "borrower_canonical"], list(lookup.columns)
    assert not lookup["borrower_name_original"].duplicated().any(), "borrower lookup has duplicate identifiers"
    canonical = lookup.set_index("borrower_name_original")["borrower_canonical"]
    return canonical.map({name: normalize_borrower_name(name) for name in canonical.unique()})


def build() -> dict:
    labels = classify_identifiers(read_text_csv(BASE_CSV, usecols=["investment_identifier"])["investment_identifier"])
    borrower = load_borrower_lookup()

    partial = OUTPUT_CSV.with_suffix(".csv.partial")
    n_rows = 0
    sources = pd.Series(dtype=int)
    for chunk in read_text_csv(BASE_CSV, chunksize=CHUNK_ROWS):
        ids = chunk["investment_identifier"]
        chunk.insert(chunk.columns.get_loc("investment_identifier") + 1, "borrower_canonical", ids.map(borrower).fillna(""))
        chunk["instrument_seniority"] = ids.map(labels["instrument_seniority"])
        chunk["sector"] = ids.map(labels["sector"])
        sources = sources.add(ids.map(labels["sector_source"]).value_counts(), fill_value=0)
        chunk = chunk.drop(columns=DROP_COLUMNS)  # KeyError on a misspelt name
        chunk.to_csv(partial, mode="w" if n_rows == 0 else "a", header=n_rows == 0, index=False)
        n_rows += len(chunk)

    # Re-read both files side by side: kept panel columns unchanged, row order unchanged.
    n_check = n_with_borrower = n_with_seniority = n_with_sector = 0
    with read_text_csv(BASE_CSV, chunksize=CHUNK_ROWS) as base_reader, read_text_csv(partial, chunksize=CHUNK_ROWS) as out_reader:
        for base_chunk, out_chunk in zip(base_reader, out_reader):
            kept = base_chunk.columns.difference(DROP_COLUMNS, sort=False)
            assert out_chunk[kept].equals(base_chunk[kept]), f"panel columns changed near row {n_check}"
            n_check += len(out_chunk)
            n_with_borrower += out_chunk["borrower_canonical"].ne("").sum()
            n_with_seniority += out_chunk["instrument_seniority"].ne("OTHER_UNKNOWN").sum()
            n_with_sector += (~out_chunk["sector"].isin([UNRESOLVED, "Other / Unknown"])).sum()
        assert n_check == n_rows and next(out_reader, None) is None, "output row count differs from the panel"
    partial.replace(OUTPUT_CSV)  # readers must be closed first on Windows

    return {
        "output": str(OUTPUT_CSV),
        "size_mb": round(OUTPUT_CSV.stat().st_size / 1e6, 1),
        "rows": n_rows,
        "columns": len(out_chunk.columns),
        "borrower_rows_matched": f"{n_with_borrower:,} ({n_with_borrower / n_rows:.2%})",
        "seniority_rows_identified": f"{n_with_seniority:,} ({n_with_seniority / n_rows:.2%})",
        "sector_rows_identified": f"{n_with_sector:,} ({n_with_sector / n_rows:.2%})",
        "sector_rows_by_source": {k: int(v) for k, v in sources.items()},
    }


if __name__ == "__main__":
    for k, v in build().items():
        print(f"{k}: {v}")
