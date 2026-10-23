"""BDC -> manager family.

Funds run by the same asset manager share one valuation process, so for crowding and
valuation dispersion they count as one holder (Risk Dimensions doc, section 0.1).

    family_map()           {cik: family} for every BDC in BDC_filings_2001_present.csv
    add_family(df)         adds a `family` column (pandas or polars)
    manager(cik, name)     family of one BDC from its name

A BDC that matches no rule is its own family (its latest filing name).
"""
import re

import pandas as pd

from paths import UNIVERSE_DIR

# (regex on any of the BDC's filing names, family); first match wins
FAMILY_RULES = [
    (r"apollo", "Apollo"),
    (r"\bares\b", "Ares"),
    (r"blackstone", "Blackstone"),
    (r"\bowl\b", "Blue Owl"),
    (r"goldman|phillip street", "Goldman Sachs"),
    (r"\bkkr\b|^fs investment", "FS KKR"),
    (r"main street|^hms income", "Main Street"),
    (r"\bhps\b", "HPS"),
    (r"carlyle|^tcg bdc", "Carlyle"),
    (r"north haven|^sl investment|morgan stanley|^t series middle market", "Morgan Stanley"),
    (r"bain capital", "Bain Capital"),
    (r"blackrock", "BlackRock"),
    (r"antares", "Antares"),
    (r"barings", "Barings"),
    (r"lord abbett", "Lord Abbett"),
    (r"new mountain|^nmf slf", "New Mountain"),
    (r"monroe capital", "Monroe Capital"),
    (r"golub", "Golub"),
    (r"^ab private", "AllianceBernstein"),
    (r"audax", "Audax"),
    (r"nuveen churchill", "Nuveen Churchill"),
    # added 2026-10-23 for family-level crowding and dispersion
    (r"oaktree|^fifth street", "Oaktree"),
    (r"twin brook", "TPG Angelo Gordon (Twin Brook)"),
    (r"sixth street|^tpg specialty lending", "Sixth Street"),     # TSLX filed as TPG Specialty Lending
    (r"\boha\b", "OHA (T. Rowe Price)"),
    (r"^crescent", "Crescent"),
    (r"^fidelity", "Fidelity"),
    (r"franklin bsp|^business development corp of america", "Franklin BSP"),
    (r"^gladstone", "Gladstone"),
    (r"^kayne", "Kayne Anderson"),
    (r"^muzinich", "Muzinich"),
    (r"pennantpark", "PennantPark"),
    (r"^silver point", "Silver Point"),
    (r"^slr\b|^solar capital|^scp private credit", "SLR"),
    (r"^stellus", "Stellus"),
    (r"^stone point", "Stone Point"),
    (r"^tcw\b", "TCW"),
    (r"^triplepoint", "TriplePoint"),
    (r"^venture lending & leasing|^wti fund", "Western Technology Investment"),
    (r"^investcorp|^cm finance", "Investcorp"),               # CM Finance -> Investcorp Credit Mgmt BDC
    (r"^bc partners|^kohlberg capital|portman ridge|logan ridge|^capitala", "BC Partners"),
    (r"^hancock park|^ofs capital", "OFS"),
    (r"^prospect|^triton pacific", "Prospect"),               # Triton Pacific -> Prospect Flexible Income
]
# BDCs whose filing names don't show the manager
CIK_FAMILIES = {
    1414932: "Oaktree",     # "Fifth Street Finance Corp" -> Oaktree Specialty Lending (OCSL)
    1379785: "Barings",     # "Triangle Capital Corp" -> Barings BDC (BBDC)
    1370755: "BlackRock",   # "TCP Capital Corp" -> BlackRock TCP Capital (TCPC)
}


def _family(cik, names):
    if cik in CIK_FAMILIES:
        return CIK_FAMILIES[cik]
    for pat, fam in FAMILY_RULES:
        if any(re.search(pat, n, re.I) for n in names):
            return fam
    return names[-1] if names else str(cik)


def manager(cik, bdc_name):
    """Family of one BDC from a single name; a BDC that matches no rule is its own family."""
    return _family(cik, [bdc_name] if isinstance(bdc_name, str) else [])


def family_map() -> dict:
    """{cik: family} for every BDC in the filing index, matching rules against all its names."""
    idx = pd.read_csv(UNIVERSE_DIR / "BDC_filings_2001_present.csv", dtype=str, usecols=["CIK", "Company", "Date"])
    idx["cik"] = idx["CIK"].astype(int)
    names = idx.sort_values("Date").groupby("cik")["Company"].agg(lambda s: list(pd.unique(s.str.strip())))
    return {cik: _family(cik, ns) for cik, ns in names.items()}


def add_family(df, cik_col: str = "cik", out_col: str = "family"):
    """Add the manager family of each row's BDC. Works on pandas and polars frames."""
    fmap = family_map()
    if isinstance(df, pd.DataFrame):
        ciks = pd.to_numeric(df[cik_col], errors="coerce")
        out = df.copy()
        out[out_col] = [fmap.get(int(c), str(int(c))) if pd.notna(c) else None for c in ciks]
        return out
    import polars as pl
    keys = df.select(pl.col(cik_col).cast(pl.Int64).unique()).to_series().to_list()
    lut = pl.DataFrame({"_cik": keys, out_col: [fmap.get(k, str(k)) for k in keys]},
                       schema={"_cik": pl.Int64, out_col: pl.Utf8})
    return (df.with_columns(pl.col(cik_col).cast(pl.Int64).alias("_cik"))
              .join(lut, on="_cik", how="left").drop("_cik"))
