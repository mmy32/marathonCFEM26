"""Tag data-center-related borrowers in the BDC iXBRL schedule-of-investments file.

Usage:
    python classify_data_center_borrowers.py [input.csv] [out_dir]

Defaults: input data/final_dataset/FINAL_DATASET.csv, out_dir data/data_center. Writes:
    data_center_borrowers.csv   one row per borrower-name variant (entity, tier, confidence, note)
    data_center_loan_rows.csv   every matched investment row, all original columns + tags

Matching is a curated alias list (regex on the borrower name: `borrower_canonical`, or
`borrower` in older files) plus an exclusion list of known homonyms. Tiers:
    1 Operator / developer              owns or develops data center real estate / capacity
    2 AI compute / GPU cloud / miners   GPU clouds, GPU-lease SPVs, AI labs with own clusters, miners
    (tier 3, managed hosting, was removed on 2026-10-20: see REVIEWED_NOT_DC)
    4 DC services / supply chain        maintenance, power, cooling, electrical equipment

Each match also gets a scope, which nests into three estimates:
    core      tiers 1-2
    support   tier 4 where data centers are the firm's main market
    grey      GREY_AREA (any tier): suppliers for which data centers are one of several
              markets, and operators whose data center business is a sideline
core + support is the headline (conservative) estimate; adding grey gives the upper bound.

Inclusion test (user, 2026-10-20): would the borrower's absence, with no substitute, significantly
impair the physical operation of data centers (power, cooling, mechanical, facilities)? Hosting,
cloud and IT services delivered from data centers, and IT hardware makers, fail it.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py

import os
import sys

import pandas as pd

from paths import DATA_CENTER_DIR, FINAL_DIR

# (tier, entity, borrower regex, confidence, note)
RULES = [
    # --- Tier 1: operators / developers ---------------------------------------------
    ("1 Operator / developer", "Vantage Data Centers", r"vantage data centers", "high", ""),
    ("1 Operator / developer", "CyrusOne", r"cyrusone", "high", "Revolving warehouse facility"),
    ("1 Operator / developer", "QTS", r"^qts\b|^qualitytech|qts project ram", "high",
     'Identifier: "Apterra Project Ram Microsoft DC" mortgage loan; QualityTech, LP is QTS\'s operating partnership'),
    ("1 Operator / developer", "STACK Infrastructure", r"^stack sj holdings", "high",
     "STACK San Jose (SVY01) campus"),
    ("1 Operator / developer", "EdgeConneX", r"edgeconnex", "high", ""),
    ("1 Operator / developer", "Flexential (ABS issuer)", r"flexential", "high", ""),
    ("1 Operator / developer", "TierPoint", r"tierpoint", "high", ""),
    ("1 Operator / developer", "DC Blox", r"dc ?blox", "high", ""),
    ("1 Operator / developer", "DartPoints", r"dartpoints", "high", ""),
    ("1 Operator / developer", "US Signal", r"us signal", "high", ""),
    ("1 Operator / developer", "Ark Data Centers", r"ark data cent", "high", ""),
    ("1 Operator / developer", "Hyscaleix Data Centers", r"hyscaleix", "high", ""),
    ("1 Operator / developer", "North America Data Center Acq.", r"north america data center|^data center acquisition$",
     "high", 'Some rows carry the canonical name without "North America"'),
    ("1 Operator / developer", "Galaxy Helios (AI campus)", r"galaxy helios", "high",
     "Galaxy Digital's Helios AI/HPC data center campus"),
    ("1 Operator / developer", "Colo Holdings", r"^colo holdings|^chv holdings", "high",
     "Colocation operator; 18-21.5% PIK term loan (CHV Holdings LLC is the same credit)"),
    ("1 Operator / developer", "Internap (INAP)", r"internap", "high", ""),
    ("1 Operator / developer", "PowerHouse Data", r"powerhouse data", "high", "PowerHouse Data Centers"),
    ("1 Operator / developer", "Iron Mountain Manassas", r"iron mountain manassas", "high",
     "Iron Mountain data center campus (Manassas, VA)"),
    ("1 Operator / developer", "Switch (ABS / HoldCo)", r"^switch (?:bbf|master holdco)", "high",
     "Switch Inc. data centers; asset-backed structure"),
    ("1 Operator / developer", "Yucca Growth Infrastructure (Oracle)", r"yucca growth infra", "high",
     'Oracle data center build SPV (identifier names "Oracle Corp")'),
    ("1 Operator / developer", "5C Data Centers (CMH01)", r"^cmh01 holdings", "high",
     "5C Group's Columbus, OH hyperscale campus"),
    ("1 Operator / developer", "Chirisa Technology Parks (Volo)", r"^chirisa", "high",
     "Chirisa's Volo, IL data center campus"),
    ("1 Operator / developer", "Blackstone data center CMBS (BX VOLT)", r"^bx 20\d\d (?:volt|vlt)", "high",
     "Single-borrower CMBS on Blackstone data centers (2021-VOLT: 10 data centers; 2025-VLT6: 4, QTS design)"),
    ("1 Operator / developer", "Related Digital Michigan (Oracle)", r"^rd michigan property owner", "medium",
     "Likely Related Digital's Saline Township, MI campus for Oracle (financed Apr 2026)"),
    ("1 Operator / developer", "Yondr (Loudoun County HPC campus)", r"^yndrdc|^yondr", "high",
     "Yondr JK 1 6.875% notes due 2031, financing an HPC data center in Loudoun County, VA (June 2026)"),
    ("1 Operator / developer", "JPMCC 2022-DATA (CMBS)", r"^jpmcc 2022 data", "high",
     "CMBS; the holding fund (BREC) lists it under property type 'Data center'"),
    # operators whose data center business is a sideline (GREY_AREA)
    ("1 Operator / developer", "FirstDigital Communications", r"^firstdigital communications", "medium",
     "Utah telecom / fiber provider that also runs colocation data centers"),
    ("1 Operator / developer", "123.Net", r"^123 ?net$", "medium",
     "Michigan fiber provider with carrier-hotel colocation"),
    ("1 Operator / developer", "Iron Mountain (corporate)", r"^iron mountain(?: information management)?$", "medium",
     "Records-management REIT with a data center segment; the Manassas campus loan is a separate core entity"),
    ("1 Operator / developer", "Long Ridge Energy", r"^long ridge energy", "medium",
     "Ohio gas power plant developing a co-located data center campus"),
    # --- Tier 2: AI compute / GPU cloud / miners -----------------------------------
    ("2 AI compute / GPU cloud / miners", "CoreWeave (incl. SPVs)", r"coreweave", "high", ""),
    ("2 AI compute / GPU cloud / miners", "CoreWeave (incl. SPVs)", r"^cw nest property owner", "medium",
     "Likely the owner of CoreWeave's NEST data center, Kenilworth, NJ (bought Aug 2025); held by Ares Core Infrastructure Fund"),
    ("2 AI compute / GPU cloud / miners", "Valor Compute Infrastructure (xAI GPUs)", r"vci asset holdings", "high",
     "Valor VCI: NVIDIA GPUs triple-net leased to xAI (Apollo-led financing)"),
    ("2 AI compute / GPU cloud / miners", "xAI", r"^x\.?ai\b", "high", "AI lab with owned Colossus clusters"),
    ("2 AI compute / GPU cloud / miners", "Nscale", r"nscale", "high", ""),
    ("2 AI compute / GPU cloud / miners", "Firmus Supercloud", r"firmus", "high", ""),
    ("2 AI compute / GPU cloud / miners", "Together Computer", r"together computer", "high", ""),
    ("2 AI compute / GPU cloud / miners", "Cirrascale Cloud", r"cirrascale", "high", ""),
    ("2 AI compute / GPU cloud / miners", "Applied Digital", r"applied digital", "high", ""),
    ("2 AI compute / GPU cloud / miners", "Crusoe Energy", r"crusoe energy", "high", ""),
    ("2 AI compute / GPU cloud / miners", "Core Scientific", r"core scientific", "high", ""),
    ("2 AI compute / GPU cloud / miners", "Hut 8", r"hut 8", "high", ""),
    ("2 AI compute / GPU cloud / miners", "CleanSpark", r"cleanspark", "high", ""),
    # --- Tier 4: services / supply chain -----------------------------------------------
    ("4 DC services / supply chain", "Park Place Technologies", r"park place tech|^ppt holdings iii", "high",
     "Third-party DC hardware maintenance"),
    ("4 DC services / supply chain", "Service Express", r"service express|^victors purchaser", "high",
     "Third-party DC hardware maintenance; legal borrower Victors Purchaser, LLC (merged with Park Place, Jan 2026)"),
    ("4 DC services / supply chain", "Computer Data Source", r"^computer data source", "high",
     "Third-party DC hardware maintenance"),
    ("4 DC services / supply chain", "Salute Mission Critical", r"salute mission|^salute$", "high", "DC operations staffing"),
    ("4 DC services / supply chain", "C3 AcquisitionCo", r"^c3 acquisitionco", "high",
     'One BDC tags it "Data Center / Other Commercial Services", another "Construction & Engineering"'),
    ("4 DC services / supply chain", "Accelevation", r"accelevation", "high", "DC containment / infrastructure products"),
    ("4 DC services / supply chain", "VoltaGrid", r"voltagrid", "high", "On-site gas power for AI data centers (Oracle 2.3 GW)"),
    ("4 DC services / supply chain", "IEM (Industrial Electric Mfg)",
     r"^iem(?: new sub|$)|^industrial electric manufacturing", "high",
     "Switchgear maker serving data centers and utilities"),
    ("4 DC services / supply chain", "LayerZero Power Systems", r"^griffon bidco", "high",
     "Static transfer switches / PDUs for data centers"),
    ("4 DC services / supply chain", "Maverick Power", r"^maverick power", "high",
     "Switchgear / power distribution for data centers"),
    ("4 DC services / supply chain", "Forgent Power", r"^forgent", "high",
     "Transformers, switchboards and PDUs for data centers"),
    ("4 DC services / supply chain", "Mission Critical Group", r"^mission critical group", "high",
     "Power/electrical equipment for data centers"),
    ("4 DC services / supply chain", "Advanced Cooling Technologies", r"advanced cooling tech", "high",
     "Thermal management supplier"),
    ("4 DC services / supply chain", "Vertiv", r"vertiv|^cortes np acquisition", "high",
     "Cortes NP Acquisition Corp. was Vertiv's borrower entity"),
    ("4 DC services / supply chain", "Uptime Institute", r"^uptime institute", "high",
     "Data center tier certification / advisory"),
    ("4 DC services / supply chain", "Electro Technical Industries", r"^electro technical industries", "medium",
     "Mission-critical electrical distribution / switchgear; data centers are one of several end markets"),
    ("4 DC services / supply chain", "Trystar", r"^trystar", "medium",
     "Power distribution equipment; data centers are one of several end markets"),
    ("4 DC services / supply chain", "MGM Transformer", r"^mgm transformer", "medium",
     "Transformers; data centers are one of several end markets"),
    ("4 DC services / supply chain", "Boyd Corp", r"^boyd$", "medium",
     "Engineered materials and thermal management (incl. data center liquid cooling)"),
    # suppliers flagged by the chat screen, reviewed 2026-10-20: data centers are one market among several
    # (all GREY_AREA)
    ("4 DC services / supply chain", "Resa Power", r"^resa power", "medium",
     "Electrical power system testing, maintenance and repair"),
    ("4 DC services / supply chain", "Total Power (generators)", r"^total power|^generator (?:us )?buyer", "medium",
     "Generator sales, rental and service; Generator (US) Buyer is the acquisition vehicle"),
    ("4 DC services / supply chain", "Legence", r"^legence|^refficiency", "medium",
     "Mechanical / energy engineering and services; Refficiency is the former name"),
    ("4 DC services / supply chain", "Rehlko (Kohler Energy)", r"^kohler energy|^rehlko", "medium",
     "Backup generators and power systems"),
    ("4 DC services / supply chain", "Modigent", r"^modigent", "medium", "Mechanical contractor (mission-critical work)"),
    ("4 DC services / supply chain", "Shermco", r"^shermco", "medium", "Electrical testing and maintenance"),
    ("4 DC services / supply chain", "Aggreko", r"^aggreko", "medium", "Temporary power and cooling rental"),
    ("4 DC services / supply chain", "INNIO", r"^innio", "medium", "Gas engines used for on-site power"),
    ("4 DC services / supply chain", "GenServe", r"^genserve", "medium", "Standby generator service"),
    ("4 DC services / supply chain", "Salas O'Brien", r"^salas obrien", "medium",
     "Engineering firm with a mission-critical design practice"),
    ("4 DC services / supply chain", "C&D Technologies", r"^c d technologies|^c&d technologies", "medium",
     "UPS / stationary batteries"),
    ("4 DC services / supply chain", "viaPhoton", r"^viaphoton", "medium", "Fiber connectivity products"),
    ("4 DC services / supply chain", "Last Energy", r"^last energy", "medium",
     "Micro-reactor developer targeting data center power"),
]

# Upper-bound layer: multi-market suppliers and operators with a data center sideline.
# Maverick Power is support, not grey: nVent's 2026 acquisition release describes it as a power
# distribution provider for data centers. C3 AcquisitionCo stays: Sound Point Direct Lending BDC
# files it under "Data Center / Other Commercial Services".
# Grey vs support re-checked case by case on 2026-10-20: support needs evidence that data centers are
# the firm's largest market. Moved to support: Forgent Power (S-1: data centers 42% of FY2025 revenue,
# the largest end market, 47% of backlog) and Last Energy (half its contracted microreactors and its
# Texas project serve data centers). The rest stay grey: their owners or filings list data centers as
# one of several end markets and no revenue split is published (IEM, Trystar, Resa, ACT, ETI, Rehlko,
# Legence, Modigent, ...), or they are operators whose data center business is a sideline.
GREY_AREA = {
    "IEM (Industrial Electric Mfg)", "Advanced Cooling Technologies",
    "C3 AcquisitionCo", "Electro Technical Industries", "Trystar", "MGM Transformer", "Boyd Corp",
    "Resa Power", "Total Power (generators)", "Legence", "Rehlko (Kohler Energy)", "Modigent", "Shermco", "Aggreko",
    "INNIO", "GenServe", "Salas O'Brien", "C&D Technologies", "viaPhoton",
    "FirstDigital Communications", "123.Net", "Iron Mountain (corporate)", "Long Ridge Energy",
}
assert GREY_AREA <= {r[1] for r in RULES}, GREY_AREA - {r[1] for r in RULES}

# Known homonyms that a looser keyword search would catch. Applied before RULES.
EXCLUDE = (
    r"vantage specialty|advantage|labvantage|vantage (?:bidco|insurance|mobility)"
    r"|lumen bidco|crusoe bidco|aidc intermediateco|hyperion refinance|x holdings"
    r"|stack sports|north haven stack|computer services|switchfly|electro switch"
    r"|cloud software group|helios service|summit interconnect|onepointone"
    # large opaque names reviewed and confirmed NOT data-center related
    r"|oak funding|covert holdco|helix ts|emergency power holdings|gft infrastructure"
)

# Reviewed-and-rejected large opaque borrowers (kept for the audit trail)
REVIEWED_NOT_DC = {
    "Oak Funding LLC": "BDC / fund financing vehicle",
    "Covert HoldCo, LP": "Waste-management / environmental services platform",
    "Helix TS, LLC": "Helix Technology Solutions, London business software / AI consulting",
    "Emergency Power Holdings, LLC": "Myers Emergency Power Systems",
    "GFT Infrastructure, Inc.": "Gannett Fleming TranSystems, general transportation / water / power engineering",
    # keyword hits ("AI infrastructure", "web hosting", "mission critical") reviewed 2026-10-19
    "FTAI Infrastructure": "Rail, terminals and power; the hit was the ticker prefix",
    "WebPros (Luxembourg / Holding / US Bidco)": "Hosting control-panel software (cPanel, Plesk), not hosting capacity",
    "Mission Critical Electronics, Inc.": "BDC-reported industry Capital Equipment; no data center link found",
    # flagged by the chat screen, reviewed 2026-10-20
    "CPI Buyer, LLC": "Antylia Scientific (Cole-Parmer) lab equipment; BDCs file it under electrical equipment",
    "Hyperion": "Hyperion Refinance (insurance broker), already in EXCLUDE; the screen matched the word",
    # unknowns checked in the holding BDCs' own filings and EDGAR full-text search, 2026-10-20
    "BCPE Nucleon (DE) SPV, LP": "Bain Capital vehicle holding its Nutanix stake (13D / Form 4 filings)",
    "Beacon Dc, Ltd.": "BXSL files it under Insurance; floating-rate loan, not Hut 8's Beacon Point notes",
    "BPPH2 Limited": "BCRED files it under Diversified Consumer Services (GBP loan)",
    "Cypress Ridge / Ridge Hill / Sterling Plaza": "Blackstone real estate credit fund senior loans listed among multifamily portfolios",
    "IP Operations II Investco / IP Operating Portfolio I": "HPS files them under Independent Power Producers",
    "PCP CW Aggregator Holdings II, L.P.": "HPS files it under Consumer Finance",
    "AlphaCoin LLC": "Twin Brook files it under Aerospace and defense",
    # screen candidates left off even the grey layer, reviewed 2026-10-20
    # unclear cases are left out, even of GREY_AREA (user's rule, 2026-10-20)
    "GoDaddy, Namecheap, Webcentral": "Unclear: domain-led registrars that also sell hosting",
    "nVent Electric (dba nVent Thermal)": "Brookfield's 2024-25 carve-out of nVent's heat-tracing business (Chemelex), "
                                          "industrial pipe heating, not the data center cooling line",
    "TTM Technologies": "Unclear: printed circuit boards, one step upstream of the hardware makers",
    # removed 2026-10-20 under the physical-operations test: they deliver hosting, cloud or IT
    # services from data centers (customers of the facilities, not inputs to running them)
    "Hostinger": "Web hosting; rents colocation space",
    "Rackspace": "Managed cloud and IT services",
    "Ensono": "Managed IT, hybrid cloud and mainframe services",
    "Navisite": "Managed cloud and IT services",
    "Newfold Digital (Bluehost, HostGator)": "Web hosting and domains",
    "team.blue": "European web hosting group",
    "Contabo": "VPS / dedicated server host",
    "UKFast (ANS)": "UK managed hosting and cloud",
    "Nexcess Digital (Liquid Web)": "Managed web hosting",
    "Recovery Point Systems": "Disaster recovery and managed hosting",
    "Sungard Availability Services": "Disaster recovery and managed hosting",
    "Thrive Networks": "Managed IT and cloud services",
    "Syntax Systems": "Managed cloud for ERP applications",
    "Options Technology": "Managed infrastructure for trading firms",
    "Summit Hosting": "Application hosting",
    "Wasabi Technologies": "Cloud object storage",
    "Backblaze": "Cloud storage",
    # IT hardware, not facility equipment
    "Celestica": "Electronics manufacturing (servers, switches)",
    "Sanmina": "Electronics manufacturing (racks, ZT Systems)",
}


def tag(names: pd.Series) -> pd.DataFrame:
    """tier, entity, confidence, note and scope for each borrower name (NA where no rule matches)."""
    b = names.fillna("")
    excluded = b.str.contains(EXCLUDE, case=False, regex=True)
    out = pd.DataFrame(pd.NA, index=names.index, columns=["tier", "entity", "confidence", "note", "scope"], dtype=object)
    for tier, entity, pat, conf, note in RULES:
        m = b.str.contains(pat, case=False, regex=True) & ~excluded & out["entity"].isna()
        scope = "grey" if entity in GREY_AREA else "core" if tier[0] in "12" else "support"
        out.loc[m, :] = [tier, entity, conf, note, scope]
    return out


def instrument_type(s: str) -> str:
    # XBRL member names end in "[Member]", which the " member" equity cue below would catch
    s = (s or "").lower().replace("[member]", "")
    if any(k in s for k in ("asset-backed", "asset backed", "structured", "warehouse", "class c", "certificate")):
        return "Structured / ABS / warehouse"
    if "mortgage" in s:
        return "Mortgage"
    if "equipment financ" in s or "equipment term loan" in s:
        return "Equipment financing"
    if any(k in s for k in ("subordinated", "second lien", "2nd lien", "mezz")):
        return "Subordinated / 2L"
    if any(k in s for k in ("preferred", "common", " units", "warrant", "equity interest", "class a", "member")):
        return "Equity"
    if any(k in s for k in ("first lien", "1st lien", "senior secured", "one stop", "unitranche",
                            "term loan", "revolv", "delayed draw", "ddtl", "secured")):
        return "First lien / senior"
    return "Unspecified"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else FINAL_DIR / "FINAL_DATASET.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DATA_CENTER_DIR
    df = pd.read_csv(src, low_memory=False)
    name = "borrower_canonical" if "borrower_canonical" in df.columns else "borrower"
    df = pd.concat([df, tag(df[name])], axis=1)
    hit = df[df["entity"].notna()].copy()
    hit["instrument_type"] = hit["investment_identifier"].map(instrument_type)
    hit["core"] = hit["tier"].str[0].isin(["1", "2"])

    latest_q = df["cal_q"].max()
    keys = ["scope", "tier", "entity", "confidence", "note", name] + (["borrower_key"] if "borrower_key" in df.columns else [])
    variants = (
        hit.groupby(keys, dropna=False)
        .agg(rows=("cik", "size"), bdcs=("cik", "nunique"), first_q=("cal_q", "min"), last_q=("cal_q", "max"))
        .reset_index()
    )
    fv = (hit[hit["cal_q"] == latest_q].groupby(name)["InvestmentOwnedAtFairValue_normalized"].sum() / 1e6)
    variants["fv_latest_q_usd_mm"] = variants[name].map(fv).fillna(0.0)
    variants = variants.sort_values(["tier", "entity", "fv_latest_q_usd_mm"], ascending=[True, True, False])

    os.makedirs(out_dir, exist_ok=True)
    variants.to_csv(os.path.join(out_dir, "data_center_borrowers.csv"), index=False, float_format="%.2f")
    hit.to_csv(os.path.join(out_dir, "data_center_loan_rows.csv"), index=False)
    print(f"{variants['entity'].nunique()} entities, {len(variants)} name variants, {len(hit)} loan rows "
          f"(latest quarter {latest_q}) -> {out_dir}")


if __name__ == "__main__":
    main()
