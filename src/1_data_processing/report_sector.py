"""sector_report: the validated sector with Software split out of Technology and Business Services split
out of Industrials (Financing the Economy 2026, Fig 18), by keyword rules on each identifier's reported
industry text, one value per borrower. Rules in docs/6_final_dataset.md."""
from __future__ import annotations

import csv
import re
from collections import defaultdict

SOFTWARE = "Software"
TECH_OTHER = "Technology (ex-software)"
BUSINESS_SERVICES = "Business Services"
INDUSTRIALS_OTHER = "Industrials (ex-business services)"
SPLIT_SECTORS = ("Technology", "Industrials")

_SUFFIX = re.compile(r"\s*\((?:\d+ of \d+ BDCs|sibling rows|other BDCs?|related entity)\)", re.I)
_SOFTWARE = re.compile(r"software|\bsaa?s\b|application|\berp\b|\bcrm\b|cyber", re.I)
_BUSINESS = re.compile(
    r"professional|commercial (?:and professional )?services|commercial services|business|consult|"
    r"support services|staffing|human resource|\bhr\b|outsourc|marketing|advertis|research|training|"
    r"testing|inspection|certification|environmental and facilities|facilit(?:y|ies) services|"
    r"security services|information services|data and analytics|payroll|benefits|document|printing|"
    r"call center|\bbpo\b|engineering services|legal services|accounting|compliance", re.I)
_NOT_BUSINESS = re.compile(r"energy equipment|oilfield|freight|logistics|transport|airline|marine|aerospace|"
                           r"construction and engineering", re.I)


def identifier_label(sector: str, evidence: str | None) -> str:
    """Report sector of one identifier from its validated sector and its industry text."""
    ev = _SUFFIX.sub("", evidence or "")
    if sector == "Technology":
        return SOFTWARE if _SOFTWARE.search(ev) else TECH_OTHER
    if sector == "Industrials":
        return BUSINESS_SERVICES if _BUSINESS.search(ev) and not _NOT_BUSINESS.search(ev) else INDUSTRIALS_OTHER
    return sector


def report_sectors(labels_csv) -> dict[str, str]:
    """investment_identifier -> sector_report, from labels_by_identifier.csv (borrower vote
    weighted by rows_in_file)."""
    csv.field_size_limit(10**9)
    own, borrower_of, votes = {}, {}, defaultdict(lambda: defaultdict(int))
    with open(labels_csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            ident, borrower = r["investment_identifier"], r["borrower_canonical"]
            label = identifier_label(r["sector"], r["sector_evidence"])
            own[ident] = label
            if borrower and r["sector"] in SPLIT_SECTORS:
                borrower_of[ident] = borrower
                votes[(borrower, r["sector"])][label] += int(r["rows_in_file"] or 1)
    winner = {k: sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for k, v in votes.items()}
    out = {}
    for ident, label in own.items():
        b = borrower_of.get(ident)
        sector = {SOFTWARE: "Technology", TECH_OTHER: "Technology",
                  BUSINESS_SERVICES: "Industrials", INDUSTRIALS_OTHER: "Industrials"}.get(label)
        out[ident] = winner[(b, sector)] if b and sector else label
    return out
