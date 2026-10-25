"""Report-aligned sector (`sector_report`): the validated sector with Software split out of
Technology and Business Services split out of Industrials.

Why: Houlihan Lokey's BDC Monitor weights in Financing the Economy 2026 (Fig 18, p.27) are
business services 25%, software 21% and industrials 17%. The validated 12-sector taxonomy has a
single Technology bucket and puts business services inside Industrials, so Fig 18 cannot be
compared without this split.

How: no new labelling. Each validated identifier already carries the industry text its BDC
reports (or other BDCs report) next to the borrower, in `sector_evidence`. Keyword rules on that
text split the two sectors:

    Technology  -> "Software" if the text names software (software, SaaS, application, ERP, ...),
                   else "Technology (ex-software)" (IT services, hardware, internet, generic
                   "high tech")
    Industrials -> "Business Services" if the text names professional / commercial / business
                   services, consulting, staffing, HR, research, testing and similar, unless it
                   names a goods or transport industry; else "Industrials (ex-business services)"
    other sectors are unchanged

Then one value per borrower, as for `sector`: each identifier of the borrower votes with its row
count and the plurality wins (ties: alphabetical), so a borrower one BDC files as "software" and
another as "high tech industries" gets a single label. Identifiers with no borrower keep their
own label. Rows whose sector is not Technology or Industrials are never changed, so
`sector_report` rolls back up to `sector` exactly.

IT services stay in Technology (ex-software). The report does not say where its taxonomy puts
them; moving them to Business Services changes that share by about 1.5 points.
"""
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
