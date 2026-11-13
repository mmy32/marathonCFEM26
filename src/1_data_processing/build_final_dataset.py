"""Step 8: FINAL_DATASET.csv from the step 7 panel: validated borrower / seniority / sector labels, the
logged rate fixes, and the bdc_type, non_accrual, HTML maturity and sector_report columns (see
docs/6_final_dataset.md). Values are copied as text, so untouched fields are exactly the inputs.
Usage: python src/1_data_processing/build_final_dataset.py [output.csv]"""

import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py

from paths import CLEANED_DIR, ENRICHED_DIR, FINAL_DIR

ENRICHED_CSV = ENRICHED_DIR / "data_private_credit_FINAL_enriched_final.csv"
LABELS_CSV = FINAL_DIR / "labels_by_identifier.csv"
RATE_FIXES_CSV = CLEANED_DIR / "rate_swap_fixes_2026-10-19.csv"
OUTPUT_CSV = FINAL_DIR / "FINAL_DATASET.csv"

LABEL_COLUMNS = ["borrower_canonical", "instrument_seniority", "sector"]   # replaced in place
PREV_IN_LABELS = ["borrower_prev", "seniority_prev", "sector_prev"]        # same values, as the labels file names them
DECISION_COLUMNS = [  # appended after the *_prev columns; validation_id is the labels file's id
    "borrower_confidence", "borrower_action", "borrower_reason",
    "seniority_confidence", "seniority_evidence", "seniority_action",
    "sector_confidence", "sector_evidence_type", "sector_evidence", "sector_action",
]
RATE = "InvestmentInterestRate_normalized"
CASH = "InvestmentInterestRatePaidInCash_normalized"
SPREAD = "InvestmentBasisSpreadVariableRate_normalized"
MATURITY = "InvestmentMaturityDate"


def _read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _bdc_types():
    """(cik, cal_q) -> bdc_type, from data/reference/bdc_type.csv."""
    import pandas as pd
    from bdc_type import OUT_CSV, type_in_quarter
    table = pd.read_csv(OUT_CSV, parse_dates=["listed_from", "listed_to", "non_traded_from"]).set_index("cik")
    cache = {}

    def lookup(cik, cal_q):
        key = (cik, cal_q)
        if key not in cache:
            quarter_end = pd.Period(cal_q, "Q").end_time.normalize()
            cache[key] = type_in_quarter(table.loc[int(cik)], quarter_end)   # KeyError: run bdc_type.py
        return cache[key]
    return lookup


def _non_accrual():
    """(accession, context_id) -> '1' / '0' / '' (unknown), from the scanned XBRL footnotes."""
    import pandas as pd
    from nonaccrual_flags import load_flags
    keys = pd.read_csv(ENRICHED_CSV, usecols=["accession", "context_id"], dtype=str)
    pairs, known, _ = load_flags(keys)

    def lookup(accession, context_id):
        if (accession, context_id) in pairs:
            return "1"
        return "0" if accession in known else ""
    return lookup


def _html_maturities():
    """(accession, context_id) -> YYYY-MM-DD, from maturity_html.py's output."""
    from maturity_html import OUT_CSV as MATURITY_CSV   # missing file: run maturity_html.py extract
    return {(r["accession"], r["context_id"]): r["maturity_html"] for r in _read_rows(MATURITY_CSV)}


def build(output_csv=OUTPUT_CSV):
    csv.field_size_limit(10**9)
    bdc_type = _bdc_types()
    non_accrual = _non_accrual()
    html_maturity = _html_maturities()
    from report_sector import report_sectors
    sector_report = report_sectors(LABELS_CSV)
    labels = {}
    for row in _read_rows(LABELS_CSV):
        assert row["investment_identifier"] not in labels, f"duplicate identifier in labels: {row['investment_identifier']!r}"
        labels[row["investment_identifier"]] = row
    fixes = {(row["accession"], row["context_id"]): row for row in _read_rows(RATE_FIXES_CSV)}

    tmp = Path(str(output_csv) + ".tmp")
    n = fixed = filled = 0
    with open(ENRICHED_CSV, encoding="utf-8", newline="") as fi, open(tmp, "w", encoding="utf-8", newline="") as fo:
        reader, writer = csv.reader(fi), csv.writer(fo, lineterminator="\n")
        header = next(reader)
        ix = {c: i for i, c in enumerate(header)}
        writer.writerow(header + [c + "_prev" for c in LABEL_COLUMNS] + ["validation_id"] + DECISION_COLUMNS
                        + ["bdc_type", "non_accrual", "maturity_source", "sector_report"])
        for row in reader:
            n += 1
            label = labels[row[ix["investment_identifier"]]]  # KeyError: an identifier the validation did not cover
            prev = [row[ix[c]] for c in LABEL_COLUMNS]
            # The labels were validated against these exact old values; stop if the enriched panel has changed since.
            assert prev == [label[c] for c in PREV_IN_LABELS], f"row {n}: old labels differ from the validated *_prev values"
            for c in LABEL_COLUMNS:
                row[ix[c]] = label[c]

            fix = fixes.get((row[ix["accession"]], row[ix["context_id"]]))
            if fix is not None:
                assert (row[ix[RATE]], row[ix[CASH]], row[ix[SPREAD]], row[ix["rate_source"]]) == (
                    fix["rate_old"], fix["cash_old"], fix["spread_old"], fix["rate_source_old"]), f"row {n}: rate fix does not match the panel"
                row[ix[RATE]], row[ix[CASH]], row[ix[SPREAD]], row[ix["rate_source"]] = (
                    fix["rate_new"], fix["cash_new"], fix["spread_new"], fix["rate_source_new"])
                fixed += 1

            source = "xbrl" if row[ix[MATURITY]] else ""
            if not source:
                mat = html_maturity.get((row[ix["accession"]], row[ix["context_id"]]))
                if mat:
                    row[ix[MATURITY]], source = mat, "html"
                    filled += 1

            writer.writerow(row + prev + [label["id"]] + [label[c] for c in DECISION_COLUMNS]
                            + [bdc_type(row[ix["cik"]], row[ix["cal_q"]]),
                               non_accrual(row[ix["accession"]], row[ix["context_id"]]), source,
                               sector_report[row[ix["investment_identifier"]]]])
    assert fixed == len(fixes), f"only {fixed} of {len(fixes)} logged rate fixes matched a row"
    tmp.replace(output_csv)
    print(f"Wrote {output_csv}: {n:,} rows, {len(labels):,} validated identifiers, {fixed} rate fixes, "
          f"{filled:,} maturities filled from HTML")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else OUTPUT_CSV)
