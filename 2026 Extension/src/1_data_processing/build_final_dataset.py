"""Build data/final_dataset/FINAL_DATASET.csv from the enriched panel and the validated labels.

Usage:
    python src/1_data_processing/build_final_dataset.py [output.csv]

Inputs:
- data/enriched/data_private_credit_FINAL_enriched_final.csv, written by build_final_enriched.py.
- data/final_dataset/labels_by_identifier.csv: one validated decision per distinct
  investment_identifier, made with final_validation_prompt.md (see AUDIT_SUMMARY.md).
- data/cleaned/rate_swap_fixes_2026-10-19.csv: log of the 2026-10-19 rate fix. Reported rates
  that were only the cash part of a cash + PIK coupon get the full rate.

For every row this script:
1. Replaces borrower_canonical, instrument_seniority and sector with the validated values. The
   old values go to *_prev columns.
2. Appends validation_id and the confidence, evidence, action and reason columns.
3. Applies the logged rate fix to the rows it lists.

Values are copied as text, so every untouched field is exactly what the inputs hold.
"""

import csv
import sys
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py

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


def _read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def build(output_csv=OUTPUT_CSV):
    csv.field_size_limit(10**9)
    labels = {}
    for row in _read_rows(LABELS_CSV):
        assert row["investment_identifier"] not in labels, f"duplicate identifier in labels: {row['investment_identifier']!r}"
        labels[row["investment_identifier"]] = row
    fixes = {(row["accession"], row["context_id"]): row for row in _read_rows(RATE_FIXES_CSV)}

    tmp = _Path(str(output_csv) + ".tmp")
    n = fixed = 0
    with open(ENRICHED_CSV, encoding="utf-8", newline="") as fi, open(tmp, "w", encoding="utf-8", newline="") as fo:
        reader, writer = csv.reader(fi), csv.writer(fo, lineterminator="\n")
        header = next(reader)
        ix = {c: i for i, c in enumerate(header)}
        writer.writerow(header + [c + "_prev" for c in LABEL_COLUMNS] + ["validation_id"] + DECISION_COLUMNS)
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

            writer.writerow(row + prev + [label["id"]] + [label[c] for c in DECISION_COLUMNS])
    assert fixed == len(fixes), f"only {fixed} of {len(fixes)} logged rate fixes matched a row"
    tmp.replace(output_csv)
    print(f"Wrote {output_csv}: {n:,} rows, {len(labels):,} validated identifiers, {fixed} rate fixes")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else OUTPUT_CSV)
