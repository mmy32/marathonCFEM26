# Building FINAL_DATASET.csv

## Overview

`data/final_dataset/FINAL_DATASET.csv` is the only dataset the analyses read. It holds one row
per position per quarter (502,299 rows, 2023Q1 to 2026Q2), with cleaned values and rates and
validated borrower, seniority and sector labels.

Two steps build it from the cleaned panel (`data/cleaned/ixbrl_cleaned_out.csv`, see
`2_ixbrl_cleaning.md`):

| Step | Script | Output |
|---|---|---|
| 7. Enrich | `src/1_data_processing/build_final_enriched.py` | `data/enriched/data_private_credit_FINAL_enriched_final.csv` |
| 8. Validate | `src/1_data_processing/build_final_dataset.py` | `data/final_dataset/FINAL_DATASET.csv` |

```
python src/1_data_processing/build_final_enriched.py
python src/1_data_processing/build_final_dataset.py
```

Neither step calls an LLM: both read cached results. Both copy panel values as text, so a
kept column is exactly what the cleaning step wrote. Both outputs were rebuilt on 2026-10-22
and are byte-identical to the saved files.

---

## Step 7: `build_final_enriched.py`

### Inputs

| File | What it is |
|---|---|
| `data/cleaned/ixbrl_cleaned_out.csv` | The cleaned panel |
| `data/enriched/borrower_name_cleaned.csv` | Teammates' borrower lookup: `investment_identifier` -> `borrower_canonical`. Despite its name it is an .xlsx workbook. |
| `data/enriched/_seniority_sector_llm_checkpoint.csv` | Cached LLM sector answers from `get_seniority_debttype.py` (see `3_seniority_sector.md`) |

### What it does

- **Borrower:** an exact join of `investment_identifier` to the lookup. Names go through
  `normalize_borrower_name()`: upper case, no punctuation, and no legal-form words such as LLC,
  Inc or GmbH. That way spelling variants of one borrower share one name. Identifiers the
  lookup does not cover stay blank.
- **Seniority:** `instrument_seniority` comes from keyword rules on `investment_identifier`.
  The rules are copied verbatim from `get_seniority_debttype.py`, because there they are nested
  in one function and cannot be imported.
- **Sector:** keyword rules first. If they give "Other / Unknown", the script uses the cached LLM
  answer when its confidence is at least 0.70 (`CONF_THRESH`). Otherwise the sector is
  "Other / Unknown / Unresolved". The LLM is never called.
- **Dropped columns:** as-reported values superseded by their `*_normalized` columns, and
  cleaning bookkeeping (`DROP_COLUMNS`). `rate_source`, `check_1`, `principal_cost_check` and
  `check_2` are kept. `5_pik/` needs `check_2` to tell tagged PIK from inferred PIK.

The output is written to a `.partial` file and then re-read next to the panel. The script stops
unless every kept column and the row order are unchanged; only then is the file renamed.

### Coverage on the current data

| | Rows | Share |
|---|---|---|
| Borrower matched | 483,804 | 96.3% |
| Seniority identified (not `OTHER_UNKNOWN`) | 327,038 | 65.1% |
| Sector identified | 397,763 | 79.2% |

Sector source by row: rules 187,733; LLM 210,185; LLM below 0.70 confidence 104,381.

---

## Step 8: `build_final_dataset.py`

### Inputs

| File | What it is |
|---|---|
| `data/enriched/data_private_credit_FINAL_enriched_final.csv` | Step 7 output |
| `data/final_dataset/labels_by_identifier.csv` | One validated decision per distinct `investment_identifier` (221,089) |
| `data/cleaned/rate_swap_fixes_2026-10-19.csv` | Log of the 2026-10-19 rate fix (92 rows) |

The labels come from a one-off LLM validation that ran outside this code. It followed
`src/1_data_processing/final_validation_prompt.md`, and `data/final_dataset/AUDIT_SUMMARY.md`
describes the rules, the judgment calls and the checks. `labels_validated.json` holds the same
decisions in the format that prompt specifies.

### What it does

For every row:

1. Replaces `borrower_canonical`, `instrument_seniority` and `sector` with the validated values.
   The old values go to `borrower_canonical_prev`, `instrument_seniority_prev` and `sector_prev`.
2. Appends `validation_id` (the labels file's `id`) and the decision columns: confidence,
   evidence, action (`keep` / `fill` / `replace` / `blank`) and reason for each label.
3. Applies the logged rate fix to the 92 rows it lists (11 BDCs, 2023Q1 to 2026Q2). In each, the
   reported rate was only the cash part of a cash + PIK coupon:
   - **86 rows:** total = cash + PIK matches base + spread, so the total rate becomes cash + PIK.
   - **6 rows:** one fixed-rate loan whose identifier states "11.00% (7.00% cash + 4.00% PIK)" and
     whose total sat in the spread column; the spread is blanked.

   In both cases the cash rate becomes the old reported rate and `rate_source` becomes `derived`.
4. Appends `bdc_type`: `listed`, `non-traded` or `private` for the row's BDC in that quarter,
   from `data/reference/bdc_type.csv`. That table is built by `src/1_data_processing/bdc_type.py`
   from each BDC's EDGAR filing history (cached in `cache/sec_submissions/`):
   - **listed** from the first Form 8-A12B, for BDCs with an exchange ticker today or a later
     Form 25 (merged away), until the delisting;
   - **non-traded** from the first prospectus (424B, or 497) after an N-2, when there are at
     least three (a public continuous offering);
   - **private** otherwise (Form 10 private placements, or an N-2 that never led to an offering).

   Two OTC-traded BDCs are set by hand (Princeton Capital, Firsthand Technology Value Fund).
   Types change over time: Morgan Stanley Direct Lending, Nuveen Churchill Direct Lending, Palmer
   Square and Kayne Anderson listed in 2024, MSC Income Fund (non-traded since 2012) and Blue Owl
   Technology Finance in 2025; Ares Strategic Income and Golub Capital Private Credit turned
   non-traded in 2023. At 2026Q2: 48 listed BDCs ($153bn FV), 24 non-traded ($233bn), 99
   private ($142bn).
5. Appends `non_accrual`: `1` when the row's XBRL facts carry a non-accrual footnote, `0` when
   not and the filing's non-accrual status is known, blank when unknown.
   - `src/1_data_processing/nonaccrual_scan.py` reads every XBRL instance in
     `data/ixbrl/raw_xml/` and writes the footnotes that mention non-accrual, the contexts they
     are linked to, and any non-accrual totals and net assets the BDC tags
     (`data/parsed/nonaccrual/`). About 20 minutes.
   - `nonaccrual_flags.py` keeps footnotes whose wording says the marked investment is on
     non-accrual ("loan was on non-accrual status as of ...", "non-accrual investment") and
     drops general notes ("interest rate ... excluding those on non-accrual", "there were no
     investments on non-accrual", notes linked to most of a filing's lines).
   - A filing's status is known when it has a flag-type footnote or says no investment is on
     non-accrual: 1,109 of 1,986 filings, 86% of fair value (listed 98%, non-traded 99%,
     private 70% at 2026Q2). Other BDCs mark non-accrual only in the HTML table.
   - Checks: flagged fair value equals the BDC's own tagged non-accrual total in most filings
     that tag one; mean non-accrual at cost across BDCs at 2026Q2 is 4.2% for listed and 1.0%
     for non-traded BDCs, against 4.2% and 1.1% in Financing the Economy 2026 (Fig 39).
   - Some BDCs also footnote preferred equity on non-accrual (non-cash dividends); use debt
     rows (par > 0) for loan outcomes.

### Safety checks

The script stops with an error if:

- an identifier has no validated label;
- a row's old labels differ from the `*_prev` values the labels were validated against, i.e.
  step 7 has changed since the validation;
- a logged rate fix does not match the row's current rate, cash rate, spread and `rate_source`;
- any logged fix is left unapplied.

If step 7 changes, for example a new borrower lookup or new cleaning rules, the validation has
to be redone for the identifiers that changed. Rerunning this script alone is not enough.

### What the validation changed (rows)

From `AUDIT_SUMMARY.md`:

| Field | Kept | Filled | Corrected | Left blank / unresolved |
|---|---|---|---|---|
| Borrower | 473,823 | 18,252 | 10,070 | 397 |
| Seniority | 444,436 | 54,431 | 3,432 | 120,880 `OTHER_UNKNOWN` |
| Sector | 322,241 | 180,058 replaced in total | — | 42,256 unresolved |

---

## Columns

| Group | Columns |
|---|---|
| Keys | `cik`, `accession` (integer, no leading zeros), `investment_identifier`, `context_id`, `period`, `cal_qe`, `cal_q` |
| Labels | `borrower_canonical`, `instrument_seniority`, `sector`, and their `*_prev` values |
| Amounts (USD) | `InvestmentOwnedAtFairValue_normalized`, `InvestmentOwnedAtCost_normalized`, `InvestmentOwnedBalancePrincipalAmount_normalized`, plus `-unitRef_normalized` |
| Rates (decimals) | `InvestmentInterestRate_normalized`, `..._PaidInCash_normalized`, `..._PaidInKind_normalized`, `InvestmentBasisSpreadVariableRate_normalized`, `InvestmentInterestRateFloor_normalized` |
| Rate flags | `currency`, `RateType`, `is_fixed`, `is_prime`, `rate_source`, `check_2`, `check_1`, `principal_cost_check` (see `2_ixbrl_cleaning.md`) |
| Validation | `validation_id` and the confidence, evidence, action and reason columns |
| BDC | `bdc_type` (`listed` / `non-traded` / `private`, by quarter; added 2026-10-23) |
| Status | `non_accrual` (`1` / `0` / blank = unknown; added 2026-10-23) |

`index_construction.load_and_prepare_investment_data()` reads this file into the short names
the analyses use: `FV`, `COST`, `PAR`, `rate_cash`, `interest_rate` and `rate_pik`.
