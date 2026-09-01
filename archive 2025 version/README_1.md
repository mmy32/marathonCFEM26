# 1. Pipeline Overview

SEC EDGAR (master.idx + submissions JSON)
   ↓
BDC universe construction (N-54A / N-54C)
   ↓
Filing download (10-K / 10-Q primary documents)
   ↓
iXBRL parsing (typed contexts + facts → wide table)
   ↓
Combine wide outputs (append + dedup)
   ↓
Preprocess & normalize (prefix merge, quarter labels, filtering, scaling fixes)
   ↓
Analysis-ready dataset (`ixbrl_clean.csv`)

---

# 2. Directory Structure

> Notes:
> - `cache/` stores raw SEC downloads + download metadata/failures.
> - `data/ixbrl/` stores parsed iXBRL artifacts.
> - `data/processed/` stores BDC universe outputs + final cleaned dataset.

## Core reusable logic (no execution)
```

├── bdc.py               # Build BDC universe from master.idx
├── filings.py           # SEC submissions -> metadata + download primary docs
├── ixbrl_parser.py      # Parse iXBRL instance XML + combine wide outputs
├── preprocessor.py      # Cleaning & normalization (final dataset)
├── paths.py             # Centralized path definitions
└── helpers.py           # Shared helper utilities (e.g., quarter labels)

```

## Entry points (execution scripts)
```

├── run_bdc_universe.py
├── run_download.py
├── run_ixbrl_parser.py
└── run_preprocessor.py

```

## Output storage
```

cache/
├── {CIK}/
│   └── {accession}.{ext}          # primary filing document (HTML/TXT)
├── all_metadata.csv               # dedup key: (cik, accessionNumber)
└── download_failures.csv          # failed downloads for retry

data/
├── processed/
│   ├── BDC_filings_2001_present.csv
│   ├── BDC_intervals.csv
│   └── ixbrl_clean.csv
└── ixbrl/
    ├── raw_xml/
    │   └── {cik}_{accession}.xml  # extracted XBRL instance document
    ├── per_file/                  # optional per-filing outputs (off by default)
    └── combined/
        ├── wide_all.csv
        └── ixbrl_parse_failures.csv

```

---

# 3. Core Modules

## 3.1 bdc.py

Construct the BDC universe from SEC EDGAR master index files.

### Outputs
```

<out_dir>/
├── BDC_filings_2001_present.csv   # raw N-54A / N-54C filings
└── BDC_intervals.csv              # derived BDC active intervals

```

- Output paths are controlled via `out_dir` or `out_csv`.
- If both are provided, `out_csv` takes precedence.

### Main APIs

#### `build_bdc_filings_from_masteridx(...)`
- Scans all quarters from `start_year` to `end_year`
- Filters forms via regex (default: `N-54A` / `N-54C`)
- Optional amendments (`/A`) via `include_amendments=True`
- Normalizes CIK to 10-digit zero-padded strings
- Deduplicates on `(CIK, Form, Date, Link)`
- Returns columns: `Company`, `CIK`, `Form`, `Date`, `Link`

#### `build_bdc_intervals_from_filings(...)`
- Sorts by `(CIK, Date)`
- Treats `N-54A` as start and `N-54C` as termination
- Handles missing start and open-ended “still active” intervals
- Returns columns: `CIK`, `Company`, `start_date`, `end_date`, `Link_A`, `Link_C`

---

## 3.2 filings.py

Download SEC 10-K / 10-Q filings by CIK with metadata-based resume and failure retry.

### Outputs
```

cache/
├── {CIK}/{accession}.{ext}   # primary filing document
├── all_metadata.csv          # dedup key: (cik, accessionNumber)
└── download_failures.csv     # failed downloads

```

- Files are named by accession (original primaryDocument filename not preserved).
- `all_metadata.csv` is used for incremental downloads when `only_new=True` and `update_metadata=True`.

### Main APIs

#### `download_for_cik(...)`
- Fetch submissions JSON → build metadata → download primary docs
- Date filtering is on **filingDate** (not reportDate)
- `only_new=True` uses `all_metadata.csv` to skip existing `(cik, accessionNumber)`
- `overwrite=False` skips existing local files
- `test_mode=True` disables metadata writes but **still downloads files**
- Returns `(metadata_df, DownloadResult)`

#### `download_failures_only(...)`
- Retries rows recorded in `download_failures.csv`
- Overwrites local files by default
- Deletes `download_failures.csv` if all retries succeed

---

## 3.3 ixbrl_parser.py

Parse iXBRL filings into structured, tabular data.

### Outputs
```

data/ixbrl/
├── raw_xml/
│   └── {cik}*{accession}.xml      # extracted XBRL instance (downloaded)
├── per_file/                      # optional, off by default
│   └── {CIK}/{accession}**.csv
└── combined/
└── wide_all.csv               # combined wide table

```

- Raw XML is downloaded from the filing index page link labeled **“Extracted XBRL Instance Document”**.
- `wide_all.csv` is deduplicated on `(cik, accession, context_id)`.

### Main APIs

#### `process_one_filing(...)`
- Input: one metadata row containing `cik`, `index_url`, `reportDate`
- Locates extracted instance XML from `index_url`
- Extracts **typed-member contexts only**
- Filters contexts to match `reportDate`
- Builds a wide table keyed by `context_id`
- Adds `investment_identifier` from typed members
- Returns `(df_wide, ok, status)`

#### `combine_wide_frames(...)`
- Concatenates successful frames
- Optional `append_if_exists=True` appends to existing `wide_all.csv`
- Deduplicates on `(cik, accession, context_id)`
- Writes `data/ixbrl/combined/wide_all.csv`

### Configuration
`IXBRLConfig` controls throttling and optional per-file outputs:
- `user_agent`, `request_timeout`, `sleep_between`, `retry`
- `write_raw_xml`, `write_per_file_wide`, `write_per_file_contexts`

---

## 3.4 preprocessor.py

Transform combined parsed iXBRL wide data into an analysis-ready dataset.

### Inputs / Outputs
- Input: `IXBRL_WIDE_ALL_CSV` (typically `data/ixbrl/combined/wide_all.csv`)
- Output: `PROCESSED_DIR/ixbrl_clean.csv`

Optional diagnostics:
- `PROCESSED_DIR/preprocess_nonnull_stats.csv` (only when enabled)

### Main Transformations
- Prefix-merge namespace variants (`us-gaap:Foo`, `dei:Foo`, `Foo` → `Foo`)
- Add calendar-quarter labels (`cal_qe`, `cal_q`) from `period`
- Filter to quarter range `[quarter_from, quarter_to]`
- Classify rows as `terms_only / amounts_only / mixed / empty`
- Default filters:
  - Keep only rows with missing `InvestmentOwnedBalanceShares` (`only_no_shares=True`)
  - Drop `amounts_only` rows (`drop_amounts_only=True`)
- Normalize spread/rate scales:
  - Adds `*_normalized` and `*_scale_flag`
  - Attempts divide-by-100 or divide-by-10000 to fit valid ranges
  - Forces PIC/PIK/Floor normalized values non-negative (flags `|sign_error`)

### Main APIs
- `run_preprocess(...)`: reads input (pyarrow), runs transforms, writes output CSV
- `preprocess_combined_wide(...)`: wrapper for explicit combined-wide input

---

## 3.5 paths.py

Single source of truth for all file paths (e.g., `METADATA_CSV`, `IXBRL_WIDE_ALL_CSV`, `PROCESSED_DIR`).

---

# 4. Execution Scripts

Scripts under `scripts/` add `final/` to `sys.path` so `import src.*` works when executed directly.

## `run_bdc_universe.py`
Outputs to `PROCESSED_DIR`:
- `BDC_filings_2001_present.csv`
- `BDC_intervals.csv`

## `run_download.py`
Modes:
- `test`: single CIK + date window; does **not** update `all_metadata.csv`
- `full`: iterates all CIKs in `BDC_intervals.csv`; uses metadata-based resume (`only_new=True`)
- `retry_failed`: retries `download_failures.csv` only

Outputs:
- `cache/{CIK}/{accession}.{ext}`
- `cache/all_metadata.csv`
- `cache/download_failures.csv` (if any failures)

## `run_ixbrl_parser.py`
Modes:
- `test`: parses a small subset (CIK/date/limit); resume disabled
- `full`: `append_if_exists=True`, `resume_parsed_only=True` (skip already-parsed `(cik, accession)`)
- `retry_failed`: parses only rows listed in `data/ixbrl/combined/ixbrl_parse_failures.csv`

Outputs:
- `data/ixbrl/combined/wide_all.csv`
- `data/ixbrl/combined/ixbrl_parse_failures.csv`

## `run_preprocessor.py`
Modes:
- `test`: smaller quarter range + diagnostics enabled
- `full`: full-quarter preprocessing for analysis

Outputs:
- `data/processed/ixbrl_clean.csv`
- (optional) `data/processed/preprocess_nonnull_stats.csv`
