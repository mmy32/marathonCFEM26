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
> - All modules and entry-point scripts live flat in this one directory (no `src/`/`scripts/` split). `paths.py` sets `ROOT = Path(__file__).resolve().parents[0]`, i.e. this directory — keep it that way if the project is ever moved or re-flattened, since a mismatched `ROOT` silently writes `data/`/`cache/` to the wrong location instead of erroring.

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
- Throttles each `master.idx` request (`sleep_between=0.25s` default) with `retry=2` retries. SEC intermittently drops/resets requests fired back-to-back with no delay — an unthrottled ~100-request burst (one per year/quarter) silently lost roughly a third of quarters with no error surfaced (failed requests are swallowed and just skipped). Do not remove the throttling to "speed this up."

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
- `fetch_all_submissions(...)` throttles each `data.sec.gov` request (`sleep_between=0.25s`, `retry=2`) for the same reason as the `master.idx` scan above — same failure mode, same fix.
- `load_existing_metadata(...)` and `save_failures(...)` tolerate an existing-but-empty `all_metadata.csv` / `download_failures.csv` (e.g. left behind by a crashed run) instead of raising `pandas.errors.EmptyDataError`. A 0-byte metadata file previously made every CIK in a `full` run fail immediately on the `only_new` lookup, before any download was attempted.
- A full download run of ~425 CIKs / ~8,600 filings uses tens of GB locally — check free disk space first; a disk-full mid-write can silently truncate `all_metadata.csv`, which is exactly the empty-file case above.

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
- Forces `cik`/`accession`/`context_id` to `str` on every reload and before dedup. `cik` and `accession` are digit-only strings with no leading zeros or dashes, so a plain `pd.read_csv` on the existing `wide_all.csv` silently infers them as `int64`, while freshly parsed frames keep them as `str` — the mismatch makes `drop_duplicates` treat every previously-parsed filing as "new" again. Confirmed by reproduction: appending an already-parsed filing doubled its row count (35 → 70) before this fix. This matters most for the intended `full`-mode workflow (`append_if_exists=True` + `resume_parsed_only=True`), which appends on every incremental run.
- `process_one_filing(...)` now re-downloads once and retries if the cached raw XML fails to parse (`etree.XMLSyntaxError`), instead of treating a truncated/corrupt cached file (e.g. left over from a disk-full crash) as permanently done. It also parses the XML tree once and reuses it for context extraction and fact extraction (previously 4 full `etree.parse` calls per filing), and adds a short delay between the index-page fetch and the XML download inside a single filing — the same request-burst issue seen in `bdc.py`/`filings.py` applies here too, just at a smaller scale (2 requests per filing instead of ~100 in one loop).

#### `append_wide_frames_bounded(...)`
- Memory-bounded replacement for `combine_wide_frames(...)` used by `run_ixbrl_parser.py`'s periodic flush loop (`combine_wide_frames` itself is unchanged, for smaller/ad hoc use).
- The full combined table is wide *and* long: hundreds of BDC filers each contribute their own XBRL extension-taxonomy columns (no common schema until `preprocessor.py`'s prefix-merge step), observed at ~745k rows x 1,746 cols, ~1.5GB on disk. `combine_wide_frames` reloads the *entire* existing file on every call — on a machine with only a few GB of RAM, that reliably OOM'd, first on the final combine of a ~2,300-filing run, then again on a periodic flush after the accumulated table reached ~340k rows.
- Never loads the full accumulated file: if a new batch's columns are already a subset of the on-disk header, rows are appended directly (only the small new batch touches memory); if the batch introduces new columns, the existing file is rewritten in bounded-size chunks (`chunksize`, read back as `str`) to add them, then the batch is appended.
- Does **not** re-check duplicates against the full existing file (that would require the same full-file load this function exists to avoid) — only within-batch duplicates are removed. Correctness relies on the caller's `resume_parsed_only` guaranteeing the filings behind a batch aren't already in `out_csv`. See `load_parsed_keys_from_wide` below — this guarantee silently broke once before and caused ~53k duplicate rows, which had to be cleaned up with a one-off chunked dedup pass.

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
- `TERM_COLS` includes `InvestmentInterestRateFloor` — it was missing before, which meant `select_analysis_columns` dropped the raw column before the normalization logic above (which already handled it) ever got a chance to run, silently discarding all loan floor-rate data and misclassifying Floor-only contexts as `empty` instead of `terms_only`. The downstream cleaning stage (`ixbrl_utils.py`, `run_ixbrl_pipeline.py`) already expected this column to be present.

### Main APIs
- `run_preprocess(...)`: reads input in row chunks (default C engine — `engine="pyarrow"` doesn't support `chunksize`), runs transforms per chunk, appends each cleaned chunk directly to the output CSV. Previously read the whole input via `pd.read_csv(..., engine="pyarrow")` in one shot, which OOM'd trying to load the full ~1.5GB/1,746-column combined table. Every transform in the pipeline (calendar-quarter tagging, prefix-merge, column selection, the quarter/share/context-type filters, rate normalization) is row-local, so chunking is exact, not an approximation — verified by comparing chunked vs. single-chunk output on a sample (identical). The one aggregate step (non-null diagnostics) is summed incrementally across chunks instead of computed on the whole frame. The returned DataFrame is read back from the written output only if it's under `reload_max_bytes` (default 500MB) — for a large output, reloading it would recreate the same OOM risk, so an empty DataFrame is returned instead with a logged warning; read the CSV directly (in chunks, if needed) in that case.
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

`full` mode wraps each CIK's `download_for_cik(...)` call in its own try/except — one CIK raising (network error, malformed submissions payload, etc.) is logged and skipped rather than aborting the remaining ~400+ CIKs in what is typically a multi-hour run.

Outputs:
- `cache/{CIK}/{accession}.{ext}`
- `cache/all_metadata.csv`
- `cache/download_failures.csv` (per-filing failures, if any)
- `data/processed/cik_level_download_failures.csv` (CIKs that failed entirely, if any)

## `run_ixbrl_parser.py`
Modes:
- `test`: parses a small subset (CIK/date/limit); resume disabled
- `full`: `append_if_exists=True`, `resume_parsed_only=True` (skip already-parsed `(cik, accession)`)
- `retry_failed`: parses only rows listed in `data/ixbrl/combined/ixbrl_parse_failures.csv`

`full` mode flushes parsed filings to disk every `flush_every` (default 100) instead of holding everything in memory for the whole run, via `append_wide_frames_bounded` (see `ixbrl_parser.py` above) — necessary at this data's actual scale, not just an optimization.

`load_parsed_keys_from_wide(...)` (drives `resume_parsed_only`) reads `wide_all.csv` in chunks rather than one `pd.read_csv(usecols=[...])` call. `usecols` does not avoid the cost of tokenizing every column on every row — the C engine parses full rows before dropping unwanted columns — so on the wide combined table this OOM'd even though only 2 columns were being kept. Because the failure was originally caught and silently treated as "nothing parsed yet," resume_parsed_only skipped nothing and a run re-parsed and duplicated ~800 already-parsed filings (~53k duplicate rows, cleaned up with a one-off chunked dedup pass) before this was caught. The function now re-raises instead of silently falling back to an empty set, since a silent fallback here directly causes duplicate work/data rather than being a safe default.

The `Progress N/M` log line now uses `stats.processed` instead of the `iterrows()` loop variable `i`. After `resume_parsed_only` filters `df_meta` via a merge, the DataFrame's index keeps its original (pre-filter) labels rather than a clean `0..N-1` range, so `i` was a stale index label, not a position counter — visible as nonsensical output like `Progress 2300/1739` (numerator exceeding the stated total) in an otherwise-correct run.

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
