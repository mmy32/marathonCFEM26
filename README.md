# Cornell x Marathon Capstone: Early Warning Signals in Private Credit

This repo builds a loan-level dataset of what BDCs (business development companies) hold, read
from the schedules of investments in their SEC 10-K and 10-Q filings, 2023Q1 to 2026Q2. It uses
the dataset for two things:

- **The midterm deck** (*Examining Early Warning Signals in Private Credit*, 5 Nov 2026): BDC
  growth, a BDC loan index against the CDLI, borrower warning signs (crowding, disagreement on
  value, PIK), each BDC's exposure to flagged borrowers, case studies and data-center lending.
- **The early-warning scorecard** (after the midterm): ranks borrowers by the chance of a credit
  event next quarter (non-accrual, or leaving the books marked below 80), from three signs (a low
  mark, a falling mark, a switch to paying interest in kind).
  Start with [docs/14](docs/14_early_warning_model.md).

All analysis reads one file: `data/final_dataset/FINAL_DATASET.csv`.

## Quick start

Use Python 3.11 (the pinned pandas 2.0.2 does not install on 3.12 or later):

```
pip install -r requirements.txt
```

The data is not in the repo; it is shared separately. Put `FINAL_DATASET.csv` (259 MB) in
`data/final_dataset/`, then run from the project root:

```
python scripts/run_model.py                                 # early-warning scorecard (docs/14), about 6 minutes
python scripts/run_risk_monitor.py                          # warning signs and BDC exposure, about 3 minutes
python scripts/run_distress_event_study.py                  # case studies, about 3 minutes
python src/2_index_construction/index_construction.py       # loan index vs CDLI
python src/1_data_processing/bdc_growth.py                  # BDC growth
python src/1_data_processing/dataset_overview.py            # dataset summary
python src/6_data_center/classify_data_center_borrowers.py  # data-center tags, then the two plot_data_center_*.py
```

`plot_data_coverage.py` (filing coverage) also needs `cache/all_metadata.csv` and
`data/ixbrl/combined/wide_all.csv`. No step needs the network or an API key.

## Layout

```
├── src/                           one folder per stage
│   ├── 1_data_processing/         SEC download, iXBRL parsing, cleaning, labels; paths.py
│   ├── 2_index_construction/      flow-based BDC loan index
│   ├── 3_concentration_crowding/  lender families per borrower (used by 8 and 9)
│   ├── 4_val_dispersion/          disagreement between lenders' marks (used by 8 and 9)
│   ├── 5_pik/                     PIK switches (used by 8 and 9)
│   ├── 6_data_center/             data-center borrowers
│   ├── 8_risk_monitor/            warning signs, BDC exposure, borrower panel, case studies
│   └── 9_model/                   early-warning model
├── scripts/                       run_*.py entry points
├── tests/                         unit tests
├── docs/                          one write-up per stage
├── outputs/                       the tables and charts behind the deck and the model
├── data/, cache/                  datasets and SEC downloads (not committed)
└── requirements.txt               exact pins: outputs reproduce byte for byte
```

`src/1_data_processing/paths.py` holds every path. Scripts add that folder to `sys.path` and call
`paths.add_code_dirs_to_path()` to import from the other `src/` folders.

## Where each result comes from

| Midterm slides | Script | Output in `outputs/` |
|---|---|---|
| 3-4 BDC growth | `src/1_data_processing/bdc_growth.py` | `eda_plots/private_credit_growth.png`, `bdc_growth.png` |
| 6 Filings collected and kept | `src/1_data_processing/plot_data_coverage.py` | `index_construction_plots/filings_by_type.png`, `parsing_coverage.png`, `par_retention.csv` |
| 7 Dataset summary | `src/1_data_processing/dataset_overview.py` | `eda_plots/dataset_overview.png` |
| 8 Index vs CDLI | `src/2_index_construction/index_construction.py` | `index_construction_plots/index_vs_cdli_panels.png`, `index_vs_cdli_stats.csv` |
| 9-10, 13, 24 Case studies | `scripts/run_distress_event_study.py` | `risk_monitor/distress/case_studies/` |
| 11-15 Warning signs, BDC exposure | `scripts/run_risk_monitor.py` | `risk_monitor/signals/` |
| 17-18 Data-center lending | `src/6_data_center/plot_data_center_timeseries.py`, `plot_data_center_sectors.py` | `data_center_plots/` |
| Early-warning model | `scripts/run_model.py` | `model/` |

Not produced by code: slide 2 (the Federal Reserve's chart), slide 7's "~96% private" (a separate
LLM check) and slide 24's news sources. On slide 6, "2,021 read" is the parsed-filings total
that `plot_data_coverage.py` prints.

## Rebuilding FINAL_DATASET.csv from the filings

| Step | Command | Writes |
|---|---|---|
| 1. BDC universe | `python scripts/run_bdc_universe.py` | `data/universe/` |
| 2. Download 10-K / 10-Q filings | `python scripts/run_downloader.py` | `cache/` |
| 3. Parse iXBRL | `python scripts/run_ixbrl_parser.py` | `data/ixbrl/combined/wide_all.csv` |
| 4. One row per position | `python scripts/run_preprocessor.py` | `data/parsed/ixbrl_clean.csv` |
| 5. Base rates | `python src/1_data_processing/augment_rates.py` | `data/reference/SOFR_augmented.csv` |
| 6. Clean rates, currencies, values | `python scripts/run_ixbrl_pipeline.py` | `data/cleaned/ixbrl_cleaned_out.csv` |
| 7. Borrower, seniority, sector labels | `python src/1_data_processing/build_final_enriched.py` | `data/enriched/` |
| 8. Validated labels and rate fixes | `python src/1_data_processing/build_final_dataset.py` | `data/final_dataset/FINAL_DATASET.csv` |

Steps 4 and 6-8 take about 11 minutes from the shared files below and reproduce
`FINAL_DATASET.csv` byte for byte. Steps 1-3 and 5 call the SEC and the central banks, whose data
keeps changing. Steps 2-3 take hours, and a rebuild would pick up filings after 2026Q2 that have no
validated labels, so step 8 would stop. `run_downloader.py`, `run_ixbrl_parser.py` and
`run_preprocessor.py` have a small `test` mode (set `mode = "test"` in `main()`). [docs/6](docs/6_final_dataset.md) covers steps 7-8, including how
`bdc_type`, `non_accrual` and HTML maturities are rebuilt.

Shared input files, each at the path shown:

| File | What it is |
|---|---|
| `data/reference/` (10 files) | CDLI / CDLI-S, FX, base rates, `bdc_type.csv`, hand-made `case_studies.csv` |
| `data/universe/`, `cache/all_metadata.csv` | BDC universe and filing metadata |
| `data/ixbrl/combined/wide_all.csv` (1.5 GB) | Parsed filings |
| `data/parsed/nonaccrual/`, `data/parsed/maturity_html/` | Non-accrual footnotes and HTML maturities |
| `data/cleaned/rate_swap_fixes_2026-10-19.csv` | Rate fixes replayed in step 8 |
| `data/enriched/borrower_name_cleaned.csv`, `_seniority_sector_llm_checkpoint.csv` | Borrower lookup and cached LLM sector answers |
| `data/final_dataset/labels_by_identifier.csv` (+ `labels_validated.json`, `AUDIT_SUMMARY.md`) | Validated borrower, seniority and sector labels |

The LLM steps (`get_seniority_debttype.py`, `screen_data_center_borrowers.py`) need
`ANTHROPIC_API_KEY`, but their answers are cached, so nothing above needs a key. Never commit a
key; `claude_api_key.txt` is git-ignored.

## Tests

```
python tests/test_model.py
python tests/test_tranches.py
python tests/test_pik_metrics.py
python tests/test_position_matching.py
python tests/test_nonaccrual_bdc_type.py
python tests/test_radar.py      # about 2 minutes: our data against Private Credit Radar 1.0, within a factor of two
```

## Docs

1. [Pipeline overview](docs/1_pipeline_overview.md): download, parse, preprocess
2. [iXBRL cleaning](docs/2_ixbrl_cleaning.md): rate, currency and value rules
3. [Seniority and sector](docs/3_seniority_sector.md)
4. [Index construction](docs/4_index_construction.md)
6. [Building FINAL_DATASET.csv](docs/6_final_dataset.md)
7. [Crowding](docs/7_concentration_crowding.md)
10. [Data-center lending](docs/10_data_center.md)
11. [Tranche matching and manager families](docs/11_tranche_matching.md)
12. [Risk monitor](docs/12_risk_monitor.md): warning signs and BDC exposure
13. [Borrower panel and case studies](docs/13_distress_event_study.md)
14. [Early-warning model](docs/14_early_warning_model.md)

(Numbers 5, 8 and 9 were retired.)

## History

The repo keeps only current work. Earlier analyses, outputs and fuller versions of these docs
(sector sub-indices, HHI concentration, the stress events built on `val_ret`, reproducibility logs)
are in `ORIE 5220/miscellaneous/`, and its `MOVES.txt` lists where each file came from. The main
steps:

- 2026-10-24: old and duplicate files moved to `../just in case/`.
- 2026-11-08: everything outside the midterm deck moved out.
- 2026-11-12: `val_ret` and the stress events removed; the early-warning model added.
- 2026-11-13: the model reduced to one outcome (a credit event next quarter: non-accrual or an
  impaired exit) and one logistic
  scorecard; exits classified; valuation outcomes, longer horizons, ridge, lasso, boosted trees and the
  neural network removed.
