# Cornell x Marathon Capstone: BDC Private Credit Index (2026 Extension)

This project builds a loan-level dataset of BDC (business development company) holdings from
the schedules of investments in SEC 10-K and 10-Q iXBRL filings, 2023Q1 to 2026Q2. On top of
that dataset it builds:

- a flow-based private credit return index, benchmarked against the CDLI;
- sector sub-indices;
- concentration and crowding measures;
- PIK (payment-in-kind) risk metrics;
- valuation dispersion across BDCs;
- a study of BDC lending to data-center borrowers.

**All analysis reads one dataset: `data/final_dataset/FINAL_DATASET.csv`.**

## Layout

```
├── src/                         modules, one folder per stage
│   ├── 1_data_processing/       SEC download, iXBRL parsing, cleaning, labelling; paths.py
│   ├── 2_index_construction/    flow-based market index and return decomposition
│   ├── 3_concentration_crowding/  HHI concentration and crowding
│   ├── 4_val_dispersion/        (empty; the dispersion functions are in outputs/exploratory_analysis.py, as on GitHub)
│   ├── 5_pik/                   PIK provenance, distress amendments, non-cash metrics
│   ├── 6_data_center/           data-center borrower tagging, screening and exposure
│   └── 7_sector/                sector and seniority sub-indices
├── scripts/                     run_*.py entry points
├── tests/                       unit tests
├── notebooks/                   end-to-end pipeline notebook
├── docs/                        detailed write-up of each stage
├── data/                        datasets (not committed, see data/.gitignore)
├── outputs/                     results: tables, plots, run logs, the executed notebook and exploratory_analysis.py
├── presentations/               slide decks
├── cache/                       SEC downloads: all_metadata.csv and filings per CIK (a partial set)
└── requirements.txt
```

`src/1_data_processing/paths.py` defines every data, output and code location. Modules and
scripts add `src/1_data_processing/` to `sys.path` to import it, and call
`paths.add_code_dirs_to_path()` to import modules from other `src/` folders. Run everything from
the project root.

## Setup

Python 3.11, then:

```
pip install -r requirements.txt
```

The versions are pinned to the ones the outputs were made with, so plots reproduce byte for byte.

The LLM steps (`get_seniority_debttype.py`, `screen_data_center_borrowers.py`) need an
Anthropic key in the `ANTHROPIC_API_KEY` environment variable. Their results are cached in
`data/`, so the rest of the pipeline runs without a key.

## Datasets and reproducing the outputs

The datasets are not in the repo (`data/` and `cache/` are git-ignored); they are shared
separately. Put each file at the path shown:

| File | Size | What it is |
|---|---|---|
| `data/reference/` (7 files) | 0.5 MB | CDLI and CDLI-S benchmarks, FX rates, SOFR and base rates, daily EFFR/SOFR (`Search.xlsx`) |
| `data/universe/BDC_filings_2001_present.csv`, `BDC_intervals.csv` | 0.1 MB | BDC universe (step 1) |
| `cache/all_metadata.csv` | 2 MB | Filing metadata (step 2) |
| `data/ixbrl/combined/wide_all.csv` | 1.5 GB | Parsed filings (step 3) |
| `data/cleaned/rate_swap_fixes_2026-10-19.csv` | 24 KB | Rate-fix log replayed by step 8 |
| `data/enriched/borrower_name_cleaned.csv` | 13 MB | Teammates' borrower lookup |
| `data/enriched/_seniority_sector_llm_checkpoint.csv` | 15 MB | Cached LLM sector answers |
| `data/final_dataset/labels_by_identifier.csv` (+ `labels_validated.json`, `AUDIT_SUMMARY.md`) | 155 MB | Validated labels |
| `data/data_center/data_center_screen_chat/` | 8 MB | Saved data-center screen replies |

Then run, from the project root:

```
python scripts/run_preprocessor.py                          # step 4
python scripts/run_ixbrl_pipeline.py                        # step 6
python src/1_data_processing/build_final_enriched.py        # step 7
python src/1_data_processing/build_final_dataset.py         # step 8 -> FINAL_DATASET.csv
python src/1_data_processing/plot_data_coverage.py
python src/2_index_construction/index_construction.py
python src/7_sector/subindex_construction.py
python src/3_concentration_crowding/concentration_crowding.py
python src/1_data_processing/bdc_growth.py
python scripts/run_pik_risk_metrics.py
python src/6_data_center/classify_data_center_borrowers.py
python src/6_data_center/data_center_analysis.py
python src/6_data_center/plot_data_center_timeseries.py
python src/6_data_center/plot_data_center_lenders.py
python src/6_data_center/screen_via_chat.py merge
```

This rebuilds `FINAL_DATASET.csv` and every file in `outputs/` byte for byte, except that a few
PIK tables can come out in a different row order. It takes about 20 minutes and needs no network
or API key.

Steps 1-3 and 5 are left out on purpose. They call the SEC and the central banks, whose data
keeps changing. A rebuild today would pick up filings after 2026Q2, which have no validated
labels, so step 8 would stop.

To run only the analyses, `FINAL_DATASET.csv` (259 MB) alone is enough: put it in
`data/final_dataset/` and start at `plot_data_coverage.py`. The two coverage plots also need
`cache/all_metadata.csv` and `wide_all.csv`.

## Pipeline

| Step | Command | Writes |
|---|---|---|
| 1. BDC universe from EDGAR master.idx | `python scripts/run_bdc_universe.py` | `data/universe/BDC_filings_2001_present.csv`, `BDC_intervals.csv` |
| 2. Download 10-K/10-Q filings | `python scripts/run_downloader.py` | `cache/` |
| 3. Parse iXBRL instance documents | `python scripts/run_ixbrl_parser.py` | `data/ixbrl/combined/wide_all.csv` |
| 4. Preprocess to one row per position | `python scripts/run_preprocessor.py` | `data/parsed/ixbrl_clean.csv` |
| 5. Base rates (EURIBOR, SONIA, CORRA, ...) | `python src/1_data_processing/augment_rates.py` | `data/reference/SOFR_augmented.csv` |
| 6. Clean rates, currencies and values | `python scripts/run_ixbrl_pipeline.py` | `data/cleaned/ixbrl_cleaned_out.csv` |
| 7. Borrower, seniority and sector labels | `python src/1_data_processing/build_final_enriched.py` | `data/enriched/data_private_credit_FINAL_enriched_final.csv` |
| 8. Apply validated labels and rate fixes | `python src/1_data_processing/build_final_dataset.py` | `data/final_dataset/FINAL_DATASET.csv` |

Steps 2 and 3 call the SEC and take hours; a full download needs tens of GB. Step 1 takes a few
minutes. `run_downloader.py` and `run_ixbrl_parser.py` have a `test` mode (one company, 2024),
and `run_preprocessor.py` has one (2024Q1-Q2 only). Set `mode = "test"` in the script's `main()`.

## Analyses

All read `FINAL_DATASET.csv` and write to `outputs/`.

| Analysis | Command | Writes |
|---|---|---|
| Market index vs. CDLI, return decompositions (market, Technology, senior loans), senior loans vs. CDLI-S, index vs. SOFR | `python src/2_index_construction/index_construction.py` | `outputs/index_construction_plots/` |
| Data coverage: filings by type, parsing success by quarter | `python src/1_data_processing/plot_data_coverage.py` | `outputs/index_construction_plots/` |
| Sector sub-indices | `python src/7_sector/subindex_construction.py` | `outputs/sector_subindices_plots/` |
| Concentration and crowding (C1-C4 by manager family) | `python src/3_concentration_crowding/concentration_crowding.py` | `outputs/concentration_crowding_plots/`, `outputs/risk_signals/crowding_borrower_metrics.csv` |
| BDC universe growth | `python src/1_data_processing/bdc_growth.py` | `outputs/eda_plots/` |
| PIK risk metrics | `python scripts/run_pik_risk_metrics.py` | `outputs/pik/`, `outputs/pik_plots/` |
| Data-center tagging | `python src/6_data_center/classify_data_center_borrowers.py` | `data/data_center/` |
| Data-center exposure | `python src/6_data_center/data_center_analysis.py`, `plot_data_center_timeseries.py`, `plot_data_center_lenders.py` | `outputs/data_center_plots/` |
| Data-center screen (from saved chat replies) | `python src/6_data_center/screen_via_chat.py merge` | `data/data_center/data_center_screen.csv` |
| Valuation dispersion (V1, V2, V4 on matched tranches) | `python scripts/run_dispersion_signals.py`; notebook views in `outputs/exploratory_analysis.py` | `outputs/risk_signals/dispersion_tranches.csv`, `tranche_match_coverage.csv` |

| Risk monitor: benchmark table, peer-relative markdown outcome, signal and evaluation layers | `python scripts/run_risk_monitor.py` | `outputs/risk_monitor/` |
| Distressed-borrower event study (persistent 20%+ markdowns) | `python scripts/run_distress_event_study.py` | `outputs/risk_monitor/distress/` |

PIK, dispersion and crowding match loans within a borrower on lien, spread and maturity, and
count each manager family as one holder (`src/1_data_processing/tranches.py`,
`manager_family.py`; see `docs/11_tranche_matching.md`).

Tests: `python tests/test_pik_metrics.py`, `python tests/test_position_matching.py`,
`python tests/test_tranches.py` and `python tests/test_nonaccrual_bdc_type.py`

FINAL_DATASET's `bdc_type` and `non_accrual` columns come from
`python src/1_data_processing/bdc_type.py` (EDGAR filing histories -> `data/reference/bdc_type.csv`)
and `python src/1_data_processing/nonaccrual_scan.py` (XBRL footnotes -> `data/parsed/nonaccrual/`),
then `build_final_dataset.py`; see `docs/6_final_dataset.md`.

## Data

| Folder | Contents | Source |
|---|---|---|
| `data/reference/` | `FX.csv`, `SOFR.csv`, `SOFR_augmented.csv`, `cdli.csv`, `cdli-s.csv`, `CDLI-History.xlsx`, `Search.xlsx` | External downloads; `SOFR_augmented.csv` is built by step 5 |
| `data/universe/`, `data/ixbrl/`, `data/parsed/`, `data/cleaned/` | Pipeline steps 1 to 6 | Code |
| `data/enriched/` | Step 7 output and its inputs: `borrower_name_cleaned.csv` (teammates' borrower lookup), `_seniority_sector_llm_checkpoint.csv` (cached LLM sector answers) | Code, plus those two inputs |
| `data/final_dataset/` | `FINAL_DATASET.csv`, `labels_by_identifier.csv`, `labels_validated.json`, `AUDIT_SUMMARY.md` | Label validation, see below |
| `data/data_center/` | Data-center tags and screen results | Code; screen replies saved in `data_center_screen_chat/` |

**How `FINAL_DATASET.csv` is made.** Step 8 (`build_final_dataset.py`) takes the step 7 output
and does two things:

- **Validated labels:** it replaces borrower, seniority and sector with the decisions in
  `data/final_dataset/labels_by_identifier.csv`. The old values are kept in `*_prev` columns,
  next to each decision's confidence, evidence and action.
- **Rate fixes:** it applies the 92 fixes logged in `data/cleaned/rate_swap_fixes_2026-10-19.csv`.

The labels themselves come from a one-off LLM validation. It followed
`src/1_data_processing/final_validation_prompt.md`, ran outside this code, and is described in
`data/final_dataset/AUDIT_SUMMARY.md`. The step stops with an error if the step 7 output no
longer matches the values the labels were validated against.

## Reproducibility

Checked on 2026-10-22 with a clean copy of the repo. Every generated file was deleted, then the
commands above were run in order, each step reading only what the step before wrote. Each output
was compared with the saved one.

| Output | Result |
|---|---|
| Step 1: BDC universe | Same rows, plus 6 BDC filings the SEC posted after the saved file was made. `BDC_intervals.csv` is written with zero-padded CIKs (`0000003898`), and the saved copy has most of them unpadded. |
| Step 2: download (test mode, one company) | Runs; 4 of 4 filings downloaded |
| Step 3: parse (test mode, one company, 2024) | 4 filings, 6,329 rows: identical to the same filings in the saved `wide_all.csv` |
| Step 4: `ixbrl_clean.csv` | Byte-identical |
| Step 5: `SOFR_augmented.csv` (rebuilt from the central banks) | Same values; only line endings differ |
| Steps 6 to 8: `ixbrl_cleaned_out.csv`, enriched CSV, `FINAL_DATASET.csv` | Byte-identical |
| Index, sub-index, concentration and BDC-growth plots and tables | Byte-identical |
| PIK tables and plots | Same values; three tables come out in a different row order |
| Data-center tags, screen and charts | Byte-identical. Three tables show one BDC under its new name from step 1 (Investcorp US Private Credit BDC II is now a "Liquidating Trust"). |
| Valuation dispersion functions, `tests/test_pik_metrics.py` | Run without errors; tests pass |

Not rerun: the full SEC download and parse (steps 2 and 3, hours), and the LLM steps. Those are
cached: the sector answers in `_seniority_sector_llm_checkpoint.csv`, the label validation in
`labels_by_identifier.csv`, and the data-center chat replies.

**Position matching changed on 2026-10-22, after the check above.** Positions are now matched
across quarters and across BDCs on the validated borrower name, plus seniority for time series,
instead of `investment_identifier` text. BDCs reword that text between filings, and most did so
in their FY2025 10-Ks.

- **Coverage:** positions linked to a prior quarter now carry 86-94% of fair value every quarter,
  against 58-73% before (17% in 2025Q4).
- **Index:** correlation with CDLI 94.6%, tracking error 0.41%, annualized return 10.1%
  (2023Q2-2026Q2; identifier matching gave 94.3%, 0.43%, 9.8%).
- **Crowding and PIK:** results change materially (see `docs/7_concentration_crowding.md` and
  `docs/9_pik_metrics.md`).
- **Old outputs:** the index, sub-index, concentration and PIK outputs were regenerated. The
  identifier-matched versions are kept locally in `_archive/` (not in the repo).

## Docs

1. [Pipeline overview](docs/1_pipeline_overview.md): download, parse, preprocess
2. [iXBRL cleaning](docs/2_ixbrl_cleaning.md): rate, currency and value rules
3. [Seniority and sector](docs/3_seniority_sector.md)
4. [Index construction](docs/4_index_construction.md)
5. [Sub-index construction](docs/5_subindex_construction.md)
6. [Building FINAL_DATASET.csv](docs/6_final_dataset.md): steps 7 and 8
7. [Concentration and crowding](docs/7_concentration_crowding.md)
8. [Valuation dispersion](docs/8_valuation_dispersion.md)
9. [PIK risk metrics](docs/9_pik_metrics.md)
10. [Data-center lending](docs/10_data_center.md)
11. [Tranche matching and manager families](docs/11_tranche_matching.md)
12. [Risk monitor](docs/12_risk_monitor.md): benchmarks, markdown and non-accrual outcomes, evaluation
13. [Distressed borrowers](docs/13_distress_event_study.md): crowding, concentration, dispersion and PIK around a persistent 20%+ markdown
- [Nowcasting problem statement](docs/nowcasting_problem.md)

## Notes

- API keys: set `ANTHROPIC_API_KEY` in the environment. Never commit a key; a local
  `claude_api_key.txt` is git-ignored.
- `_archive/` (local only, git-ignored) holds old and duplicate files. `_archive/MOVES.txt` lists
  every file moved during the October 2026 reorganization.
- `notebooks/index_construction_pipeline_github.ipynb` and `outputs/index_construction_pipeline_executed_github.ipynb`
  are teammates' copies from GitHub. They use GitHub's older paths and are kept for reference only.
- The web-search check of uncertain data-center borrowers was never run: `data/data_center/data_center_screen_chat/chunks/verify.txt`
  lists 96 borrowers for it, and `results/verify.txt` does not exist yet.
