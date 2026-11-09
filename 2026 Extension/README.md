# Cornell x Marathon Capstone: Early Warning Signals in Private Credit

This project builds a loan-level dataset of BDC (business development company) holdings from
the schedules of investments in SEC 10-K and 10-Q iXBRL filings, 2023Q1 to 2026Q2, and uses it
for the midterm presentation (*Examining Early Warning Signals in Private Credit*, 5 Nov 2026):

- market context: BDC growth and the cleaned dataset (slides 2-7);
- a flow-based BDC loan index, benchmarked against the CDLI and CDLI-S (slide 8);
- stress events: a borrower's fair value falling 19% or more in a quarter (slides 5, 9, 10, 23, 24);
- three borrower warning signs (crowding, disagreement on value, PIK) and each BDC's
  exposure to flagged borrowers (slides 11-15);
- BDC lending to data-center borrowers (slides 17-18).

**All analysis reads one dataset: `data/final_dataset/FINAL_DATASET.csv`.**

The repo holds only what the midterm deck uses. Earlier analyses that are not in the deck
(sector sub-indices, HHI concentration, PIK provenance, standalone dispersion panels, the
evaluation layer, older meeting decks, notebooks and run logs) were moved out on 2026-11-08 to
`ORIE 5220/miscellaneous/` (`../../../../miscellaneous` from here). Its `MOVES.txt` lists each
file's original path, so any of them can be put back.

## Layout

```
├── src/                         modules, one folder per stage
│   ├── 1_data_processing/       SEC download, iXBRL parsing, cleaning, labelling; paths.py
│   ├── 2_index_construction/    flow-based market index (slide 8)
│   ├── 3_concentration_crowding/  crowding measures (imported by the risk monitor)
│   ├── 4_val_dispersion/        valuation dispersion (imported by the risk monitor)
│   ├── 5_pik/                   PIK metrics (imported by the risk monitor)
│   ├── 6_data_center/           data-center borrower tagging and charts (slides 17-18)
│   └── 8_risk_monitor/          warning signs, BDC exposure, stress-event study (slides 5, 9-15, 23)
├── scripts/                     run_*.py entry points
├── tests/                       unit tests
├── docs/                        detailed write-up of each stage
├── data/                        datasets (not committed, see data/.gitignore)
├── outputs/                     the tables and plots behind the deck
├── presentations/               slide template
├── cache/                       SEC downloads: all_metadata.csv and filings per CIK (a partial set)
└── requirements.txt
```

Folders 3, 4 and 5 have no output of their own in the deck. They stay because
`src/8_risk_monitor/monitor_panel.py` imports them to build the loan-quarter panel, and the
tests import them too. On 2026-11-08 the code was cut to what the deck, the data pipeline and the
tests use (`7_sector/`, `benchmarks.py` and `evaluation.py` went, with unused functions elsewhere);
the full versions are in `ORIE 5220/miscellaneous/code_before_simplify_2026-11-08/`.

`src/1_data_processing/paths.py` defines every data, output and code location. Modules and
scripts add `src/1_data_processing/` to `sys.path` to import it, and call
`paths.add_code_dirs_to_path()` to import modules from other `src/` folders. Run everything from
the project root.

## Slide map

| Slide | What | Script | Output |
|---|---|---|---|
| 2 | US private credit (Federal Reserve FEDS Note) | none: screenshot of the Fed's chart | `outputs/eda_plots/private credit growth fed.png` |
| 3 | Global private credit vs BDCs | `src/1_data_processing/bdc_growth.py` | `outputs/eda_plots/private_credit_growth.png` |
| 4 | BDC fair value and count | `src/1_data_processing/bdc_growth.py` | `outputs/eda_plots/bdc_growth.png` |
| 5 | Markdown dollars in the worst borrowers; non-accrual share | `scripts/run_tail_concentration.py` | `outputs/risk_monitor/distress/tail_concentration_timeseries.png`, `tail_concentration_markdown_total.png`, `tail_concentration_nonaccrual_timeseries.csv` |
| 6 | Filings collected, read and kept | `src/1_data_processing/plot_data_coverage.py` | `outputs/index_construction_plots/par_retention.csv`, `filings_by_type.png`, `parsing_coverage.png`. "2,021 read" is the parsed-filings total the script prints ("2,232 attempted, 2,021 parsed"); `par_retention.csv`'s Read row counts 2,017, the parsed filings that yield at least one position |
| 7 | Cleaned dataset summary | `src/1_data_processing/dataset_overview.py` | `outputs/eda_plots/dataset_overview.png` (+ `.csv`, `_sectors.csv`). "87% loans" is the share of rows with a principal amount (`dataset_overview.csv`: 0.865). "~96% private" comes from a separate LLM check of the borrowers, not from this repo |
| 8 | Index vs CDLI and CDLI-S | `src/2_index_construction/index_construction.py` | `outputs/index_construction_plots/index_vs_cdli_panels.png`, `index_vs_cdli_stats.csv` |
| 9 | Distribution of quarterly value changes; Athletico | `scripts/run_tail_concentration.py`; `scripts/run_distress_event_study.py` | `outputs/risk_monitor/distress/markdown_distribution.png`, `drop_cutoffs.csv`; `case_studies/athletico_management_marks.png` |
| 10 | Medallia timeline | `scripts/run_distress_event_study.py` | `outputs/risk_monitor/distress/case_studies/medallia_marks.png` (par-weighted average across the lender families still holding the loan in 2026Q2; Apollo left before then and is not drawn; `CASE_AVERAGE` in `src/8_risk_monitor/charts.py`) |
| 11 | Share of borrowers flagged (~8%) | `scripts/run_risk_monitor.py` | `outputs/risk_monitor/signals/warning_tail_summary.csv`, `borrower_warning_flags.csv` |
| 12 | Crowding | `scripts/run_risk_monitor.py` | `outputs/risk_monitor/signals/warning_crowding.png` |
| 13 | Disagreement on value; WebPT | `scripts/run_risk_monitor.py`; `scripts/run_distress_event_study.py` | `outputs/risk_monitor/signals/warning_disagreement.png`; `case_studies/webpt_marks.png` |
| 14 | PIK trend | `scripts/run_risk_monitor.py` | `outputs/risk_monitor/signals/warning_pik.png` |
| 15 | BDC exposure to flagged borrowers | `scripts/run_risk_monitor.py` | `outputs/risk_monitor/signals/bdc_exposure.png`, `bdc_exposure_by_quarter.csv` |
| 17 | Data-center lending over time | `src/6_data_center/plot_data_center_timeseries.py` | `outputs/data_center_plots/dc_loans_total.png`, `dc_loans_timeseries.csv` |
| 18 | Data-center borrowers by sector | `src/6_data_center/plot_data_center_sectors.py` | `outputs/data_center_plots/dc_sector_mix.png` (+ `.csv`, `dc_sector_fills.csv`) |
| 19 | "88% of stress events have no second lender value" | `scripts/run_distress_event_study.py` | Not saved: the share of `distress_events.csv` events with no valuation dispersion (`V1_range`) in the event quarter, 87.5% of 551 (see `docs/13_distress_event_study.md`) |
| 23 | Market move vs stress cutoff | `scripts/run_tail_concentration.py` | `outputs/risk_monitor/distress/market_move.png` |
| 24 | News check of the case studies | `scripts/run_distress_event_study.py` (+ hand-made sources) | `case_studies/` for BVI Medical and LogRhythm; `data/reference/distress_event_validation.csv`; `outputs/risk_monitor/distress/event_validation_multi_family.csv` (hand-made, not rebuilt by any script) |

## Setup

Python 3.11, then:

```
pip install -r requirements.txt
```

The versions are pinned to the ones the outputs were made with, so plots reproduce byte for byte.
Use Python 3.11: the pinned pandas 2.0.2 has no installable package for 3.12 or later.

The LLM steps (`get_seniority_debttype.py`, `screen_data_center_borrowers.py`) need an
Anthropic key in the `ANTHROPIC_API_KEY` environment variable. Their results are cached in
`data/` (and the data-center screen's decisions are written into
`classify_data_center_borrowers.py`), so the rest of the pipeline runs without a key.

## Datasets and reproducing the outputs

The datasets are not in the repo (`data/` and `cache/` are git-ignored); they are shared
separately. Put each file at the path shown:

| File | Size | What it is |
|---|---|---|
| `data/reference/` (10 files) | 0.5 MB | CDLI and CDLI-S benchmarks, FX rates, SOFR and base rates, daily EFFR/SOFR (`Search.xlsx`), `bdc_type.csv`, and the hand-made `case_studies.csv` and `distress_event_validation.csv` |
| `data/parsed/nonaccrual/` (4 files) | 8 MB | Non-accrual footnotes scanned from the XBRL instances (read by step 8) |
| `data/parsed/maturity_html/maturity_from_html.csv` | 39 MB | Maturities read from the HTML schedules (read by step 8) |
| `data/universe/BDC_filings_2001_present.csv`, `BDC_intervals.csv` | 0.1 MB | BDC universe (step 1) |
| `cache/all_metadata.csv` | 2 MB | Filing metadata (step 2) |
| `data/ixbrl/combined/wide_all.csv` | 1.5 GB | Parsed filings (step 3) |
| `data/cleaned/rate_swap_fixes_2026-10-19.csv` | 24 KB | Rate-fix log replayed by step 8 |
| `data/enriched/borrower_name_cleaned.csv` | 13 MB | Teammates' borrower lookup |
| `data/enriched/_seniority_sector_llm_checkpoint.csv` | 15 MB | Cached LLM sector answers |
| `data/final_dataset/labels_by_identifier.csv` (+ `labels_validated.json`, `AUDIT_SUMMARY.md`) | 155 MB | Validated labels |
| `data/data_center/data_center_screen_chat/` | 8 MB | Saved data-center screen replies (only needed to rerun the screen) |

Then run, from the project root:

```
python scripts/run_preprocessor.py                          # step 4
python scripts/run_ixbrl_pipeline.py                        # step 6
python src/1_data_processing/build_final_enriched.py        # step 7
python src/1_data_processing/build_final_dataset.py         # step 8 -> FINAL_DATASET.csv
python src/1_data_processing/plot_data_coverage.py          # slide 6
python src/1_data_processing/bdc_growth.py                  # slides 3-4
python src/1_data_processing/dataset_overview.py            # slide 7
python src/2_index_construction/index_construction.py       # slide 8
python src/6_data_center/classify_data_center_borrowers.py  # data/data_center/ (slides 17-18, stress study)
python src/6_data_center/plot_data_center_timeseries.py     # slide 17
python src/6_data_center/plot_data_center_sectors.py        # slide 18
python scripts/run_risk_monitor.py                          # slides 11-15 (a few minutes)
python scripts/run_distress_event_study.py                  # slides 9-10, 13, 19, 24
python scripts/run_tail_concentration.py                    # slides 5, 9, 23
```

It needs no network or API key and takes about 25 minutes (steps 4 to 8 about 11; the three
risk-monitor scripts about 4 each).

Steps 1-3 and 5 are left out on purpose. They call the SEC and the central banks, whose data
keeps changing. A rebuild today would pick up filings after 2026Q2, which have no validated
labels, so step 8 would stop.

To run only the analyses, `FINAL_DATASET.csv` (259 MB) alone is enough: put it in
`data/final_dataset/` and start at `plot_data_coverage.py`. The two coverage plots also need
`cache/all_metadata.csv` and `wide_all.csv`.

**Side outputs.** On 2026-11-08 the scripts above were trimmed so they write only what the deck
uses; the files they used to write besides (return decompositions, extra data-center charts, the
risk monitor's market and evaluation layers, the watchlist and event profiles, the pooled tail
chart) are in `miscellaneous/`. One exception, kept on purpose: `run_distress_event_study.py`
profiles every borrower in `data/reference/case_studies.csv`, so it also writes the Premier Imaging
case files (on watch, not in the deck), kept for later case studies. Add a row to that file to
profile another borrower.

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

FINAL_DATASET's `sector_report` column (Software and Business Services split out) comes from
`src/1_data_processing/report_sector.py`, run inside `build_final_dataset.py`.

FINAL_DATASET's `bdc_type`, `non_accrual` and HTML-filled maturity columns come from three
files that step 8 reads and that are shared with the datasets. To rebuild them, run these after
step 7 and before step 8 (see `docs/6_final_dataset.md`):

- `python src/1_data_processing/bdc_type.py`: EDGAR filing histories -> `data/reference/bdc_type.csv`.
  Uses `cache/sec_submissions/` and downloads any BDC missing from it.
- `python src/1_data_processing/nonaccrual_scan.py`: XBRL footnotes -> `data/parsed/nonaccrual/`.
  Needs every filing's XBRL instance in `data/ixbrl/raw_xml/` (kept by step 3). If any is
  missing it stops and leaves the existing files alone.
- `python src/1_data_processing/maturity_html.py download`, then `extract`: HTML schedules ->
  `data/parsed/maturity_html/maturity_from_html.csv`. Needs `cache/html_gz/` or the network.

Tests: `python tests/test_pik_metrics.py`, `python tests/test_position_matching.py`,
`python tests/test_tranches.py` and `python tests/test_nonaccrual_bdc_type.py`

The warning signs and stress events match loans within a borrower on lien, spread and maturity,
and count each manager family as one holder (`src/1_data_processing/tranches.py`,
`manager_family.py`; see `docs/11_tranche_matching.md`).

## Data

| Folder | Contents | Source |
|---|---|---|
| `data/reference/` | `FX.csv`, `SOFR.csv`, `SOFR_augmented.csv`, `cdli.csv`, `cdli-s.csv`, `CDLI-History.xlsx`, `Search.xlsx`, `bdc_type.csv`, `case_studies.csv`, `distress_event_validation.csv` | External downloads; `SOFR_augmented.csv` is built by step 5; the last two are hand-made |
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

**Midterm deck, checked 2026-11-08.** A clean copy holding only `src/`, `scripts/`, `tests/` and the
input files in the dataset table (no saved outputs or intermediate files) ran the 14 commands
above in order, with the pinned environment (Python 3.11). Results, against the saved files:

| Output | Result |
|---|---|
| `ixbrl_clean.csv`, `ixbrl_cleaned_out.csv`, enriched CSV, `FINAL_DATASET.csv`, `data/data_center/` tags | Byte-identical |
| Every file in `outputs/` behind the deck (55 files) | Byte-identical, except the six changed on purpose that day and replaced by the rerun: the case-study charts (slide 10's Medallia chart redrawn as presented, with a par-weighted average; the non-accrual footnote restored on the others) and one row added to `dataset_overview.csv` |
| `tests/` | All four test files pass (26 tests; four PIK tests of removed functions were dropped with them) |
| Not produced by code | Slide 2 (the Federal Reserve's chart), slide 7's "~96% private" (separate LLM check), slide 24's news sources, `data/reference/case_studies.csv`, `distress_event_validation.csv` and `outputs/risk_monitor/distress/event_validation_multi_family.csv` (hand-made) |

The whole run took about 25 minutes. It was repeated after the code was simplified the same day (unused code removed, docstrings shortened): every file came out byte-identical again.

**Full pipeline, checked 2026-10-25** (before the 2026-11-08 cleanup), with a clean copy holding only the code and
the files in the dataset table above, in a new virtual environment installed from
`requirements.txt` (Python 3.11). The full run list of that time (which also included the
analyses now in `miscellaneous/`) was run in order, each step reading only what the step before
wrote, and each output was compared with the saved one. The analyses were then run a second time
and compared with the first.

| Output | Result |
|---|---|
| Steps 4, 6, 7, 8: `ixbrl_clean.csv`, `ixbrl_cleaned_out.csv`, enriched CSV, `FINAL_DATASET.csv` | Byte-identical |
| `bdc_type.csv` (rebuilt from `cache/sec_submissions/`) | Byte-identical |
| Every table and plot in `outputs/` and `data/data_center/` at the time (178 files) | Byte-identical, and identical between two runs |
| `tests/` | 30 tests pass |

Fixes made for this check: the Polars group-bys keep row order (`maintain_order=True`),
so sums no longer change in the last digit between runs; Pillow is pinned, since other versions
write different PNG bytes; and `paths.py` switches printing to UTF-8, since on Windows the scripts
crashed with `UnicodeEncodeError` when their output went to a file.

Steps 1, 2, 3 and 5 call the network and were last checked on 2026-10-22:

| Output | Result |
|---|---|
| Step 1: BDC universe | Same rows, plus 6 BDC filings the SEC posted after the saved file was made. `BDC_intervals.csv` is written with zero-padded CIKs (`0000003898`), and the saved copy has most of them unpadded. |
| Step 2: download (test mode, one company) | Runs; 4 of 4 filings downloaded |
| Step 3: parse (test mode, one company, 2024) | 4 filings, 6,329 rows: identical to the same filings in the saved `wide_all.csv` |
| Step 5: `SOFR_augmented.csv` (rebuilt from the central banks) | Same values; only line endings differ |

Not rerun: the full SEC download and parse (steps 2 and 3, hours), `nonaccrual_scan.py` and
`maturity_html.py` (they need the raw filings), and the LLM steps. Those are cached: the sector
answers in `_seniority_sector_llm_checkpoint.csv`, the label validation in
`labels_by_identifier.csv`, the data-center chat replies, and the files in `data/parsed/`.

**Position matching changed on 2026-10-22, after the check above.** Positions are now matched
across quarters and across BDCs on the validated borrower name, plus seniority for time series,
instead of `investment_identifier` text. BDCs reword that text between filings, and most did so
in their FY2025 10-Ks.

- **Coverage:** positions linked to a prior quarter now carry 86-94% of fair value every quarter,
  against 58-73% before (17% in 2025Q4).
- **Index:** correlation with CDLI 94.6%, tracking error 0.41%, annualized return 10.1%
  (2023Q2-2026Q2; identifier matching gave 94.3%, 0.43%, 9.8%).
- **Old outputs:** the identifier-matched versions are kept outside the repo in
  `../just in case/repo_cleanup_2026-10-24/_archive/`.

## Docs

1. [Pipeline overview](docs/1_pipeline_overview.md): download, parse, preprocess
2. [iXBRL cleaning](docs/2_ixbrl_cleaning.md): rate, currency and value rules
3. [Seniority and sector](docs/3_seniority_sector.md)
4. [Index construction](docs/4_index_construction.md)
6. [Building FINAL_DATASET.csv](docs/6_final_dataset.md): steps 7 and 8
7. [Concentration and crowding](docs/7_concentration_crowding.md)
10. [Data-center lending](docs/10_data_center.md)
11. [Tranche matching and manager families](docs/11_tranche_matching.md)
12. [Risk monitor](docs/12_risk_monitor.md): warning signs, BDC exposure, benchmarks, evaluation
13. [Distressed borrowers](docs/13_distress_event_study.md): stress events (pooled bottom 1%, -19.3%), case studies, news check

Docs 5 (sub-indices), 8 (valuation dispersion), 9 (PIK metrics) and the nowcasting problem
statement were moved to `miscellaneous/docs/`. The numbering is kept so the remaining docs'
cross-references still read correctly.

## Notes

- **How a markdown is measured (changed 2026-11-01).** A loan's quarterly change in value is
  `(FV - FV_prev - (COST - COST_prev)) / (FV_prev + max(COST - COST_prev, 0))`: the change in
  fair value net of new money and repayments, over the capital at risk (last quarter's value
  plus any new money). Until 2026-11-01 the denominator was `FV_prev` alone. When a small loan
  took a large draw (a delayed-draw or revolver funding, or a small tranche refinanced into a
  much larger one), any discount on the new money was divided by the tiny prior value: 36
  borrower-quarters showed losses beyond -100% (BVI Medical 2026Q1: -251%, now -7%). Negative
  fair values (unfunded commitments) were not the cause: loans with a non-positive prior value
  were already excluded. The fix is in all three places that compute the return:
  `src/8_risk_monitor/monitor_panel.py` (loan markdown events),
  `src/8_risk_monitor/distress_event_study.py` (borrower stress events) and
  `src/5_pik/pik_metrics.py` (PIK markdown flags). The rerun swapped 70 of the 804 bottom-1%
  borrower-quarters, cut the distressed-borrower register from 541 to 528 (BVI Medical is no
  longer an event) and moved the pooled worst-1% cutoff from -21% to -19%; the 2026Q1 cutoff is
  now -25% (was -26%). On 2026-11-01 the stress event switched from the per-quarter bottom 1% to
  the pooled cutoff (-19.3% for every quarter): 551 distressed borrowers (29 out, 52 in; Medallia,
  Planview and Cornerstone OnDemand are events again). Details in `docs/13_distress_event_study.md`.

- API keys: set `ANTHROPIC_API_KEY` in the environment. Never commit a key; a local
  `claude_api_key.txt` is git-ignored.
- Old and duplicate files (`_archive/`, `data/_archive/`, teammates' GitHub notebook copies) were
  moved out of the repo to `../just in case/repo_cleanup_2026-10-24/` on 2026-10-24; `MOVES.txt`
  there lists each one. Material not used in the midterm deck was moved to `ORIE 5220/miscellaneous/`
  on 2026-11-08; its `MOVES.txt` lists each file.
- The web-search check of uncertain data-center borrowers was never run: `data/data_center/data_center_screen_chat/chunks/verify.txt`
  lists 96 borrowers for it, and `results/verify.txt` does not exist yet.
