# BDC Lending to Data-Center Borrowers

## Overview

This study measures how much BDCs lend to borrowers that build, run or supply data centers,
how that changed from 2023Q1 to 2026Q2, and which managers hold it. Scripts are in
`src/6_data_center/`:

| Step | Command | Writes |
|---|---|---|
| 1. Tag borrowers | `python src/6_data_center/classify_data_center_borrowers.py` | `data/data_center/data_center_borrowers.csv`, `data_center_loan_rows.csv` |
| 2. Trends and BDC exposure | `python src/6_data_center/data_center_analysis.py` | `outputs/data_center_plots/` |
| 3. Time series by scope and manager | `python src/6_data_center/plot_data_center_timeseries.py` | `outputs/data_center_plots/` |
| 4. Largest borrowers and their lenders | `python src/6_data_center/plot_data_center_lenders.py` | `outputs/data_center_plots/` |
| Check: LLM screen of every borrower | `python src/6_data_center/screen_via_chat.py export` / `merge` (or `screen_data_center_borrowers.py` with an API key) | `data/data_center/data_center_screen.csv` |

All steps read `data/final_dataset/FINAL_DATASET.csv`. Steps 2 and 3 also read
`data/universe/BDC_filings_2001_present.csv` for BDC names. Amounts are quarter-end fair values,
i.e. holdings, not new lending.

---

## 1. Tagging: `classify_data_center_borrowers.py`

A borrower is tagged by a curated alias list (`RULES`): regular expressions on
`borrower_canonical`, with an exclusion list (`EXCLUDE`) for names that only look similar, e.g.
"Vantage Specialty" or "LabVantage".

**Inclusion test** (2026-10-20): would the borrower's absence, with no substitute, significantly
impair the physical operation of data centers (power, cooling, mechanical, facilities)?
Web hosting, cloud and IT services delivered from data centers fail it, and so do IT hardware
makers. Borrowers reviewed and rejected are listed in `REVIEWED_NOT_DC` with the reason.

| Tier | Who |
|---|---|
| 1 Operator / developer | Owns or develops data center real estate or capacity |
| 2 AI compute / GPU cloud / miners | GPU clouds, GPU-lease vehicles, AI labs with their own clusters, miners |
| 4 DC services / supply chain | Maintenance, power, cooling, electrical equipment |

Tier 3 (managed hosting) was removed on 2026-10-20.

Each tagged borrower also gets a **scope**. The scopes nest into three estimates:

| Scope | Includes | Estimate |
|---|---|---|
| `core` | Tiers 1 and 2 | Core |
| `support` | Tier 4 where data centers are the firm's main market | Headline = core + support |
| `grey` | `GREY_AREA`: suppliers for which data centers are one of several markets, and operators whose data center business is a sideline | Upper bound = headline + grey |

The current list tags 122 borrower-name variants:

| Tier | core | support | grey |
|---|---|---|---|
| 1 Operator / developer | 39 | — | 6 |
| 2 AI compute | 24 | — | — |
| 4 DC services / supply chain | — | 21 | 32 |

## 2-4. Analysis and charts

- **`data_center_analysis.py`:**
  - `trend_by_function.csv`: quarterly fair value, cost and net new cost by function.
  - `bdc_exposure.csv`: each BDC's latest-quarter exposure in dollars and as a % of its
    portfolio.
  - `bdc_exposure_by_manager.csv`: the same, with each manager's BDCs combined.
  - `bdc_exposure_panel.csv`: BDC by quarter.
  - Four charts.
  - **Base:** these tables count every tagged borrower, i.e. the upper-bound set. BDCs are
    grouped by asset manager with `manager()` from `src/1_data_processing/manager_family.py`
    (shared with crowding and dispersion; see `11_tranche_matching.md`): a list of name
    patterns plus `CIK_FAMILIES`. A BDC that matches none is its own manager. The list was
    extended on 2026-10-23 (TCW, PennantPark, Oaktree, SLR, Sixth Street and others), so the
    manager charts can group a few more BDCs than the saved outputs, which have not been rerun.
- **`plot_data_center_timeseries.py`:**
  - **Totals:** splits loans into the three scopes and the nested estimates
    (`dc_loans_timeseries.csv`, `dc_loans_total.png`). A loan is a tagged row that is not equity
    and that carries a principal, a rate or a spread.
  - **By manager:** charts exposure by manager in dollars and as a % of portfolio, for the 5
    largest managers plus all others (`dc_loans_by_bdc.png`, `dc_exposure_share.png`). These
    charts count every tagged investment, of any scope or instrument, like the other manager
    charts.
- **`plot_data_center_lenders.py`:** the 15 largest data-center borrowers at the latest quarter,
  each bar split by lending manager (`dc_top_borrowers.png`, `dc_borrowers.csv`,
  `dc_borrower_lenders.csv`).

## Results on the current data

| | 2023Q1 | 2026Q2 |
|---|---|---|
| Core | $159M | $3,153M |
| Headline (core + support) | $280M | $5,013M |
| Upper bound (+ grey) | $329M | $6,608M |
| Headline as % of all BDC fair value | 0.1% | 1.0% |
| Headline borrowers / BDCs | 9 / 16 | 36 / 75 |

Largest managers at 2026Q2 (upper-bound set, all their BDCs combined):

| Manager | $M | % of manager's BDC portfolio |
|---|---|---|
| Ares | 1,367 | 2.9% |
| Blackstone | 1,236 | 1.3% |
| Apollo | 1,070 | 3.4% |
| Blue Owl | 584 | 0.9% |
| Goldman Sachs | 422 | 1.8% |

---

## The LLM screen (a check on the alias list)

The alias list only finds borrowers someone already knew about. The screen asks Claude about
every distinct `borrower_canonical` (15,419), given its name, the BDCs' descriptions and the
reported industry. Claude returns a tier (1 to 4, `none` or `unknown`), a confidence and a
reason.

The result is a review list, not a label. A candidate has to be confirmed before it is added
to `RULES`.

- **`screen_via_chat.py export`** writes prompts and 16 chunks of borrowers to
  `data/data_center/data_center_screen_chat/`. Each chunk goes into a claude.ai chat, and the
  reply is saved under `results/`.
- **`screen_via_chat.py merge`** reads the replies, counts every borrower in a replied chunk that
  Claude did not flag as `none`, and writes `data_center_screen.csv`.
- **Uncertain borrowers:** the merge also lists the `unknown` ones, and flags below 0.7
  confidence that the alias list does not cover, in `chunks/verify.txt`. They are meant for a
  second chat with web search on, using `PROMPT_VERIFY.md`.
- **`screen_data_center_borrowers.py`** does the same through the API: it needs
  `ANTHROPIC_API_KEY` and resumes from a checkpoint.

**Status:** all 16 chunks have replies. The web-search check has not been run: `verify.txt`
lists 96 borrowers, and there is no `results/verify.txt` yet. `data_center_screen.csv` was
regenerated on 2026-10-22 with the current alias list.

---

## Limitations

- **Tags follow the alias list, so they miss unknown borrowers.** The LLM screen is the check
  for that, and its candidates still need review.
- **The scope of a supplier is a judgment call.** The headline estimate leaves out the grey area
  on purpose, and the upper bound includes it.
- **Amounts are fair values of holdings.** A rise can come from new loans, from BDCs entering the
  panel, or from marks. `trend_by_function.csv` reports net new cost to help separate these.
