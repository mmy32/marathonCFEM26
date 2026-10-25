# Nowcasting the Private Credit Index: Problem Statement

Branch: `nowcasting`. Scoping document; implementation roadmap lives in `TODO-max.md`.

---

## 1. Motivating Problem

Our private credit indices (spreads, YTM, PIK share, marks) are built from BDC 10-Q/10-K iXBRL schedules of investments. These filings arrive roughly 45-60 days after quarter-end, so the latest index reading is always 2-4 months stale. A deterioration that starts in week 3 of a quarter is invisible in our data until the filings land in week 17 or later, and the CDLI benchmark is published on a similar quarterly lag.

This matters most for the use case the index exists to serve: spotting stress in private credit early. Private loans are marked infrequently and are not traded, so the filing lag is the main delay in seeing how marks, spreads, and non-accruals are moving. By the time the quarterly data confirms a problem, public credit markets have usually already repriced.

Part of the signal arrives earlier than the filing. Many BDCs announce quarter-end NAV per share in an earnings release, typically weeks before the 10-Q is filed. NAV is the aggregate of the same fair-value marks that our index is built from, so it carries information about the quarter's index value well before the structured data exists. Today we do not use it.

## 2. Proposal

Build a weekly nowcast of the current quarter's private credit index, update it as information arrives, and replace it with the realized value once filings are processed.

| Element | Specification |
|---|---|
| **Target variable** | Overall private credit index level / spread, as built by the existing index pipeline |
| **Horizon** | Current quarter only (nowcast of quarter *t* during quarter *t* and the post-quarter-end filing gap) |
| **Output frequency** | Weekly |
| **Truth** | The eventual private credit index values built from the filings, and the CDLI |
| **Metrics** | MAE and RMSE against each truth series |
| **Use case** | Early detection of private credit issues at the sector level |

**Information set.** The nowcast combines two kinds of inputs:

1. *BDC-reported NAV from earnings releases (the key input).* As each BDC reports NAV before its 10-Q, the nowcast incorporates it, so accuracy should step up during the post-quarter-end earnings window.
2. *High-frequency market data* available within the quarter: BDC prices and P/NAV discounts, public leveraged loan and HY spreads, and loan ETF prices. These drive the nowcast before earnings releases begin.

All inputs are used only as of their real release date (real-time discipline), so the backtest sees what a user would have seen at each weekly nowcast date.

**Evaluation.** The nowcast is scored against two truths, because they answer different questions:

- *Our filing-based index* measures whether the nowcast recovers the quantity we actually build.
- *CDLI* is an independent benchmark and guards against the nowcast only matching a quirk of our own construction.

MAE and RMSE are reported for each. Because both truths are quarterly while the output is weekly, each week's nowcast is scored against the realized value for the quarter it targets, and results are reported as a function of week within the quarter. Baselines are last-quarter carry-forward, AR(1), and a public-spread-only regression. A model that does not beat carry-forward is not useful.

**Use case and sector level.** The target is the overall index, but the purpose is sector-level monitoring. The overall index is the first stage because it has the most data, the least noise, and a CDLI counterpart. The sector extension is the stage that serves the use case, and we should expect it to be harder: sector indices are built from fewer BDCs, the BDC-level NAV signal is diluted by sector mix, and some sectors (e.g., Software) are specific to private credit and are less visible in public markets. Part of the evaluation is checking where the signal holds at sector level and where it does not.

**Constraints and risks.**

- Small sample: about 14 quarterly observations (2023Q1 onward). Models must be parsimonious (bridge equations, U-MIDAS) and out-of-sample results will have wide error bands.
- NAV is a noisy proxy: it also reflects leverage, fees, buybacks, and realized gains, not just marks on loans. The mapping from NAV changes to index changes has to be estimated, not assumed.
- Coverage of NAV-in-earnings-release varies by BDC and over time. The number of BDCs reporting early at each week is itself an input to how much weight the signal gets.
- Revisions: filings can be amended after the first release. We score against the first filing-based value and track revisions separately.

## 3. High-Level Steps

1. **Fix the target and truth series.** Confirm the overall index and spread definitions, assemble the quarterly truth panel (index from filings, CDLI), and align quarters.
2. **Inventory data and publication lags.** List the candidate predictors, record each series' release lag, and count how many BDCs report NAV in the earnings release and how many weeks before the 10-Q. Decide the data source and add loaders under `src/`.
3. **Build the NAV-from-earnings-release dataset.** Collect quarter-end NAV per share and report dates per BDC (8-K / earnings release), with the as-of date stamped on each observation.
4. **Establish baselines and the backtest harness.** Carry-forward, AR(1), and public-spread regression, evaluated in an expanding-window, real-time backtest with weekly nowcast dates and tests under `tests/`.
5. **Build the nowcast models.** Start with bridge equations and U-MIDAS combining market data and the NAV signal. Move to a mixed-frequency dynamic factor model only if the simpler models leave a clear gap.
6. **Evaluate.** Report MAE and RMSE against both truths, accuracy by week within the quarter, the gain from the NAV signal versus market data alone, and results on stress episodes (2023 regional bank stress, 2025-26 PIK rise).
7. **Extend to sectors.** Repeat at the sector level, report where the signal weakens, and assess whether the nowcast flags sector deterioration earlier than the filings do.
8. **Deliver.** `scripts/run_nowcast.py` runner (logic in `src/`), output CSVs, a plot of nowcast versus realized by quarter, and a short write-up of method, results, and limitations.

## Open Questions

1. Is CDLI scored on its level, its quarterly return, or both? The index outputs are levels/spreads while CDLI is published as returns, so one needs a consistent transform.
2. Are paid sources (Bloomberg, LSTA, Pitchbook) available, or are we restricted to free and public data?
3. How should a week-by-week nowcast be scored when truth is quarterly: all weeks pooled, or reported by week-in-quarter only?
