# Concentration and Crowding Across BDCs

## Overview

`src/3_concentration_crowding/concentration_crowding.py` measures two things per quarter:

- **Concentration:** how evenly the BDC market's assets are spread across managers. Measured
  with the Herfindahl-Hirschman Index (HHI), the effective number of BDCs and the top-N share.
- **Crowding:** how much of the market sits in borrowers that two or more BDCs lend to at the
  same time, and which borrowers drive it.

Both signal where credit risk is shared. A concentrated market leans on a few managers' marks;
a crowded one means the same loans sit on many balance sheets.

```
python src/3_concentration_crowding/concentration_crowding.py
```

Reads `data/final_dataset/FINAL_DATASET.csv` through
`index_construction.load_and_prepare_investment_data()` and
`compute_final_interest_rates()`, adds each BDC's manager family (`manager_family.add_family`,
see `11_tranche_matching.md`), then writes three plots to `outputs/concentration_crowding_plots/`
and one table to `outputs/risk_signals/`:

| File | Shows |
|---|---|
| `concentration_hhi.png` | Effective number of BDCs (1 / HHI) and top-10 AUM share |
| `hhi_vs_equal_weight.png` | Actual HHI against the HHI of an equal split across active BDCs (1 / n) |
| `crowding_aum_share.png` | Share of market AUM in borrowers held by two or more manager families |
| `crowding_borrower_metrics.csv` | Borrower-level crowding signals C1-C4, one row per borrower-quarter |

It also prints the quarterly concentration and crowding tables, and the most-crowded positions
in the latest quarter.

Concentration here is the market-level measure (how BDC assets spread across BDCs). The Risk
Dimensions doc treats that as context only; its lender-side concentration (L1-L4, each BDC's
own portfolio concentration) is not built yet.

---

## Concentration

**`compute_cik_aum_by_quarter(df)`**: AUM per (CIK, quarter) is the sum of fair value (`FV`).
No market weighting, since concentration is about how assets are distributed, not about
returns.

**`compute_market_concentration(aum_by_cik_q, top_ns=(5, 10))`**: per quarter, on AUM shares
among CIKs with positive total AUM:

- `hhi`: sum of squared shares, on a 0 to 1 scale, not the conventional 0 to 10,000.
- `effective_n_ciks` = 1 / HHI: how many equal-sized managers would give the same
  concentration.
- Top-5 and top-10 AUM shares.
- `n_active_ciks`, used for the equal-weight comparison.

A CIK whose net fair value is negative in a quarter is left out of that quarter's share base.
Otherwise it would inflate the other managers' shares through a negative denominator.

## Crowding

Holders are **manager families**, not CIKs: funds of one manager share one valuation and
usually co-invest, so they count once (Risk Dimensions doc, C1). Pass `cik_col="cik"` to count
individual BDCs.

**`compute_position_crowding(df, min_holders=2, cik_col="family")`**: matches holders on the
validated borrower name (`borrower_canonical`), trimmed and upper-cased, within a quarter. Rows
with no borrower name are dropped so they cannot all match each other. Only rows with FV > 0
count as held positions. `investment_identifier` is not used, because each BDC words it
differently. Under identifier matching, First Brands Group in 2025Q2 had 15 BDC lenders but
showed no crowding at all. Returns:

- `position_level`: one row per (quarter, borrower) with the number of distinct holders and
  total FV.
- `quarterly_summary`: per quarter, the share of market AUM in positions with at least
  `min_holders` holders (`crowded_aum_share`).

**`compute_borrower_crowding_metrics(df)`**: the borrower-level crowding signals, on funded
debt (par > 0, FV > 0), one row per (quarter, borrower):

| Code | Column | Definition |
|---|---|---|
| C1 | `n_families` | Manager families holding the borrower (`n_ciks`: BDCs) |
| C2 | `lender_hhi` | Sum of squared family shares of BDC-held par (1 = one family holds it all) |
| C3 | `top_share` | Largest family's share of par |
| C4 | `d_families_4q` | C1 now minus C1 four quarters earlier (absent then = 0; null for 2023Q1-Q4) |
| | `log_par` | Log of par held by all BDCs, the size control |
| | `C1_pct` ... `C4_pct` | Within-quarter percentiles |

C5 (portfolio overlap between BDCs) and the classification of exits are not built yet.

**`top_crowded_positions(position_level, quarter, top_n=15)`**: the most widely held names in
one quarter, ranked by number of holders, with ties broken by FV. Use it to see which borrowers
drive a rise in crowding.

### Why the two measures net fair value differently

Concentration nets FV per manager first, so a write-down offsets the rest of that manager's
book. Crowding filters to FV > 0 per position, because a non-positive mark is not a real
holding. The difference is well under 1% of AUM on this panel, and only a handful of
(CIK, quarter) pairs change inclusion.

---

## Results on the current data

| | 2023Q1 | 2024Q1 | 2025Q2 | 2026Q2 |
|---|---|---|---|---|
| Active BDCs | 97 | 132 | 161 | 171 |
| Market AUM ($bn) | 241 | 306 | 461 | 528 |
| HHI | 0.068 | 0.051 | 0.047 | 0.042 |
| Effective number of BDCs | 14.7 | 19.7 | 21.3 | 23.6 |
| Top-5 share | 45.3% | 38.2% | 37.3% | 35.9% |
| Top-10 share | 59.0% | 51.5% | 51.9% | 50.6% |
| Crowded AUM share, 2+ manager families | 49.3% | 57.1% | 62.0% | 61.2% |
| Crowded AUM share, 2+ BDCs (before 2026-10-23) | 72.4% | 77.8% | 82.9% | 82.6% |

The market became less concentrated as new BDCs entered, and more crowded: the share of AUM in
borrowers with two or more manager families lending rose from 49% to 61-63% by 2025. Counting
BDCs instead of families adds about 20 points: that part was one manager's funds co-investing.

The most widely held borrowers in 2026Q2: Integrity Marketing (21 families, 39 BDCs), MRI
Software (19, 35), Galway Borrower (19, 35), World Insurance Associates (17, 30) and Flexera
Software (17, 32). The median borrower has one family lender; the 90th percentile has three.

(Before 2026-10-22, crowding matched on `investment_identifier` text. It found only 35-46% crowded,
at most 6-10 holders per name, and a false drop in 2025Q4, when most BDCs reworded their
identifiers.)

---

## Limitations

- **Crowding is borrower-level**, as in the Risk Dimensions doc: two families in different
  tranches of one borrower (e.g. a first lien and a second lien) count as crowded.
- Concentration counts each CIK separately; it describes the market, not any BDC's risk.
- The family map (`manager_family.py`) is built from filing names; a BDC whose name shows no
  manager is its own family.
