# Distressed Borrowers: Borrower-Quarter Panel and Case Studies

The borrower-quarter panel behind the warning signs (`12_risk_monitor.md`), and the case
studies. (Until 2026-11-12 this step also defined stress events from `val_ret`; they were
replaced by the non-accrual model in `14_early_warning_model.md`, and the old version of this
doc is in `ORIE 5220/miscellaneous/docs/`.)

## What this step produces

```
python scripts/run_distress_event_study.py      # about 3 minutes
```

Code: `src/8_risk_monitor/distress_event_study.py`. Reads only `FINAL_DATASET.csv`.

- `borrower_quarter_panel()` builds one row per borrower and quarter, with every BDC's loans to
  the borrower combined. `run_risk_monitor.py` also uses it for the warning signs. It is not saved
  to disk.
- `case_study_tables()` writes `outputs/risk_monitor/distress/case_studies/`. For each borrower in
  `data/reference/case_studies.csv` (WebPT, BVI Medical, Medallia, LogRhythm, Premier Imaging,
  Athletico Management, First Brands Group) it writes `<name>_quarterly.csv` (signal path by
  quarter), `<name>_lenders.csv` (each BDC's family, type, par, its mark, the loan's weight in
  its book, PIK rate and non-accrual) and `<name>_marks.png`, plus all cases combined. Add a row
  to the reference file to add a case.

**Borrower-quarter columns:**

| Group | Column(s) |
|---|---|
| Price (mark level) | `price`: sum of FV / sum of PAR over priceable funded loans; above 1.5 set to missing (par left in local currency) |
| Crowding | C1 lender families, C2 lender HHI, C3 largest family's share, C4 change in families over four quarters |
| Lender-side concentration | `max_lender_weight`, `mean_lender_weight`: the borrower's cost as a share of each lending BDC's total cost (largest, and cost-weighted mean). Uses cost, so a markdown does not shrink it mechanically |
| Dispersion | Par-weighted over the borrower's loans held by 2+ manager families: V1 (std and range of family marks), V2 (lowest mark minus consensus), each lender's distance from consensus (`mark_distance`) |
| PIK | `pik_share` (PIK share of the coupon), `pik_par_share` (share of par paying PIK), `toggle_par_share` (share of par with a PIK switch that quarter), `acquired_par_share` (share of par on PIK acquired after origination) |
| Non-accrual | `na_cost_share`: share of the borrower's debt cost flagged non-accrual (filings where the flag is known; 86% of fair value) |
| Size | `par`, `fv`, `cost`, `log_borrower_par` |

## Case studies (2026-10-23 notes)

**WebPT** (healthcare software). This case has the widest lender disagreement in the set. In
2026Q2 Golub marks it 67-68 and has it on non-accrual, while Goldman Sachs, Ares and Monroe mark
it 80-81 and keep it accruing. Dispersion has risen every quarter since late 2024 (std of family
marks 0.02 to 0.11). PIK appeared in 2026Q1, with none before, and is now 67% of the coupon, all
acquired after origination. It has four lender families and seven BDCs; Ares added $79m in
2026Q2. The price slid from 0.98 to 0.84.

**BVI Medical** (healthcare). Lenders crowded in quickly: one lender family in 2024, five (seven
BDCs) by 2026Q2, and BDC exposure up from $21m to $464m. It is 7% of CCS IX Portfolio Holdings'
book, 3.5% of APS BDC's and 2.8% of Diameter's. Marks range from 77 (APS) and 84 (Ares) to 93
(Crescent, CCS) and have widened for four quarters. Every tranche carries a 5% PIK component. It
is not yet on non-accrual; its price fell from 0.99 to 0.86 in two quarters.

**Medallia** (software; the benchmark case). It had seven lender families, was up to 6% of a
lender's book, and showed rising dispersion and growing acquired PIK for four quarters before
its 2026Q1 markdown. Lenders then declined to extend PIK relief, it moved to non-accrual, and
lenders are taking control from Thoma Bravo.

Also on watch: **LogRhythm** (7 families; dispersion jumped in 2025Q4; PIK appeared in 2026Q2;
price 0.87) and **Premier Imaging** (price 0.78).
