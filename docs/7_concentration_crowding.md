# Crowding: How Many Lenders Share a Borrower

`src/3_concentration_crowding/concentration_crowding.py` has one function,
`compute_borrower_crowding_metrics(df)`. It is called by the risk monitor (`12_risk_monitor.md`)
and the model panel, and has no script of its own.

Holders are **manager families**, not individual BDCs: funds of one manager share one valuation
and usually co-invest, so they count once (`manager_family.py`, see `11_tranche_matching.md`).
Lenders are matched on the validated borrower name (`borrower_canonical`), never on
`investment_identifier` text, which each BDC words differently. Only funded debt counts (par > 0,
fair value > 0). One row per quarter and borrower:

| Code | Column | Definition |
|---|---|---|
| C1 | `n_families` | Manager families holding the borrower (`n_ciks`: BDCs) |
| C2 | `lender_hhi` | Sum of squared family shares of BDC-held par (1 = one family holds it all) |
| C3 | `top_share` | Largest family's share of par |
| C4 | `d_families_4q` | C1 now minus C1 four quarters earlier (absent then = 0; missing in 2023) |
| | `log_par` | Log of par held by all BDCs, the size control |
| | `C1_pct` ... `C4_pct` | Within-quarter percentiles |

**On the current data.** The median borrower has one lender family and the 90th percentile three.
The most widely held borrowers in 2026Q2 are Integrity Marketing (21 families, 39 BDCs), MRI
Software (19, 35), Galway Borrower (19, 35), World Insurance Associates (17, 30) and Flexera
Software (17, 32). Big borrowers have more lenders (rank correlation of BDC-held par and families
about 0.5), so the risk monitor ranks crowding within size fifths.

**Limitations.**

- Crowding is borrower-level: two families in different tranches of one borrower (a first lien
  and a second lien) count as sharing it.
- The family map is built from filing names; a BDC whose name shows no manager is its own family.

The market-concentration measures (HHI, effective number of BDCs) and their charts were retired on
2026-11-08; the old version of this doc is in `ORIE 5220/miscellaneous/`.
