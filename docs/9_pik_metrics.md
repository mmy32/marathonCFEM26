# PIK Risk Metrics

## Overview

PIK (payment-in-kind) interest is paid by adding to the loan's principal instead of in cash.
PIK written into a deal at origination is a structuring choice. PIK switched on later is often
a sign of borrower stress, because the lender accepts a promise to pay later instead of cash
now.

`src/5_pik/pik_metrics.py` separates these cases and tracks them over time.
`scripts/run_pik_risk_metrics.py` runs the whole set:

```
python scripts/run_pik_risk_metrics.py
python tests/test_pik_metrics.py        # unit tests
```

It reads `data/final_dataset/FINAL_DATASET.csv` and prepares it in five steps:

1. `load_and_prepare_investment_data()`
2. `prepare_subindex_input()`: merges near-duplicate sector labels and applies the rate ceiling
   (see `5_subindex_construction.md`).
3. `compute_final_interest_rates()`: gives `PIC_Final`, `PIK_Final` and `IR_Final`.
4. `tranches.build_tranche_panel()`: funded debt tranches matched within each borrower on lien,
   spread and maturity, linked over time into chains (`tranche_id`), matched across BDCs, and
   P3 (PIK at origination). See `11_tranche_matching.md`. Equity and undrawn commitments drop
   out here (no PIK).
5. `compute_position_level_flows(group_cols=["cik", "tranche_id"])` and
   `compute_market_weights()`: each tranche's prior quarter, income and market weight.

Provenance (section 1) runs on the rows before step 4.

Outputs:

| File | Contents |
|---|---|
| `outputs/pik/pik_provenance_summary.csv` | Per quarter: positions, FV and par by provenance (explicit / inferred / zero PIK) |
| `outputs/pik/pik_provenance_by_cik.csv` | Per BDC: how often its PIK is explicit vs. inferred |
| `outputs/pik/pik_tranche_signals.csv` | One row per tranche-quarter: P1 (`pik_share`, `P1_pct`), P2 (`is_distress_amendment`), P3 (`pik_at_origination`, `origination_censored`), `pik_acquired`, chain `link_rule` and cross-BDC `xb_id` / `xb_tier` |
| `outputs/pik/pik_transitions_summary.csv` | Per quarter: counts and FV by transition state, amendment rates |
| `outputs/pik/pik_non_cash_metrics.csv` | Per quarter: the three non-cash shares, weighted and unweighted |
| `outputs/pik/pik_sector_time_series.csv` | The same metrics per sector and quarter |
| `outputs/pik/pik_amendment_event_study.csv` | Markdown rates around a distress amendment, amended vs. never-amended loans |
| `outputs/pik_plots/*.png` | Eight charts of the above |

---

## 1. Provenance: was the PIK rate in the filing?

**`classify_pik_provenance(df)`** labels each position by where its PIK rate came from. It uses
`check_2`, which lists the rate fields the filing tagged, e.g. `spread, rate, pik`.

| Label | Rule |
|---|---|
| `explicit_pik` | PIK > 0, and the filing tagged both the cash and PIK rates (`check_2` is `rate, pik, pic` or `spread, rate, pik, pic`) |
| `inferred_pik` | PIK > 0 otherwise: PIK tagged but cash rate inferred, or PIK derived by the cleaning step |
| `zero_pik` | No PIK |

Without a `check_2` column the function falls back to "any positive tagged PIK is explicit".
`FINAL_DATASET.csv` has carried `check_2` since 2026-10-22, so the fallback is not used.

Under this definition explicit PIK is rare: 136 positions in 2026Q2, against 4,734 inferred.
Most loans with a tagged PIK rate count as inferred because their cash rate was not tagged.
If "explicit" should mean "the PIK rate itself was tagged", change the two `check_2` lists in
`classify_pik_provenance`.

## 2. Transitions: contractual vs. distress PIK

**`classify_loan_pik_transitions(df)`** follows each loan's chain (`cik`, `tranche_id`) from
quarter to quarter, so a toggle is on the same loan, not on a borrower's other tranche (Risk
Dimensions doc: "without a change of tranche"). PIK under 25 bp counts as none (rounding in
total minus cash rate). The PIK share of the coupon is `pik_share` = PIK / total rate (P1).

| State | Rule |
|---|---|
| `contractual_pik` | PIK in the chain's first quarter, and the loan had PIK when it first appeared in any BDC (P3), or that is unknown |
| `acquired_pik` | PIK in the chain's first quarter, but the loan was cash-pay when it first appeared in another BDC (P3 = False) |
| `distress_amendment_pik` | P2, the toggle: a seasoned loan whose PIK appears, or whose PIK share rises by at least 10 points |
| `seasoned_pik_continuation` | Ongoing PIK, no toggle |
| `cured_pik` | PIK goes back to none |
| `stable_cash` | No PIK |

Also added: `P1_pct` (within-quarter percentile of `pik_share`) and `pik_acquired` (PIK now,
and either P3 = False or the chain toggled earlier), which is the "acquired PIK" the Risk
Dimensions doc wants P1 tested on.

The 10-point share threshold replaces the earlier 150 bp rate jump: the doc defines P2 on the
share, and smaller moves are mostly rounding (e.g. 94% -> 100% when the inferred PIK rate
catches up with the total).

**`aggregate_pik_transitions()`** gives quarterly counts and FV, and the amendment rates
`pct_amendment_count` (share of seasoned tranches) and `pct_amendment_fv` (share of FV).

On the current data, 1.3% to 2.1% of seasoned tranches turn to PIK each quarter (152 to 397
tranches), with no break in 2025Q4. Over the panel: 3,661 toggles, 3,731 PIK-at-start chains
and 172 acquired at entry. Of 51,231 chains, P3 is False for 30,511, True for 2,714, and
censored (first seen in 2023Q1) for 18,006.

## 3. Non-cash shares

**`compute_non_cash_metrics()`** gives three position-level measures:

| Metric | Formula | Measures |
|---|---|---|
| A. Rate share | `PIK_Final / IR_Final` | Contracted non-cash part of the coupon |
| B. Income share | `pik_income / (cash_income + pik_income)` | Non-cash part of interest accrued in the quarter |
| C. Capitalized burden | `(PAR_t - PAR_incept) / PAR_t` | Principal growth since the loan was first seen |

**`aggregate_non_cash_metrics()`** gives market-value-weighted and simple means per quarter.
The weighted income share stays between 4.8% and 6.1% from 2023Q2 to 2026Q2 (5.2% in 2026Q2),
on funded debt tranches. 2023Q1 has no prior quarter, so its shares are zero. The rise from
1.3% reported before 2026-10-22 came from identifier matching, which missed most continuing
positions early on.

Metric C is kept for reference only: the Risk Dimensions doc drops it, because par also moves
with draws, repayments and upsizing.

## 4. Sector sweep

**`compute_sector_time_series_pik_metrics()`** reruns steps 1 to 3 within each sector.
It feeds `pik_sector_time_series.csv` and the sector amendment chart.

## 5. Event study: do amendments predict markdowns?

**`compute_amendment_event_study(window=(-2, 1), pct=0.10)`** takes loans at their first
distress amendment (quarter t) and compares them with loans that never amend.

- **Comparison loans:** from the same sector and event quarter, reweighted to the amended loans'
  sector-by-quarter mix.
- **Balanced panel:** both groups must be observed in every quarter of the window.
- **Outcomes (peer-relative since 2026-10-24):** the share of loans whose price (FV / par) is in
  the bottom decile of that quarter's cross-section of all loans, and the share with a
  peer-relative markdown: valuation return `(FV - FV_prev - (COST - COST_prev)) / FV_prev`
  negative and in the quarter's bottom decile, the risk monitor's outcome rung 1
  (`add_peer_relative_flags()`). Both are 10% across all loans by construction.

These replace the fixed outcomes used before (FV/Cost below 0.80, and a one-quarter FV drop of
more than 20%), for the reason the midterm outline gives: a fixed threshold misfires when the
whole market re-marks, as in 2026Q1, while ranks within the quarter do not move. The old drop
measure also used raw FV change, so a paydown counted as a markdown.

On the current data (tranche chains) there are 1,840 amended loans and 126,751 comparison loans:

| Quarter relative to amendment | Amended: price in bottom decile | Never amended | Amended: markdown in bottom decile | Never amended |
|---|---|---|---|---|
| t-2 | 26.8% | 7.5% | 26.6% | 7.3% |
| t-1 | 33.9% | 8.0% | 28.3% | 8.0% |
| t | 35.9% | 8.6% | 26.7% | 8.5% |
| t+1 | 38.6% | 9.3% | 27.1% | 8.9% |

- **Amended loans are already in the low-price tail two quarters before the amendment** (27% vs
  7.5%), and keep sinking through it (39% at t+1).
- **They are marked down in the bottom decile about three times as often as comparison loans**,
  in every quarter of the window (27-28% vs 7-9%). Under the old fixed -20% rule the gap
  appeared only around t (10.1% vs 4.5%). The peer-relative cutoff is mild (about -1.5% to -2%
  in calm quarters, -4.4% in 2026Q1), so it picks up the repeated small markdowns that precede
  and follow an amendment, not just one large drop.

This is descriptive; the risk monitor's evaluation layer tests P2 against later markdowns and
non-accrual over the price baseline.

**Amendments against markdowns over time** (`amendment_markdown_series()`,
`pik_amendments_vs_markdowns.csv`, `distress_amendments_vs_markdowns.png`). Each quarter, the
share of loans amended at t or t-1 that have a peer-relative markdown, against all loans (10% by
construction). It is 23-30% from 2023Q3 on and highest in 2026Q2 (32%). The chart has two
panels on one time axis; the earlier version put amendment volume and the FV/Cost < 0.80 /
0.90 shares on two y-axes (the "lead-lag" chart the Risk Dimensions doc rejects).

---

## Charts (`outputs/pik_plots/`)

| File | Shows |
|---|---|
| `pik_provenance_positions.png` | Share of positions by provenance; explicit vs. inferred among PIK positions |
| `pik_provenance_dollar_volume.png` | FV and par by provenance ($bn) |
| `bdc_idiosyncratic_tagging.png` | BDCs with the most inferred PIK, by fund name (`CIK_NAME_MAP`) |
| `distress_amendments_vs_markdowns.png` | Amendment volume ($bn), and the share of recently amended loans vs all loans with a peer-relative markdown |
| `sector_distress_amendments.png` | Amendment rate by sector |
| `pik_distress_vs_contractual_share.png` | PIK payments by state, and the distress share of PIK |
| `pik_non_cash_trends.png` | The three non-cash shares over time |
| `amendment_event_study.png` | The event study above (peer-relative outcomes), with ±1.96 SE bands for amended loans |

---

## Limitations

- **Chain links vary in strength.** P2 fires on about 9% of rule-2 and rule-4 links (repricing,
  or an amendment that moved both terms) against about 1% of rule-1 and rule-3 links. That fits
  PIK coming with amendments, but rule-4 links are also where a wrong link would sit; check
  results on a rule 1-3 subsample (`link_rule` in `pik_tranche_signals.csv`).
- **P3 is censored** for loans first seen in 2023Q1 (35% of chains), and looks one hop back.
- **The explicit-only sample** (Risk Dimensions doc, data caveat) uses this repo's provenance
  rule, under which a PIK rate tagged without a cash rate counts as inferred (e.g. Computer
  Data Source). Decide which definition the explicit sample should use before testing.
- **The first quarter (2023Q1) has no seasoned tranches** by construction, so its amendment rate
  is empty.
- Outputs before the tranche switch (borrower + seniority positions, 150 bp rule) are in
  `../just in case/repo_cleanup_2026-10-24/_archive/outputs_before_2026-10-23/pik/`.
