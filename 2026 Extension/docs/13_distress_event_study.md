# Distressed Borrowers: Crowding, Concentration, Dispersion and PIK Around a Persistent Markdown

## Question

How do borrowers that suffer a lasting fall in fair value, deep enough to put them in the
bottom 1% of all borrower-quarters (a markdown of about 19% or more), look before and after the fall, on crowding (how many lenders, how concentrated), lender-side concentration (how big the
borrower is in its lenders' books), valuation dispersion across BDCs, and PIK? And can those
signals tell a lasting decline (First Brands) from a one-quarter blip?

```
python scripts/run_distress_event_study.py      # about 5 minutes
```

Code: `src/8_risk_monitor/distress_event_study.py`. Reads only `FINAL_DATASET.csv`. Writes to
`outputs/risk_monitor/distress/`:

| File | Contents |
|---|---|
| `drop_cutoffs.csv` | Per quarter: eligible borrowers, stress drops, the pooled cutoff, that quarter's own 1% return (reference), share marked down, and what a fixed -20% rule would flag |
| `distress_events.csv` | First stress drop per borrower: quarter, drop, its pooled percentile, status, share of the drop regained |
| `case_studies/` | For each borrower in `data/reference/case_studies.csv` (WebPT, BVI Medical, Medallia, LogRhythm, Premier Imaging, Athletico Management, First Brands Group): `<name>_quarterly.csv` (signal path by quarter) and `<name>_lenders.csv` (each BDC: family, type, par, its mark, the loan's weight in its book, PIK rate, non-accrual), plus all cases combined. Add a row to the reference file to add a case |
| `distressed_borrowers_register.csv` | **The record of distressed borrowers** (551): event quarter and drop, status, price / non-accrual / PIK before and after, lender families and number of BDCs before the event, BDC exposure, data center tag, and the outside check with its source where one was done |

Not in the midterm deck and no longer written (removed 2026-11-08; the saved copies are in
`ORIE 5220/miscellaneous/outputs/risk_monitor/distress/`): `borrower_quarter_panel.csv`, the
signal watchlist (`signal_watchlist*.csv`), the event-time profiles (`event_profiles.csv`,
`distress_profiles.png`) and `pre_event_contrast.csv`. Their functions were removed from the repo
on 2026-11-08 (full versions in `ORIE 5220/miscellaneous/code_before_simplify_2026-11-08/`); the sections on them below are a record of the last
run. The Premier Imaging case files are still written, because it is listed in
`data/reference/case_studies.csv`.

## Definitions

**Borrower valuation return.** Fair value can fall because a loan is repaid, a lender exits or
a BDC sells, not only because the loan is worth less. So the return is measured on the same
loans held by the same BDCs in both quarters (tranche chains, `11_tranche_matching.md`), net of
the change in cost, pooled across every BDC holding the borrower:
`sum(FV - FV_prev - (COST - COST_prev)) / sum(FV_prev + max(COST - COST_prev, 0))`. Draws,
paydowns, PIK capitalisation, new loans and exits do not count.

**Stress event (changed 2026-11-01).** Among borrowers with at least $1M of prior fair value, the
borrower's first quarter whose return is at or below the **1st percentile of the pooled
distribution**: all 81,123 eligible borrower-quarters, 2023Q2-2026Q2, ranked together
(`pooled_cutoff()`, `flag_drops()`). The cutoff is **-19.3%**, the same for every quarter. It
replaces the per-quarter bottom 1% used from 2026-10-24, whose cutoff moved from -14% to -25%
with the quarter. A fixed cutoff is simpler to state and does not need a per-quarter
adjustment: the market-wide move is at most -1.2% a quarter (2026Q1, `market_move.png`, from
`scripts/run_tail_concentration.py`),
small next to the cutoff.

**Return denominator (changed 2026-11-01).** The return is taken over the capital at risk, prior
fair value plus any new money (`FV_prev + max(COST - COST_prev, 0)`), not prior fair value
alone. Before, a discount on a large draw into a small loan gave returns below -100% (36
borrower-quarters; BVI Medical 2026Q1 read -251%, now -7%, so it is not an event).

| Quarter | Eligible borrowers | Stress drops (pooled -19.3%) | That quarter's own 1% return | Share marked down | Fixed -20% rule would flag |
|---|---|---|---|---|---|
| 2023Q2-2025Q4 (each) | 4,400-7,500 | 38-76 | -14% to -21% | 38-60% | 37-68 |
| 2026Q1 | 7,748 | **133** | **-25%** | **75%** | **123** |
| 2026Q2 | 7,653 | 69 | -18% | 53% | 68 |

812 borrower-quarters fall at or below the cutoff (804 under the per-quarter rule). In 2026Q1,
when three quarters of borrowers were marked down, the fixed cutoff counts about twice the usual
number of drops; that quarter's software re-mark is part of the event set. Per quarter:
`drop_cutoffs.csv`.

**Persistence.** A drop can also come from a one-off re-mark, one lender's view or a technical
markdown. The event is classified by the next two quarters' cumulative return:

| Status | Rule | Events |
|---|---|---|
| persistent | regains less than half of the drop (the First Brands pattern) | 337 |
| recovered | regains more than half | 76 |
| exited | no BDC holds its loans the next quarter (repaid, sold or written off; unresolved) | 98 |
| too_recent | event in 2026Q2, nothing after it yet | 40 |

551 distressed borrowers in all. The median persistent event loses 27% in the event quarter and
keeps falling (median regained -13%). First Brands: -61% in 2025Q3. First events per quarter are
28 to 53, except 2026Q1 (84).

**What the switch to the pooled cutoff changed.** 551 events against 528 under the per-quarter
bottom 1% (29 out, 52 in). The 2026Q1 software drops of 21-24% come back: **Medallia** (confirmed
distress, lender takeover), Planview and Cornerstone OnDemand (confirmed, sector-driven). Of the
53 borrowers with an outside check, 44 are events (42 before); Alliance Environmental Group (no
reporting found) drops out. The two false positives the check flagged (KWOR; eResearch
Technology) and Banyan Software are not events.

**Signals** (borrower x quarter):

| Group | Signal |
|---|---|
| Price | Par-weighted fair value / par of priced loans |
| Crowding | C1 lender families, C2 lender HHI, C3 largest family's share, C4 change in families over four quarters |
| Lender-side concentration | The borrower's cost as a share of each lending BDC's total cost: largest across lenders, and cost-weighted mean. Cost, so a markdown does not shrink it mechanically |
| Dispersion | Par-weighted over the borrower's loans held by 2+ manager families: V1 (std and range of family marks), V2 (lowest mark minus consensus), each lender's distance from consensus |
| PIK | PIK share of the coupon; share of par paying PIK; share of par with a PIK toggle that quarter; share of par on acquired PIK |
| Non-accrual | Share of the borrower's debt cost flagged non-accrual (known filings) |

**Controls.** Borrowers never at or below the pooled cutoff, matched to each event on quarter,
sector and size (quintile of BDC-held par within the quarter), weighted so every cell counts
as many times as it has events. Event time k = 0 is the drop quarter; the panel is unbalanced,
so each k averages the borrowers observed then.

## Results

Persistent events (337) against matched controls, means:

| Signal | k = -4 | k = -1 | k = 0 | k = +2 | Controls (k = -1) |
|---|---|---|---|---|---|
| Price | 0.90 | 0.81 | 0.57 | 0.48 | 0.98 |
| Share of cost on non-accrual | 10% | 22% | 47% | 62% | 1% |
| PIK share of coupon | 14% | 18% | 19% | 27% | 4% |
| Share of par on acquired PIK | 13% | 13% | 17% | 23% | 4% |
| Lender families (C1) | 1.57 | 1.45 | 1.38 | 1.33 | 1.51 |
| Change in families over 4q (C4) | +0.31 | +0.18 | +0.10 | +0.00 | +0.54 |
| Largest weight in a lender's book | 1.4% | 1.4% | 1.4% | 1.6% | 1.1% |
| V1 dispersion (std of marks)* | 0.022 | 0.031 | 0.052 | 0.052 | 0.009 |
| V2 lowest mark minus consensus* | -0.017 | -0.022 | -0.045 | -0.041 | -0.008 |
| Lender distance from consensus* | 0.016 | 0.023 | 0.036 | 0.039 | 0.006 |

\* Defined only for loans held by 2+ manager families: 50 events at k = -1.

**Before the drop, distressed borrowers already look different from their matched peers:**

- **Price drifts down** for a year (0.90 to 0.81) while peers stay at 0.98. The drop is
  usually the end of a slide, not a surprise.
- **Non-accrual starts early:** 10% of cost a year before, 22% the quarter before, as some of
  the borrower's loans (or some lenders) recognise trouble first.
- **PIK is three to four times the peer level** (14-18% of the coupon vs 4%) and rising, and
  much of it is acquired after origination.
- **Lenders stop arriving.** The four-quarter change in lender families falls from +0.31 to
  +0.18 (peers +0.54), and the lender base becomes more concentrated (HHI up). The largest
  lender's share edges up as others hold back or leave.
- **Lenders disagree more**, where there are several: dispersion is two and a half to three and
  a half times the peer level in the year before and rises into the event; the gap between each
  lender and consensus widens. This is the First Brands pattern in 2025Q2, before its bankruptcy
  marks.
- **Lender-side concentration is a little higher** (largest weight in a lender's book 1.4% vs
  1.1%) and flat; distressed borrowers are not unusually large positions.

**After the drop**, persistent decliners keep losing lenders (C4 near zero), non-accrual
reaches 62% of cost, PIK rises to 27% of the coupon, and lender disagreement stays wide.

**Persistent vs recovered.** Before the drop the two groups are almost indistinguishable on
price, non-accrual, crowding, concentration and dispersion (rank-sum p > 0.2). The exceptions:
recovered borrowers had **more** PIK toggles (8.3% vs 2.8% of par, p = 0.005) and more acquired
PIK (25% vs 13%, p = 0.006) beforehand. They were already being restructured, and the drop is
the mark catching up before the restructured loan recovers. After the drop they regain lenders
and non-accrual falls (47% to 29%), while persistent ones do not.

**Robustness.** Without 2026Q1 events (286 persistent) the profiles are the same (dispersion
0.026 to 0.036 before the event; non-accrual 10% to 24%). On the 65 persistent events with 2+
lender families, the average borrower goes from 3.2 lender families the quarter before to 2.6
two quarters after, and dispersion rises the same way (0.022 to 0.033 before).

## Validation of the 74 multi-family persistent events

*Run on 2026-10-23 on the events of the earlier fixed -20% rule. 59 of the 74 borrowers are
multi-family persistent events under the current pooled -19.3% cutoff, which adds 6 that were
never checked. `event_validation_multi_family.csv` is not regenerated by the script.*

Were these drops caused by the companies' own financial distress? Two checks, saved per event in
`outputs/risk_monitor/distress/event_validation_multi_family.csv` (hand-checked verdicts and
sources in `data/reference/distress_event_validation.csv`):

1. **Internal evidence** (filings): non-accrual within a quarter before to two after the event,
   restructuring terms in the identifier (DIP, exit facility, take-back), a new equity stake in
   the same borrower, how many lender families marked it down 20%+, and the price two quarters on.
2. **External evidence** (news, rating actions, court filings), checked by web search for 51
   names: all of the weaker groups except one, and 18 of the 40 strongest.

| Internal evidence | Events | Confirmed distress | Partial | Not distress | No reporting found | Not checked |
|---|---|---|---|---|---|---|
| A: non-accrual or restructuring, price stays down | 40 | 13 | 0 | 0 | 5 | 22 |
| A?: same, but price back near par (restructured into new debt) | 5 | 2 | 0 | 1 | 2 | 0 |
| B: most lenders marked down 20%+, no flag | 11 | 6 | 1 | 0 | 4 | 0 |
| B?: some lenders marked down 20%+, no flag | 12 | 5 | 1 | 0 | 5 | 1 |
| C: price back at par, no flag | 6 | 1 | 0 | 1 | 4 | 0 |
| All | 74 | 27 | 2 | 2 | 20 | 23 |

- **Where reporting exists, it is distress in 27 of 31 cases:** bankruptcies (First Brands,
  CareerBuilder + Monster, STG Logistics, Multi-Color, Walker Edison, Renovo), distressed
  exchanges or selective defaults (Ivanti, Quest, LaserShip, Ingenovis, Alvaria), and lender
  takeovers (Medallia, Implus, Sandvine, Teasdale, Sonrava). Two are stress without default
  (MoneyGram downgrade; Isagenix, restructured a year earlier).
- **Two are not distress:** eResearch Technology (Clario) and KWOR (Worldwide Clinical Trials),
  both healthy companies. eResearch also has an impossible return (below -100%), so it is a
  refinancing or matching artifact. Drop both; Banyan Software (price back at par, impossible
  return, nothing reported) should go too.
- **Some distress is sector-driven:** Planview, Perforce and Cornerstone OnDemand fell in the
  2026 AI-driven software selloff; Planview and Cornerstone also had refinancing problems.
- **BDC marks often lead public news:** Physician Partners (event 2024Q1, S&P downgrade Oct 2024),
  Ivanti (2024Q2, selective default 2025), Quest (2024Q3, exchanges in 2025), Ingenovis (2024Q4,
  exchange in 2026). Others are marked down at or after the public event (STG, CareerBuilder,
  LaserShip).
- **"No reporting found" is mostly small private borrowers** with no press coverage; most of
  them have internal distress evidence. Absence of news is not evidence of health.
- **Six events have impossible returns** (below -100%: cost rose while value fell). Three are
  confirmed distress (Teasdale, and the restructurings behind 48Forty and Lucky Bucks show
  non-accrual), so the return is distorted by new money or PIK added to cost during a
  restructuring, not necessarily wrong about distress.

**Data center borrowers.** Two of the 551 are tagged data-center related, both with a single
lender family (so neither is among the 74): Internap (core operator; -35% in 2023Q2, exited; its
second Chapter 11 was in May 2023) and Computer Data Source (support; -23% in 2026Q2, too recent;
non-accrual at Main Street). Both are confirmed in the validation file.

**Keeping the record current.** The register is rebuilt by `run_distress_event_study.py`. New
outside checks go in `data/reference/distress_event_validation.csv` (borrower, verdict, note,
source) and are joined on the next run.

For case studies, the 27 confirmed events (plus the 22 unchecked group-A events, whose internal
evidence matches the confirmed ones) are the defensible set.

## Watchlist: all four signals at once

*Record of the 2026-11-01 run; no longer written (see the outputs table).*

Which borrowers show crowding, lender-side concentration, valuation dispersion and a rising PIK
trend at the same time? Among borrowers with 2+ lender families (about 2,200 a quarter), each
signal is ranked within the quarter; a borrower-quarter is flagged when all four are in the top
30% and PIK is rising (`signal_watchlist()`). 46 borrowers were flagged in some quarter (48 before the
2026-10-24 manager-family merges; WebPT, BVI Medical, Medallia, LogRhythm and Premier Imaging
are all still on it).

| File | Contents |
|---|---|
| `signal_watchlist.csv` | One row per flagged borrower: flagged quarters, latest price / lenders / weight / dispersion / PIK / non-accrual, and any distress event from the register |
| `signal_watchlist_quarters.csv` | Every flagged borrower-quarter with the four percentiles |

These are signals, not predictions; several of the names are software and healthcare credits in
2026, when the sector itself was repriced.

### Cases picked for a closer look (2026-10-23)

**WebPT** (healthcare software; flagged 2026Q1-Q2). The widest lender disagreement in the set:
in 2026Q2 Golub marks it 67-68 and has it on non-accrual, while Goldman Sachs, Ares and Monroe
mark it 80-81 and keep it accruing. Dispersion has risen every quarter since late 2024 (std of
family marks 0.02 to 0.11). PIK appeared in 2026Q1 (none before) and is now 67% of the coupon,
all acquired after origination. Four lender families, seven BDCs; Ares added $79m in 2026Q2.
Concentration is modest (1.9% of a Goldman fund's book). The price slid from 0.98 to 0.84 without
a quarter at or below the -19.3% cutoff, so it is not in the distress register. Next: the BDCs' 10-Q footnotes on why only
Golub moved it to non-accrual, and news on any amendment.

**BVI Medical** (healthcare; flagged 2025Q4-2026Q2). Crowding and concentration built up fast:
from one lender family in 2024 to five (seven BDCs) by 2026Q2, BDC exposure from $21m to $464m.
It is 7% of CCS IX Portfolio Holdings' book (16% in 2025Q4, 32% in 2025Q1), 3.5% of APS BDC's and
2.8% of Diameter's, all private BDCs. Marks range from 77 (APS) and 84 (Ares) to 93 (Crescent,
CCS), widening for four quarters. Every tranche carries a 5% PIK component (about half the
coupon). No non-accrual yet; price 0.99 to 0.86 in two quarters.

**Medallia** (software; flagged 2024Q4-2025Q3; the benchmark case). All four signals were high
for four quarters before its 2026Q1 markdown (-21%, a persistent stress event in the
register): seven lender families, up to 6% of a lender's book,
rising dispersion, growing acquired PIK. Lenders then declined to extend PIK relief, it moved to
non-accrual, and lenders are taking control from Thoma Bravo (confirmed in
`data/reference/distress_event_validation.csv`). The signals led the event by about a year.

Also on watch: **LogRhythm** (7 families; dispersion jumped in 2025Q4; PIK appeared 2026Q2; price
0.87) and **Premier Imaging** (flagged in 7 quarters; price 0.78).

## What this does and does not show

- **Descriptive, not predictive.** The signals are compared at fixed event times; whether they
  predict who will fall is the evaluation layer's job (`12_risk_monitor.md`), where PIK share
  and a falling lender count add to the price baseline and hold out of sample.
- **Persistence cannot be predicted from these signals** (persistent and recovered look alike
  beforehand), except that prior PIK amendments point to a restructuring that later recovers.
- **Most events have a single lender family** (82% of all 551; 81% of the 337 persistent ones),
  and 88% have no loan marked by two lender families, so there is no second value to compare.
  Dispersion and the crowding measures rest on the 65 multi-family persistent events (Computer
  Data Source is a single-family case).

## Limitations

- **Exited events (98) are unresolved:** the borrower left every BDC's book within a quarter of
  the drop, through repayment, a sale, or a write-off. Rung 4 (exit at a loss) would classify them.
- **Pre-event window is short for 2023 events** (the panel starts in 2023Q1); 82% of events have
  a full year before them.
- **Unbalanced panel:** each k averages whoever is observed; borrowers that leave drop out.
- **Matching** is on quarter, sector and size only; BDC type and vintage are not matched.
- Non-accrual is known for 86% of fair value (private BDCs 70%).
- **A fixed cutoff counts sector re-marks as events.** 2026Q1 has 133 drops (84 first events)
  against 38-76 in other quarters, much of it the software re-mark; three of those are checked
  as sector-driven (Planview, Perforce, Cornerstone OnDemand). Exclude 2026Q1 (see Robustness)
  when a re-mark-free set is needed.
