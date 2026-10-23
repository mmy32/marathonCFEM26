# Distressed Borrowers: Crowding, Concentration, Dispersion and PIK Around a Persistent Markdown

## Question

How do borrowers that suffer a lasting 20%+ fall in fair value look, before and after the
fall, on crowding (how many lenders, how concentrated), lender-side concentration (how big the
borrower is in its lenders' books), valuation dispersion across BDCs, and PIK? And can those
signals tell a lasting decline (First Brands) from a one-quarter blip?

```
python scripts/run_distress_event_study.py      # about 5 minutes
```

Code: `src/8_risk_monitor/distress_event_study.py`. Reads only `FINAL_DATASET.csv`. Writes to
`outputs/risk_monitor/distress/`:

| File | Contents |
|---|---|
| `borrower_quarter_panel.csv` | One row per borrower x quarter (all BDCs combined) with every signal |
| `distress_events.csv` | First 20%+ drop per borrower: quarter, drop, status, share of the drop regained |
| `signal_watchlist.csv`, `signal_watchlist_quarters.csv` | **Watchlist**: borrowers with all four signals high at once (51); see "Watchlist" below |
| `case_studies/` | For each borrower in `data/reference/case_studies.csv` (WebPT, BVI Medical, Medallia, LogRhythm, Premier Imaging): `<name>_quarterly.csv` (signal path by quarter) and `<name>_lenders.csv` (each BDC: family, type, par, its mark, the loan's weight in its book, PIK rate, non-accrual), plus all cases combined. Add a row to the reference file to add a case |
| `distressed_borrowers_register.csv` | **The record of distressed borrowers** (593): event quarter and drop, status, price / non-accrual / PIK before and after, lender families and number of BDCs before the event, BDC exposure, data center tag, and the outside check with its source where one was done |
| `event_profiles.csv` | Mean of each signal at k = -4..+2 for four event groups and matched controls, with n |
| `pre_event_contrast.csv` | Persistent vs recovered events before the drop (k = -2, -1), rank-sum p-values |
| `distress_profiles.png` | The profiles, one panel per signal |

## Definitions

**Borrower valuation return.** Fair value can fall because a loan is repaid, a lender exits or
a BDC sells, not only because the loan is worth less. So the return is measured on the same
loans held by the same BDCs in both quarters (tranche chains, `11_tranche_matching.md`), net of
the change in cost, pooled across every BDC holding the borrower:
`sum(FV - FV_prev - (COST - COST_prev)) / sum(FV_prev)`. Draws, paydowns, PIK capitalisation,
new loans and exits do not count.

**Event.** The borrower's first quarter with a return of -20% or worse, on at least $1M of
prior fair value.

**Persistence.** A drop can also come from a one-off re-mark, one lender's view or a technical
markdown. The event is classified by the next two quarters' cumulative return:

| Status | Rule | Events |
|---|---|---|
| persistent | regains less than half of the drop (the First Brands pattern) | 362 |
| recovered | regains more than half | 79 |
| exited | no BDC holds its loans the next quarter (repaid, sold or written off; unresolved) | 100 |
| too_recent | event in 2026Q2, nothing after it yet | 52 |

The median persistent event loses 29% in the event quarter and keeps falling (median regained
-11%). First Brands: -64% in 2025Q3, 0.2% regained. Events by quarter are 20 to 40 persistent
ones a quarter, except 2026Q1 (60), the market-wide re-mark quarter; results are shown with and
without it.

**Signals** (borrower x quarter):

| Group | Signal |
|---|---|
| Price | Par-weighted fair value / par of priced loans |
| Crowding | C1 lender families, C2 lender HHI, C3 largest family's share, C4 change in families over four quarters |
| Lender-side concentration | The borrower's cost as a share of each lending BDC's total cost: largest across lenders, and cost-weighted mean. Cost, so a markdown does not shrink it mechanically |
| Dispersion | Par-weighted over the borrower's loans held by 2+ manager families: V1 (std and range of family marks), V2 (lowest mark minus consensus), each lender's distance from consensus |
| PIK | PIK share of the coupon; share of par paying PIK; share of par with a PIK toggle that quarter; share of par on acquired PIK |
| Non-accrual | Share of the borrower's debt cost flagged non-accrual (known filings) |

**Controls.** Borrowers that never have a 20% quarterly drop, matched to each event on quarter,
sector and size (quintile of BDC-held par within the quarter), weighted so every cell counts
as many times as it has events. Event time k = 0 is the drop quarter; the panel is unbalanced,
so each k averages the borrowers observed then.

## Results

Persistent events (362) against matched controls, means:

| Signal | k = -4 | k = -1 | k = 0 | k = +2 | Controls (k = -1) |
|---|---|---|---|---|---|
| Price | 0.91 | 0.82 | 0.60 | 0.51 | 0.98 |
| Share of cost on non-accrual | 9% | 20% | 44% | 58% | 1% |
| PIK share of coupon | 13% | 17% | 19% | 27% | 4% |
| Share of par on acquired PIK | 13% | 13% | 18% | 24% | 4% |
| Lender families (C1) | 1.57 | 1.45 | 1.39 | 1.37 | 1.56 |
| Change in families over 4q (C4) | +0.33 | +0.17 | +0.09 | +0.01 | +0.57 |
| Largest weight in a lender's book | 1.4% | 1.4% | 1.4% | 1.6% | 1.1% |
| V1 dispersion (std of marks)* | 0.021 | 0.032 | 0.050 | 0.062 | 0.009 |
| V2 lowest mark minus consensus* | -0.017 | -0.022 | -0.043 | -0.058 | -0.007 |
| Lender distance from consensus* | 0.014 | 0.023 | 0.035 | 0.048 | 0.006 |

\* Defined only for loans held by 2+ manager families: 56 events at k = -1, 2,625 controls.

**Before the drop, distressed borrowers already look different from their matched peers:**

- **Price drifts down** for a year (0.91 to 0.82) while peers stay at 0.98. The drop is
  usually the end of a slide, not a surprise.
- **Non-accrual starts early:** 9% of cost a year before, 20% the quarter before, as some of
  the borrower's loans (or some lenders) recognise trouble first.
- **PIK is three to four times the peer level** (13-17% of the coupon vs 4%) and rising, and
  much of it is acquired after origination.
- **Lenders stop arriving.** The four-quarter change in lender families falls from +0.33 to
  +0.17 (peers +0.57), and the lender base becomes more concentrated (HHI up). The largest
  lender's share edges up as others hold back or leave.
- **Lenders disagree more**, where there are several: dispersion is two to three times the peer
  level a year before and rises into the event; the gap between each lender and consensus
  widens. This is the First Brands pattern in 2025Q2, before its bankruptcy marks.
- **Lender-side concentration is a little higher** (largest weight in a lender's book 1.4% vs
  1.1%) and flat; distressed borrowers are not unusually large positions.

**After the drop**, persistent decliners keep losing lenders (C4 near zero), non-accrual
reaches 58% of cost, PIK rises to 27% of the coupon, and lender disagreement keeps widening.

**Persistent vs recovered.** Before the drop the two groups are almost indistinguishable on
price, non-accrual, crowding, concentration and dispersion (rank-sum p > 0.1). The exceptions:
recovered borrowers had **more** PIK toggles (8.5% vs 3.6% of par, p = 0.01) and more acquired
PIK (25% vs 13%, p = 0.004) beforehand. They were already being restructured, and the drop is
the mark catching up before the restructured loan recovers. After the drop they regain lenders
and non-accrual falls (48% to 28%), while persistent ones do not.

**Robustness.** Without 2026Q1 events (302) the profiles are the same, slightly stronger
(dispersion 0.025 to 0.038 before the event; non-accrual 10% to 21%). On the 74 persistent
events with 2+ lender families, the average borrower goes from 3.0 lender families the quarter
before to 2.6 two quarters after, and dispersion rises the same way (0.019 to 0.033 before).

## Validation of the 74 multi-family persistent events

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

**Data center borrowers.** Two of the 593 are tagged data-center related, both with a single
lender family (so neither is among the 74): Internap (core operator; -35% in 2023Q2, exited; its
second Chapter 11 was in May 2023) and Computer Data Source (support; -23% in 2026Q2, too recent;
non-accrual at Main Street). Both are confirmed in the validation file.

**Keeping the record current.** The register is rebuilt by `run_distress_event_study.py`. New
outside checks go in `data/reference/distress_event_validation.csv` (borrower, verdict, note,
source) and are joined on the next run.

For case studies, the 27 confirmed events (plus the 22 unchecked group-A events, whose internal
evidence matches the confirmed ones) are the defensible set.

## Watchlist: all four signals at once

Which borrowers show crowding, lender-side concentration, valuation dispersion and a rising PIK
trend at the same time? Among borrowers with 2+ lender families (about 2,200 a quarter), each
signal is ranked within the quarter; a borrower-quarter is flagged when all four are in the top
30% and PIK is rising (`signal_watchlist()`). 51 borrowers were flagged in some quarter.

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
a 20% quarter, so it is not in the distress register. Next: the BDCs' 10-Q footnotes on why only
Golub moved it to non-accrual, and news on any amendment.

**BVI Medical** (healthcare; flagged 2025Q4-2026Q2). Crowding and concentration built up fast:
from one lender family in 2024 to five (seven BDCs) by 2026Q2, BDC exposure from $21m to $464m.
It is 7% of CCS IX Portfolio Holdings' book (16% in 2025Q4, 32% in 2025Q1), 3.5% of APS BDC's and
2.8% of Diameter's, all private BDCs. Marks range from 77 (APS) and 84 (Ares) to 93 (Crescent,
CCS), widening for four quarters. Every tranche carries a 5% PIK component (about half the
coupon). No non-accrual yet; price 0.99 to 0.86 in two quarters.

**Medallia** (software; flagged 2024Q4-2025Q3; the benchmark case). All four signals were high
for four quarters before its 2026Q1 markdown: seven lender families, up to 6% of a lender's book,
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
- **Two-thirds of events have a single lender family**, so dispersion and the crowding
  measures rest on the 74 multi-family events (Computer Data Source is a single-family case).

## Limitations

- **Exited events (100) are unresolved:** the borrower left every BDC's book within a quarter of
  the drop, through repayment, a sale, or a write-off. Rung 4 (exit at a loss) would classify them.
- **Pre-event window is short for 2023 events** (the panel starts in 2023Q1); 78% of events have
  a full year before them.
- **Unbalanced panel:** each k averages whoever is observed; borrowers that leave drop out.
- **Matching** is on quarter, sector and size only; BDC type and vintage are not matched.
- Non-accrual is known for 86% of fair value (private BDCs 70%).
