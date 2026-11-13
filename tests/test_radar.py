"""
Private Credit Radar check: rebuilds the six indicators of Private Credit Radar version 1.0
(S. Steffen, 2 November 2026; filings to 30 June 2026) on FINAL_DATASET.csv and prints them beside
the published figures.

    python tests/test_radar.py      # about 2 minutes; without FINAL_DATASET.csv only the definition tests run

The Radar cleans the filings differently, so the figures are not expected to be equal. The data
test asserts only that each figure is within a factor of LEVEL_BAND of the Radar's (a wrong unit or
denominator would fail) and that the Radar's findings hold in our data. The definition tests run on
a hand-made panel.

A position is one row of tranches.build_tranche_panel: a BDC's funded loan to one borrower (same
lien, maturity and spread), CLO tranches left out. Its mark is fair value / par x 100, on the rows
our cleaning prices (cost at least 80% of par, so a partly drawn line is not read as a low mark).
A position is followed when its chain has a row in the quarter before.

    1  below90, below80  marked below 90 / 80, % of priced positions
    2  down5             marked down 5 points or more in the quarter, % of followed priced positions
    3  mark_gap          borrowers whose managers' marks are 5 points or more apart, % of borrowers
                         held by 2+ manager families (each family's mark: its fair value / par in
                         the borrower); mark_gap_same_loan compares marks loan by loan instead
    4  pik_switch        no PIK the quarter before, PIK now, % of followed positions
    5  tier2, tier3      PIK switch, spread up 50 bp or more, or maturity extended when it was within
                         a year, % of followed positions; tier 2 if marked below 90 or down 5 points
                         that quarter, tier 3 if not. "Within a year" is our reading of the Radar's
                         "shortly before maturity"
    6  non_accrual       cost flagged non-accrual, % of all debt cost (unknown status counts as
                         accruing, as in the Radar)

Groups: all BDCs; constant (a position in every quarter); listed; non_listed (the Radar's
"non-traded": every BDC that is not listed).

Indicator 5 is the least reliable here. Maturity and spread are also what link a loan from one
quarter to the next, so a changed term is seen only through the looser chain rules (2 and 4 in
docs/11_tranche_matching.md), where a refinancing into a new loan can look like an amendment.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "1_data_processing"))
from paths import add_code_dirs_to_path, FINAL_DIR
add_code_dirs_to_path()

import tranches as T
import val_dispersion as VD
import monitor_panel as MP

FINAL_CSV = FINAL_DIR / "FINAL_DATASET.csv"

DOWN_POINTS = 5              # indicator 2
GAP_POINTS = 5               # indicator 3
SPREAD_STEP = 0.005          # indicator 5: 50 bp
NEAR_MATURITY_DAYS = 365     # indicator 5: an extension counts when the old maturity was this close
MAX_MARK = VD.MAX_PRICE * 100
LEVEL_BAND = 2.0             # ours must lie between Radar / 2 and Radar x 2

# Published figures, percent. Chart pack p.11 and presentation p.3; "-" there is None here.
RADAR = pd.DataFrame(
    [("below90", 7.71, 10.33, 10.26, 11.5, 12.8, 9.3),
     ("below80", 4.35, 6.06, 5.91, 6.7, 7.7, 5.2),
     ("down5", 3.13, 6.57, 4.01, 4.6, 5.1, 3.6),
     ("mark_gap", 9.30, 11.64, 12.29, None, None, None),
     ("pik_switch", 1.15, 1.52, 1.44, 1.63, 1.67, 1.35),
     ("tier2", 0.40, 0.46, 0.53, 0.62, 0.71, 0.46),
     ("tier3", 2.11, 2.17, 2.74, 2.76, 3.51, 2.41),
     ("non_accrual", 1.29, 1.95, 1.84, 2.12, 3.75, 1.13)],
    columns=["indicator", "2025Q2", "2026Q1", "2026Q2", "constant", "listed", "non_listed"]).set_index("indicator")
SIX = ["below90", "down5", "mark_gap", "pik_switch", "tier2", "non_accrual"]


def positions(tr: pd.DataFrame) -> pd.DataFrame:
    """The tranche panel without CLO tranches, with each position's mark and its flags."""
    p = tr[tr["lien_group"] != "CLO"].sort_values(["chain_id", "qsort"]).reset_index(drop=True)
    p["mark"] = (p["price"] * 100).where(p["price"] * 100 <= MAX_MARK)
    prev = p.groupby("chain_id")[["mark", "qsort", "PIK_Final", "spread", "maturity"]].shift(1)
    p["followed"] = p["qsort"] - prev["qsort"] == 1
    p["mark_followed"] = p["followed"] & p["mark"].notna() & prev["mark"].notna()
    p["below90"] = p["mark"] < 90
    p["below80"] = p["mark"] < 80
    p["down5"] = p["mark_followed"] & (p["mark"] - prev["mark"] <= -DOWN_POINTS + 1e-9)

    def pays_pik(rate):
        return rate.fillna(0) >= T.MIN_PIK_RATE
    p["pik_switch"] = p["followed"] & pays_pik(p["PIK_Final"]) & ~pays_pik(prev["PIK_Final"])
    spread_up = p["spread"] - prev["spread"] >= SPREAD_STEP - 1e-9
    prev_quarter_end = pd.Series((pd.PeriodIndex(p["cal_q"], freq="Q") - 1).end_time.normalize(), index=p.index)
    extended = (((p["maturity"] - prev["maturity"]).dt.days > T.MATURITY_TOL_DAYS)
                & ((prev["maturity"] - prev_quarter_end).dt.days <= NEAR_MATURITY_DAYS))
    amended = p["followed"] & (p["pik_switch"] | spread_up | extended)
    stressed = p["below90"] | p["down5"]
    p["tier2"] = amended & stressed
    p["tier3"] = amended & ~stressed
    return p


def indicators(p: pd.DataFrame) -> pd.DataFrame:
    """Indicators 1, 2, 4, 5 and 6 per quarter, percent, for the positions given."""
    def share(rows, flag):
        return rows.groupby("cal_q")[flag].mean() * 100
    priced, marked, followed = p[p["mark"].notna()], p[p["mark_followed"]], p[p["followed"]]
    on_non_accrual = p["COST"].where(p["non_accrual"] == 1, 0.0).groupby(p["cal_q"]).sum()
    return pd.DataFrame({
        "below90": share(priced, "below90"), "below80": share(priced, "below80"),
        "down5": share(marked, "down5"),
        "pik_switch": share(followed, "pik_switch"),
        "tier2": share(followed, "tier2"), "tier3": share(followed, "tier3"),
        "non_accrual": on_non_accrual / p.groupby("cal_q")["COST"].sum() * 100,
    })


def mark_gap(tr: pd.DataFrame) -> pd.Series:
    """Indicator 3 per quarter, percent of the borrowers held by two or more manager families."""
    t = tr[(tr["lien_group"] != "CLO") & (tr["PAR_px"] > 0)]
    fam = t.groupby(["cal_q", "borrower_key", "family"])[["FV_px", "PAR_px"]].sum()
    fam["mark"] = fam["FV_px"] / fam["PAR_px"] * 100
    b = fam[fam["mark"] <= MAX_MARK].groupby(["cal_q", "borrower_key"])["mark"].agg(["min", "max", "size"])
    b = b[b["size"] >= 2]
    return (b["max"] - b["min"] >= GAP_POINTS - 1e-9).groupby("cal_q").mean() * 100


def mark_gap_same_loan(tr: pd.DataFrame) -> pd.Series:
    """Indicator 3 as our monitor measures disagreement: on loans matched across BDCs. A borrower
    counts when any of its loans held by two or more families has marks 5 points or more apart."""
    disp = VD.tranche_dispersion(tr[tr["lien_group"] != "CLO"], min_families=2, max_tier=2)
    widest = disp.groupby(["cal_q", "borrower_key"])["V1_range"].max() * 100
    return (widest >= GAP_POINTS - 1e-9).groupby("cal_q").mean() * 100


def radar_indicators(tr: pd.DataFrame) -> tuple[dict, int]:
    """({group: indicators per quarter}, number of BDCs in the constant panel)."""
    p = positions(tr)
    quarters_present = p.groupby("cik")["cal_q"].nunique()
    panel = quarters_present.index[quarters_present == p["cal_q"].nunique()]
    listed = p["bdc_type"] == "listed"
    out = {"all": indicators(p), "constant": indicators(p[p["cik"].isin(panel)]),
           "listed": indicators(p[listed]), "non_listed": indicators(p[~listed])}
    out["all"]["mark_gap"] = mark_gap(tr)
    out["all"]["mark_gap_same_loan"] = mark_gap_same_loan(tr)
    return out, len(panel)


def comparison(ours: dict) -> pd.DataFrame:
    """Radar and our figure side by side: all BDCs at three quarters, then the groups at 2026Q2."""
    rows = list(RADAR.index) + ["mark_gap_same_loan"]
    t = pd.DataFrame(index=rows)
    for q in ("2025Q2", "2026Q1", "2026Q2"):
        t[f"Radar {q}"] = RADAR[q]
        t[f"ours {q}"] = ours["all"].loc[q]
    for g in ("constant", "listed", "non_listed"):
        t[f"Radar {g}"] = RADAR[g]
        t[f"ours {g}"] = ours[g].loc["2026Q2"]
    return t.round(2)


# ---------------------------------------------------------------- definitions, on a hand-made panel

def row(chain, cal_q, price, pik=0.0, spread=0.05, maturity="2030-06-30", cost=100.0, non_accrual=0,
        borrower="B", family="F1", cik=1, lien_group="1L"):
    return {"chain_id": chain, "cal_q": cal_q, "qsort": int(cal_q[:4]) * 4 + int(cal_q[-1]), "price": price,
            "PIK_Final": pik, "spread": spread, "maturity": pd.Timestamp(maturity), "COST": cost,
            "non_accrual": non_accrual, "borrower_key": borrower, "family": family, "cik": cik,
            "lien_group": lien_group, "bdc_type": "listed", "FV_px": price * 100.0, "PAR_px": 100.0}


class TestRadarDefinitions(unittest.TestCase):

    def flags(self, rows, quarter="2026Q2"):
        p = positions(pd.DataFrame(rows))
        return p[p["cal_q"] == quarter].set_index("chain_id")

    def test_marks(self):
        rows = [row(1, "2026Q1", 0.96), row(1, "2026Q2", 0.89),      # below 90, down 7
                row(2, "2026Q1", 0.85), row(2, "2026Q2", 0.84),      # below 90 and 80 is not: 84
                row(3, "2026Q1", 1.00), row(3, "2026Q2", 0.79),      # below 80, down 21
                row(4, "2025Q4", 1.00), row(4, "2026Q2", 0.90),      # 90 is not below 90; a skipped quarter is not followed
                row(5, "2026Q1", 1.00), row(5, "2026Q2", 0.50, lien_group="CLO")]   # CLO: out of scope
        f = self.flags(rows)
        self.assertEqual(list(f.index), [1, 2, 3, 4])
        self.assertEqual(f["below90"].tolist(), [True, True, True, False])
        self.assertEqual(f["below80"].tolist(), [False, False, True, False])
        self.assertEqual(f["down5"].tolist(), [True, False, True, False])
        self.assertEqual(f["followed"].tolist(), [True, True, True, False])
        q2 = indicators(positions(pd.DataFrame(rows))).loc["2026Q2"]
        self.assertAlmostEqual(q2["below90"], 75.0)
        self.assertAlmostEqual(q2["down5"], 200 / 3)                 # 2 of the 3 followed positions

    def test_amendment_tiers(self):
        rows = [row(1, "2026Q1", 0.99), row(1, "2026Q2", 0.99, pik=0.03),                    # switch, no stress: tier 3
                row(2, "2026Q1", 0.99, pik=0.03), row(2, "2026Q2", 0.99, pik=0.05),          # already PIK: no switch
                row(3, "2026Q1", 0.86), row(3, "2026Q2", 0.85, spread=0.056),                # spread +60 bp below 90: tier 2
                row(4, "2026Q1", 0.99), row(4, "2026Q2", 0.99, spread=0.054),                # +40 bp: nothing
                row(5, "2026Q1", 0.99, maturity="2026-09-30"),
                row(5, "2026Q2", 0.93, maturity="2028-09-30"),                               # extended near maturity, down 6: tier 2
                row(6, "2026Q1", 0.99), row(6, "2026Q2", 0.99, maturity="2031-06-30")]       # extended four years out: nothing
        f = self.flags(rows)
        self.assertEqual(f["pik_switch"].tolist(), [True, False, False, False, False, False])
        self.assertEqual(f["tier2"].tolist(), [False, False, True, False, True, False])
        self.assertEqual(f["tier3"].tolist(), [True, False, False, False, False, False])

    def test_non_accrual_share_of_all_cost(self):
        rows = [row(1, "2026Q2", 0.50, cost=30.0, non_accrual=1), row(2, "2026Q2", 1.0, cost=50.0),
                row(3, "2026Q2", 1.0, cost=20.0, non_accrual=float("nan"))]                  # unknown: in the denominator
        self.assertAlmostEqual(indicators(positions(pd.DataFrame(rows))).loc["2026Q2", "non_accrual"], 30.0)

    def test_mark_gap_is_between_families(self):
        rows = [row(1, "2026Q2", 0.95, borrower="X", family="F1"), row(2, "2026Q2", 0.89, borrower="X", family="F2"),
                # Y: one family's two BDCs are one holder (mark 94.5), 2.5 points from the other family
                row(3, "2026Q2", 0.99, borrower="Y", family="F1", cik=1), row(4, "2026Q2", 0.90, borrower="Y", family="F1", cik=2),
                row(5, "2026Q2", 0.97, borrower="Y", family="F2"),
                row(6, "2026Q2", 0.60, borrower="Z", family="F1")]                           # one family: not counted
        self.assertAlmostEqual(mark_gap(pd.DataFrame(rows)).loc["2026Q2"], 50.0)


# ---------------------------------------------------------------- the Radar's figures and findings, on our data

@unittest.skipUnless(FINAL_CSV.exists(), f"needs {FINAL_CSV}")
class TestRadarOnOurData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ours, cls.n_panel = radar_indicators(T.build_tranche_panel(MP.load_rows(FINAL_CSV)))
        cls.all = cls.ours["all"]
        print(f"\nPercent. Constant panel: {cls.n_panel} BDCs (Radar: 87).")
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(comparison(cls.ours).to_string(na_rep="-"))
            print("\nOur series, all BDCs:")
            print(cls.all.round(2).to_string(na_rep="-"))

    def test_levels_within_band(self):
        for q in ("2025Q2", "2026Q1", "2026Q2"):
            for name, radar in RADAR[q].items():
                ours = self.all.loc[q, name]
                self.assertTrue(radar / LEVEL_BAND <= ours <= radar * LEVEL_BAND, f"{name} {q}: ours {ours:.2f}, Radar {radar}")

    def test_all_six_higher_than_a_year_earlier(self):
        for name in SIX:
            self.assertGreater(self.all.loc["2026Q2", name], self.all.loc["2025Q2", name], name)

    def test_marks_broke_in_2026Q1(self):
        self.assertEqual(self.all["down5"].idxmax(), "2026Q1")          # the broadest markdown in the series
        self.assertEqual(self.all["non_accrual"].idxmax(), "2026Q1")    # non-accrual peaked that quarter
        self.assertGreater(self.all.loc["2026Q2", "below90"], self.all.loc["2025Q4", "below90"])   # and marks have not recovered

    def test_new_funds_hold_the_headline_down(self):
        self.assertGreater(self.ours["constant"].loc["2026Q2", "below90"], self.all.loc["2026Q2", "below90"])

    def test_listed_bdcs_carry_more_stress(self):
        for name in ("below90", "non_accrual"):
            self.assertGreater(self.ours["listed"].loc["2026Q2", name], self.ours["non_listed"].loc["2026Q2", name], name)


if __name__ == "__main__":
    unittest.main()
