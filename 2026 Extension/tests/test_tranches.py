"""
Tests for tranche matching (src/1_data_processing/tranches.py), manager families
(manager_family.py) and family-level crowding (concentration_crowding.py).
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "1_data_processing"))
from paths import add_code_dirs_to_path
add_code_dirs_to_path()

import tranches as T
from manager_family import manager
from concentration_crowding import compute_borrower_crowding_metrics


def rows(records):
    """FINAL_DATASET-style rows: (cik, family, cal_q, borrower, seniority, maturity, spread, par, cost, fv[, pik])."""
    out = []
    for r in records:
        cik, fam, q, b, sen, mat, spr, par, cost, fv = r[:10]
        out.append({
            "cik": cik, "family": fam, "cal_q": q, "borrower_canonical": b, "instrument_seniority": sen,
            "investment_identifier": f"{b} loan", "InvestmentMaturityDate": mat,
            "InvestmentBasisSpreadVariableRate_normalized": spr,
            "InvestmentOwnedBalancePrincipalAmount_normalized": par,
            "InvestmentOwnedAtCost_normalized": cost, "InvestmentOwnedAtFairValue_normalized": fv,
            "PIK_Final": r[10] if len(r) > 10 else 0.0, "IR_Final": 0.10, "sector": "Industrials",
        })
    return pd.DataFrame(out)


def panel(records, origination=False):
    return T.build_tranche_panel(rows(records), with_origination=origination)


class TestChains(unittest.TestCase):

    def test_lien_relabel_keeps_chain(self):
        # Same loan; the BDC's label moves from 1L to OTHER_UNKNOWN (an ANY lien) between quarters.
        tr = panel([
            (1, "F1", "2024Q1", "ACME", "SENIOR_SECURED_1L", None, 0.05, 100, 99, 98),
            (1, "F1", "2024Q2", "ACME", "OTHER_UNKNOWN", None, 0.05, 100, 99, 97),
        ])
        self.assertEqual(tr["chain_id"].nunique(), 1)
        self.assertEqual(sorted(tr["link_rule"]), [0, 3])

    def test_known_liens_never_link(self):
        # A first lien that disappears and a second lien that appears are different loans.
        tr = panel([
            (1, "F1", "2024Q1", "ACME", "SENIOR_SECURED_1L", None, 0.05, 100, 99, 98),
            (1, "F1", "2024Q2", "ACME", "SENIOR_SECURED_2L", None, 0.08, 100, 99, 97),
        ])
        self.assertEqual(tr["chain_id"].nunique(), 2)

    def test_repricing_and_rescue_loan(self):
        # Computer Data Source pattern: no maturity tagged; the term loan reprices 8.25% -> 10.25%
        # while a small new rescue loan with no spread appears. The term loan keeps its chain
        # and the rescue loan starts its own.
        tr = panel([
            (1, "F1", "2026Q1", "CDS", "SENIOR_SECURED_UNSPEC", None, 0.0825, 31.0, 31.0, 26.6),
            (1, "F1", "2026Q1", "CDS", "SENIOR_SECURED_UNSPEC", None, None, 0.76, 0.74, 0.74),
            (1, "F1", "2026Q2", "CDS", "SENIOR_SECURED_UNSPEC", None, 0.1025, 31.1, 31.1, 20.3),
            (1, "F1", "2026Q2", "CDS", "SENIOR_SECURED_UNSPEC", None, None, 0.79, 0.78, 0.78),
        ])
        term = tr[tr["PAR"] > 10].sort_values("cal_q")
        rescue = tr[tr["PAR"] < 10].sort_values("cal_q")
        self.assertEqual(term["chain_id"].nunique(), 1)
        self.assertEqual(rescue["chain_id"].nunique(), 1)
        self.assertNotEqual(term["chain_id"].iloc[0], rescue["chain_id"].iloc[0])
        self.assertEqual(term["link_rule"].tolist(), [0, 4])


class TestCrossBDC(unittest.TestCase):

    def test_maturity_match_is_tier_1(self):
        tr = panel([
            (1, "F1", "2025Q1", "ACME", "SENIOR_SECURED_1L", "2030-06-30", 0.05, 100, 99, 98),
            (2, "F2", "2025Q1", "ACME", "SENIOR_SECURED_1L", "2030-07-15", 0.05, 50, 49, 90),
        ])
        self.assertEqual(tr["xb_id"].nunique(), 1)
        self.assertEqual(tr["xb_tier"].tolist(), [1.0, 1.0])

    def test_wildcard_joins_on_spread(self):
        # BDC 2 tags no maturity and labels the loan OTHER_UNKNOWN; the spread singles out the 1L
        # term loan (not the 2L), so it joins at tier 2.
        tr = panel([
            (1, "F1", "2025Q1", "ACME", "SENIOR_SECURED_1L", "2030-06-30", 0.05, 100, 99, 98),
            (1, "F1", "2025Q1", "ACME", "SENIOR_SECURED_2L", "2031-06-30", 0.09, 40, 39, 30),
            (2, "F2", "2025Q1", "ACME", "OTHER_UNKNOWN", None, 0.05, 50, 49, 47),
        ])
        first = tr[tr["spread"] == 0.05]
        self.assertEqual(first["xb_id"].nunique(), 1)
        self.assertEqual(sorted(first["xb_tier"]), [1.0, 2.0])
        self.assertNotEqual(tr.loc[tr["lien"] == "SENIOR_SECURED_2L", "xb_id"].iloc[0], first["xb_id"].iloc[0])

    def test_no_terms_fallback_only_for_single_loan(self):
        # Neither BDC reports maturity or spread: matched (tier 3) only while each holds one loan.
        one = panel([
            (1, "F1", "2025Q1", "ACME", "SENIOR_SECURED_1L", None, None, 100, 99, 98),
            (2, "F2", "2025Q1", "ACME", "SENIOR_SECURED_1L", None, None, 50, 49, 47),
        ])
        self.assertEqual(one["xb_id"].nunique(), 1)
        self.assertEqual(one["xb_tier"].tolist(), [3.0, 3.0])
        two = panel([
            (1, "F1", "2025Q1", "ACME", "SENIOR_SECURED_1L", None, None, 100, 99, 98),
            (1, "F1", "2025Q1", "ACME", "SENIOR_SECURED_1L", None, 0.05, 30, 29, 29),
            (2, "F2", "2025Q1", "ACME", "SENIOR_SECURED_1L", None, None, 50, 49, 47),
        ])
        unmatched = two[two["spread"].isna()]
        self.assertEqual(unmatched["xb_id"].nunique(), 2)
        self.assertTrue(unmatched["xb_tier"].isna().all())


class TestPricesAndDispersion(unittest.TestCase):

    def test_partly_drawn_line_not_priced(self):
        # Par is the whole 60m commitment but only 8m is drawn (cost); FV / par would read 0.12.
        tr = panel([(1, "F1", "2025Q1", "ACME", "SENIOR_SECURED_1L", "2030-06-30", 0.05, 60, 8.4, 7.3)])
        self.assertTrue(np.isnan(tr["price"].iloc[0]))
        self.assertEqual(tr["chain_id"].nunique(), 1)          # still in the chains

    def test_family_counts_once(self):
        # Two funds of family F1 and one of F2 on the same loan: two holders, not three.
        tr = panel([
            (1, "F1", "2025Q1", "ACME", "SENIOR_SECURED_1L", "2030-06-30", 0.05, 100, 100, 90),
            (2, "F1", "2025Q1", "ACME", "SENIOR_SECURED_1L", "2030-06-30", 0.05, 100, 100, 90),
            (3, "F2", "2025Q1", "ACME", "SENIOR_SECURED_1L", "2030-06-30", 0.05, 100, 100, 80),
        ])
        d = T.tranche_dispersion(tr)
        self.assertEqual(d["n_families"].iloc[0], 2)
        self.assertAlmostEqual(d["V4_consensus"].iloc[0], 0.85)
        self.assertAlmostEqual(d["V1_range"].iloc[0], 0.10)
        self.assertAlmostEqual(d["V2_low_mark"].iloc[0], -0.05)

    def test_one_family_has_no_dispersion(self):
        tr = panel([
            (1, "F1", "2025Q1", "ACME", "SENIOR_SECURED_1L", "2030-06-30", 0.05, 100, 100, 90),
            (2, "F1", "2025Q1", "ACME", "SENIOR_SECURED_1L", "2030-06-30", 0.05, 100, 100, 70),
        ])
        self.assertTrue(T.tranche_dispersion(tr).empty)


class TestOrigination(unittest.TestCase):

    def test_pik_added_after_origination(self):
        # BDC 1 holds a cash-pay loan from 2024Q2 (after the panel start). BDC 2 enters in 2024Q4
        # when it is already PIK: P3 is False for both, so BDC 2's PIK is acquired.
        tr = panel([
            (1, "F1", "2024Q1", "OTHER", "SENIOR_SECURED_1L", "2029-01-31", 0.06, 10, 10, 10),
            (1, "F1", "2024Q2", "ACME", "SENIOR_SECURED_1L", "2030-06-30", 0.05, 100, 99, 98, 0.0),
            (1, "F1", "2024Q3", "ACME", "SENIOR_SECURED_1L", "2030-06-30", 0.05, 100, 99, 95, 0.0),
            (1, "F1", "2024Q4", "ACME", "SENIOR_SECURED_1L", "2030-06-30", 0.05, 102, 99, 90, 0.05),
            (2, "F2", "2024Q4", "ACME", "SENIOR_SECURED_1L", "2030-06-30", 0.05, 51, 50, 45, 0.05),
        ], origination=True)
        acme = tr[tr["borrower_key"] == "ACME"]
        self.assertFalse(acme["pik_at_origination"].any())
        self.assertFalse(acme["origination_censored"].any())


class TestFamiliesAndCrowding(unittest.TestCase):

    def test_manager_map(self):
        self.assertEqual(manager(1396440, "Main Street Capital Corp"), "Main Street")
        self.assertEqual(manager(1535778, "HMS Income Fund, Inc."), "Main Street")
        self.assertEqual(manager(1508655, "TPG Specialty Lending, Inc."), "Sixth Street")
        self.assertEqual(manager(1, "Some Standalone BDC"), "Some Standalone BDC")

    def test_crowding_metrics(self):
        df = pl.DataFrame({
            "cal_q": ["2024Q1"] * 3 + ["2025Q1"] * 3,
            "borrower_canonical": ["ACME"] * 6,
            "cik": [1, 2, 3, 1, 2, 4],
            "family": ["F1", "F1", "F2", "F1", "F1", "F3"],
            "PAR": [50.0, 25.0, 25.0, 50.0, 25.0, 25.0],
            "FV": [50.0, 25.0, 25.0, 50.0, 25.0, 25.0],
        })
        m = compute_borrower_crowding_metrics(df).sort("cal_q")
        self.assertEqual(m["n_families"].to_list(), [2, 2])          # C1: 3 BDCs but 2 families
        self.assertEqual(m["n_ciks"].to_list(), [3, 3])
        np.testing.assert_allclose(m["lender_hhi"].to_numpy(), [0.625, 0.625])   # 0.75^2 + 0.25^2
        np.testing.assert_allclose(m["top_share"].to_numpy(), [0.75, 0.75])
        self.assertEqual(m["d_families_4q"].to_list(), [None, 0])        # C4: null before the panel has 4 quarters


class TestDistressEvents(unittest.TestCase):

    def test_persistent_recovered_exited(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "8_risk_monitor"))
        import distress_event_study as D
        rows = []
        # A: -40% then -10%: persistent. B: -30% then +50%: recovered. C: -25% then gone: exited.
        for b, rets in [("A", [0.0, -0.4, -0.1, 0.0]), ("B", [0.0, -0.3, 0.5, 0.0]), ("C", [0.0, -0.25])]:
            for i, r in enumerate(rets):
                rows.append(dict(borrower=b, cal_q=f"2024Q{i + 1}", qsort=2024 * 4 + i + 1, val_ret=r if i else np.nan,
                                 fv_prev=5e6, price=0.9, sector="X", C1_families=1, multi_family=False,
                                 log_borrower_par=16.0))
        rows.append(dict(borrower="Z", cal_q="2024Q4", qsort=2024 * 4 + 4, val_ret=0.0, fv_prev=5e6, price=1.0,
                         sector="X", C1_families=1, multi_family=False, log_borrower_par=16.0))
        ev = D.find_events(pd.DataFrame(rows)).set_index("borrower")
        self.assertEqual(ev.loc["A", "status"], "persistent")
        self.assertEqual(ev.loc["B", "status"], "recovered")
        self.assertEqual(ev.loc["C", "status"], "exited")
        self.assertEqual(ev.loc["A", "cal_q"], "2024Q2")


if __name__ == "__main__":
    unittest.main()
