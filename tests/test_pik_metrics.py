"""
test_pik_metrics.py

Comprehensive unit test suite for PIK risk detection, provenance tracking,
loan transition states, and the three distinct non-cash metrics.
"""

import sys
from pathlib import Path
import unittest
import polars as pl
import numpy as np

# Project root (for paths.py) and every code folder on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "1_data_processing"))
from paths import add_code_dirs_to_path
add_code_dirs_to_path()

from pik_metrics import (
    classify_pik_provenance,
    aggregate_pik_provenance,
    classify_loan_pik_transitions,
    aggregate_pik_transitions,
    compute_non_cash_metrics,
    aggregate_non_cash_metrics,
    compute_sector_time_series_pik_metrics,
)


class TestPIKMetrics(unittest.TestCase):

    def test_classify_pik_provenance(self):
        """Test classification of explicit, inferred, and zero PIK rows."""
        df = pl.DataFrame({
            "rate_pik": [0.03, None, None, 0.0, None],
            "PIK_Final": [0.03, 0.025, 0.0, 0.0, None],
        })
        result = classify_pik_provenance(df)

        expected_prov = ["explicit_pik", "inferred_pik", "zero_pik", "zero_pik", "zero_pik"]
        self.assertEqual(result["pik_provenance"].to_list(), expected_prov)
        self.assertEqual(result["is_explicit_pik"].to_list(), [True, False, False, False, False])
        self.assertEqual(result["is_inferred_pik"].to_list(), [False, True, False, False, False])

    def test_aggregate_pik_provenance(self):
        """Test quarterly and CIK aggregation of PIK provenance."""
        df = pl.DataFrame({
            "cal_q": ["2024Q1", "2024Q1", "2024Q2", "2024Q2"],
            "cik": ["100", "100", "200", "200"],
            "rate_pik": [0.02, None, 0.03, None],
            "PIK_Final": [0.02, 0.02, 0.03, 0.0],
            "FV": [100.0, 200.0, 300.0, 400.0],
            "PAR": [100.0, 200.0, 300.0, 400.0],
        })
        summary = aggregate_pik_provenance(df)
        q_df = summary["quarterly"]
        cik_df = summary["by_cik"]

        self.assertIn("quarterly", summary)
        self.assertIn("by_cik", summary)
        # Check CIK 100 has 1 explicit and 1 inferred
        c100 = cik_df.filter(pl.col("cik") == "100")
        self.assertEqual(c100["n_explicit_pik"][0], 1)
        self.assertEqual(c100["n_inferred_pik"][0], 1)

    def test_classify_loan_pik_transitions(self):
        """Test multi-period loan state transition tracking along tranche chains."""
        # Loan A: cash (Q1), PIK switched on in Q2 (P2 toggle), continues in Q3, cures in Q4.
        # Loan B: PIK at origination (Q1, 20% of the coupon), unchanged in Q2, PIK share up to
        #         45% in Q3 (P2: share rises >= 10 points), up 3 points more in Q4 (noise, no P2).
        # Loan C: a BDC enters in Q2 already on PIK, but the loan was cash-pay when it first
        #         appeared elsewhere (pik_at_origination False): acquired, not contractual.
        df = pl.DataFrame({
            "cik": ["A"] * 4 + ["B"] * 4 + ["C"] * 2,
            "tranche_id": ["Loan_A"] * 4 + ["Loan_B"] * 4 + ["Loan_C"] * 2,
            "cal_q": ["2024Q1", "2024Q2", "2024Q3", "2024Q4"] * 2 + ["2024Q2", "2024Q3"],
            "PIK_Final": [0.0, 0.03, 0.03, 0.0, 0.02, 0.02, 0.045, 0.048, 0.05, 0.05],
            "IR_Final": [0.10, 0.11, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.11, 0.11],
            "PAR": [100.0, 100.75, 101.5, 101.5, 50.0, 50.25, 50.8, 51.4, 30.0, 30.4],
            "FV": [100.0, 98.0, 95.0, 99.0, 50.0, 50.0, 48.0, 47.0, 28.0, 27.0],
            "pik_at_origination": [False] * 4 + [True] * 4 + [False] * 2,
        })

        res = classify_loan_pik_transitions(df)

        res_a = res.filter(pl.col("tranche_id") == "Loan_A").sort("cal_q")
        self.assertEqual(res_a["pik_transition_state"].to_list(), [
            "stable_cash", "distress_amendment_pik", "seasoned_pik_continuation", "cured_pik",
        ])
        self.assertEqual(res_a["is_distress_amendment"].to_list(), [False, True, False, False])
        self.assertEqual(res_a["pik_acquired"].to_list(), [False, True, True, False])

        res_b = res.filter(pl.col("tranche_id") == "Loan_B").sort("cal_q")
        self.assertEqual(res_b["pik_transition_state"].to_list(), [
            "contractual_pik", "seasoned_pik_continuation", "distress_amendment_pik", "seasoned_pik_continuation",
        ])
        np.testing.assert_allclose(res_b["pik_share"].to_numpy(), [0.2, 0.2, 0.45, 0.48])

        res_c = res.filter(pl.col("tranche_id") == "Loan_C").sort("cal_q")
        self.assertEqual(res_c["pik_transition_state"].to_list(), ["acquired_pik", "seasoned_pik_continuation"])
        self.assertEqual(res_c["pik_acquired"].to_list(), [True, True])

    def test_pik_under_25bp_is_noise(self):
        """A PIK rate under 25 bp (total minus cash rounding) is not a toggle."""
        df = pl.DataFrame({
            "cik": ["A", "A"], "tranche_id": ["L", "L"], "cal_q": ["2024Q1", "2024Q2"],
            "PIK_Final": [0.0, 0.001], "IR_Final": [0.10, 0.10], "PAR": [100.0, 100.0], "FV": [99.0, 99.0],
        })
        res = classify_loan_pik_transitions(df).sort("cal_q")
        self.assertEqual(res["is_distress_amendment"].to_list(), [False, False])

    def test_compute_non_cash_metrics(self):
        """Test calculation of the three distinct non-cash metrics."""
        df = pl.DataFrame({
            "cal_q": ["2024Q1", "2024Q2"],
            "IR_Final": [0.12, 0.10],
            "PIK_Final": [0.03, 0.05],
            "cash_income": [9.0, 5.0],
            "pik_income": [3.0, 5.0],
            "PAR": [103.0, 108.0],
            "PAR_incept": [100.0, 100.0],
            "w_mkt": [0.6, 0.4],
        })

        res = compute_non_cash_metrics(df)

        # Metric A: Rate share = PIK / IR
        # 0.03 / 0.12 = 0.25; 0.05 / 0.10 = 0.50
        np.testing.assert_allclose(res["rate_share"].to_numpy(), [0.25, 0.50], rtol=1e-5)

        # Metric B: Income share = pik_income / (cash_income + pik_income)
        # 3 / 12 = 0.25; 5 / 10 = 0.50
        np.testing.assert_allclose(res["income_share"].to_numpy(), [0.25, 0.50], rtol=1e-5)

        # Metric C: Cap burden = (PAR - PAR_incept) / PAR
        # (103 - 100) / 103 = 3 / 103 ≈ 0.029126
        # (108 - 100) / 108 = 8 / 108 ≈ 0.074074
        np.testing.assert_allclose(res["cap_burden"].to_numpy(), [3.0 / 103.0, 8.0 / 108.0], rtol=1e-5)

    def test_sector_time_series_sweep(self):
        """Test sector time-series multi-period sweep."""
        df = pl.DataFrame({
            "cik": ["1", "1", "2", "2"],
            "tranche_id": ["L1", "L1", "L2", "L2"],
            "cal_q": ["2024Q1", "2024Q2", "2024Q1", "2024Q2"],
            "sector": ["Technology", "Technology", "Healthcare", "Healthcare"],
            "rate_pik": [0.0, 0.04, 0.02, 0.02],
            "PIK_Final": [0.0, 0.04, 0.02, 0.02],
            "IR_Final": [0.10, 0.12, 0.10, 0.10],
            "cash_income": [10.0, 8.0, 8.0, 8.0],
            "pik_income": [0.0, 4.0, 2.0, 2.0],
            "PAR": [100.0, 104.0, 100.0, 102.0],
            "FV": [100.0, 95.0, 100.0, 100.0],
            "w_mkt": [0.5, 0.5, 0.5, 0.5],
        })

        sweep = compute_sector_time_series_pik_metrics(df)
        self.assertEqual(sweep.height, 4)  # 2 sectors x 2 quarters
        self.assertIn("pct_amendment_count", sweep.columns)
        self.assertIn("mean_rate_share", sweep.columns)

        # Check Technology in 2024Q2 had 1 distress amendment
        tech_q2 = sweep.filter((pl.col("sector") == "Technology") & (pl.col("cal_q") == "2024Q2"))
        self.assertEqual(tech_q2["n_distress_amendments"][0], 1)


if __name__ == "__main__":
    unittest.main()

