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

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
        """Test multi-period loan state transition tracking."""
        # 2 loans across 4 quarters:
        # Loan A: starts pure cash (Q1), converts to PIK in Q2 (distress amendment),
        #         continues PIK in Q3, cures in Q4 back to cash.
        # Loan B: starts contractual PIK in Q1, continues in Q2, jumps +200bps in Q3 (amendment).
        df = pl.DataFrame({
            "cik": ["CIK_A"] * 4 + ["CIK_B"] * 3,
            "investment_identifier": ["Loan_A"] * 4 + ["Loan_B"] * 3,
            "cal_q": ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2024Q1", "2024Q2", "2024Q3"],
            "PIK_Final": [0.0, 0.03, 0.03, 0.0, 0.02, 0.02, 0.045],
            "PAR": [100.0, 100.75, 101.5, 101.5, 50.0, 50.25, 50.8],
            "FV": [100.0, 98.0, 95.0, 99.0, 50.0, 50.0, 48.0],
        })

        res = classify_loan_pik_transitions(df, min_amendment_bps=0.015)

        # Check Loan A classifications
        res_a = res.filter(pl.col("investment_identifier") == "Loan_A").sort("cal_q")
        self.assertEqual(res_a["pik_transition_state"].to_list(), [
            "stable_cash",               # Q1 inception with PIK=0
            "distress_amendment_pik",    # Q2 0 -> >0 transition
            "seasoned_pik_continuation", # Q3 ongoing 0.03
            "cured_pik",                 # Q4 >0 -> 0 transition
        ])
        self.assertEqual(res_a["is_distress_amendment"].to_list(), [False, True, False, False])

        # Check Loan B classifications
        res_b = res.filter(pl.col("investment_identifier") == "Loan_B").sort("cal_q")
        self.assertEqual(res_b["pik_transition_state"].to_list(), [
            "contractual_pik",           # Q1 inception with PIK=0.02
            "seasoned_pik_continuation", # Q2 ongoing 0.02
            "distress_amendment_pik",    # Q3 jumped 0.02 -> 0.045 (+250 bps >= 150 bps)
        ])
        self.assertEqual(res_b["is_distress_amendment"].to_list(), [False, False, True])

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
            "investment_identifier": ["L1", "L1", "L2", "L2"],
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

