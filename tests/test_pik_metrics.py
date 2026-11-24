"""Tests for classify_loan_pik_transitions (src/5_pik/pik_metrics.py): PIK switch-on/off per loan,
the input to the PIK warning sign."""

import sys
from pathlib import Path
import unittest
import polars as pl
import numpy as np

# Project root (for paths.py) and every code folder on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "1_data_processing"))
from paths import add_code_dirs_to_path
add_code_dirs_to_path()

from pik_metrics import classify_loan_pik_transitions


class TestPIKMetrics(unittest.TestCase):


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

    def test_clean_switch_skips_filing_artifacts(self):
        """pik_switch_clean: a real switch and a real rise count; PIK back after one filing at zero
        with the same total coupon (the PIK split missing) does not, nor a share rise from a lower
        coupon with the PIK rate unchanged. A real election (coupon lower at the zero filing) does."""
        q = ["2024Q1", "2024Q2", "2024Q3"]
        rows = {"S": ([0.0, 0.03, 0.03], [0.10, 0.11, 0.11]),      # cash, then PIK starts
                "R": ([0.02, 0.045, 0.045], [0.10, 0.10, 0.10]),   # PIK rate and share up
                "F": ([0.03, 0.0, 0.03], [0.11, 0.11, 0.11]),      # PIK split missing from one filing
                "E": ([0.03, 0.0, 0.03], [0.11, 0.08, 0.11]),      # elected cash for a quarter, then PIK
                "C": ([0.04, 0.04, 0.04], [0.11, 0.11, 0.05])}     # coupon falls, PIK unchanged
        df = pl.DataFrame({"cik": [k for k in rows for _ in q], "tranche_id": [k for k in rows for _ in q],
                           "cal_q": q * len(rows), "PIK_Final": sum((v[0] for v in rows.values()), []),
                           "IR_Final": sum((v[1] for v in rows.values()), []), "PAR": [100.0] * 15, "FV": [99.0] * 15})
        res = classify_loan_pik_transitions(df).sort(["tranche_id", "cal_q"])
        got = {k: res.filter(pl.col("tranche_id") == k)["pik_switch_clean"].to_list() for k in rows}
        self.assertEqual(got, {"S": [False, True, False], "R": [False, True, False], "F": [False, False, False],
                               "E": [False, False, True], "C": [False, False, False]})
        old = res.filter(pl.col("tranche_id").is_in(["F", "C"]))["is_distress_amendment"].to_list()
        self.assertEqual(old, [False, False, True, False, False, True])          # the P2 flag counted both


if __name__ == "__main__":
    unittest.main()

