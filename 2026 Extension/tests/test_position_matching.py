"""
Tests for matching positions on borrower + seniority (index_construction.py):
add_position_key, aggregate_positions and compute_position_level_flows.
"""

import sys
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "1_data_processing"))
from paths import add_code_dirs_to_path
add_code_dirs_to_path()

from index_construction import (
    add_position_key,
    aggregate_positions,
    compute_position_level_flows,
    qsort_expr,
    safe_div,
)


def panel():
    # One BDC, one borrower, two first-lien tranches in 2024Q1. In 2024Q2 the BDC rewords the
    # identifiers (the FY2025 10-K style change), which must not break the match.
    return pl.DataFrame({
        "cik": ["1", "1", "1", "1", "1"],
        "cal_q": ["2024Q1", "2024Q1", "2024Q1", "2024Q2", "2024Q2"],
        "investment_identifier": ["Acme, LLC, Term loan", "Acme, LLC, Revolver", "Acme, LLC, Equity",
                                  "Acme, LLC | Term loan | Non-Affiliated Issuer", "Mystery Co"],
        "borrower_canonical": ["ACME", "ACME", "ACME", "ACME", None],
        "instrument_seniority": ["SENIOR_SECURED_1L", "SENIOR_SECURED_1L", "OTHER_UNKNOWN", "SENIOR_SECURED_1L", "OTHER_UNKNOWN"],
        "sector": ["Industrials"] * 5,
        "FV": [90.0, 10.0, 5.0, 101.0, 7.0],
        "COST": [95.0, 10.0, 4.0, 100.0, 7.0],
        "PAR": [100.0, None, None, 100.0, 7.0],
        "PIC_Final": [0.10, 0.08, 0.0, 0.11, 0.09],
        "PIK_Final": [0.02, 0.0, 0.0, 0.0, 0.0],
        "IR_Final": [0.12, 0.08, 0.0, 0.11, 0.09],
    })


class TestPositionMatching(unittest.TestCase):
    def test_position_key(self):
        keys = add_position_key(panel())["position_key"].to_list()
        self.assertEqual(keys[0], "ACME | SENIOR_SECURED_1L")
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(keys[2], "ACME | OTHER_UNKNOWN")
        self.assertEqual(keys[4], "ID: Mystery Co | OTHER_UNKNOWN")  # no borrower -> identifier fallback

    def test_aggregate_preserves_amounts_and_income(self):
        agg = aggregate_positions(panel()).filter(
            (pl.col("cal_q") == "2024Q1") & (pl.col("position_key") == "ACME | SENIOR_SECURED_1L"))
        self.assertEqual(agg.height, 1)
        row = agg.row(0, named=True)
        self.assertEqual(row["n_rows"], 2)
        self.assertAlmostEqual(row["FV"], 100.0)
        self.assertAlmostEqual(row["COST"], 105.0)
        self.assertAlmostEqual(row["PAR"], 100.0)
        # rate x PAR equals the rows' total income (the revolver has no PAR, so no income)
        self.assertAlmostEqual(row["PIC_Final"] * row["PAR"], 0.10 * 100.0)
        self.assertEqual(row["investment_identifier"], "Acme, LLC, Term loan")  # largest-FV row

    def test_reworded_identifier_still_matches(self):
        flows = compute_position_level_flows(panel(), qsort_expr, safe_div)
        q2 = flows.filter((pl.col("cal_q") == "2024Q2") & (pl.col("position_key") == "ACME | SENIOR_SECURED_1L")).row(0, named=True)
        self.assertAlmostEqual(q2["FV_prev"], 100.0)
        self.assertAlmostEqual(q2["cash_income"], 0.10 * 100.0 / 4)
        # 2024Q1 + 2024Q2 rows collapse to one per (cik, position_key, quarter)
        self.assertEqual(flows.select(["cik", "position_key", "cal_q"]).unique().height, flows.height)


if __name__ == "__main__":
    unittest.main()
