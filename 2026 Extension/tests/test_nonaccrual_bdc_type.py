"""
Tests for the non-accrual footnote classifier (nonaccrual_flags.py) and BDC-type dating
(bdc_type.py).
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "1_data_processing"))
from paths import add_code_dirs_to_path
add_code_dirs_to_path()

from nonaccrual_flags import classify_texts
from bdc_type import classify, type_in_quarter


class TestNonAccrualFootnotes(unittest.TestCase):

    def classify(self, text, share=0.05):
        t = pd.DataFrame({"cik": [1], "accession": ["1"], "footnote": ["fn-1"], "n_contexts": [3], "text": [text]})
        return bool(classify_texts(t, pd.Series({("1", "fn-1"): share}))["is_flag"].iloc[0])

    def test_flags(self):
        for text in ["Loan was on non-accrual status as of June 30, 2026.",
                     "Non-accrual investment.",
                     "Debt investment currently on non-accrual status.",
                     "PIK non-accrual investment",
                     "Partial non-accrual PIK security",
                     "The investment is on non-accrual status. See Note 2 Significant Accounting Policies."]:
            self.assertTrue(self.classify(text), text)

    def test_not_flags(self):
        for text in ["The interest rate on these investments, excluding those on non-accrual, contains a PIK provision.",
                     "All funded debt investments are income producing. As of December 31, 2025, there were no investments on non-accrual.",
                     "As of June 30, 2026, the Company had no portfolio company investments on non-accrual status.",
                     "Gross reductions include decreases in the cost basis of investments, including those on non-accrual.",
                     "Security pays, or has the option to pay, some or all of its interest in kind; certain bonds are on non-accrual status."]:
            self.assertFalse(self.classify(text), text)

    def test_general_note_by_share(self):
        short = "Investment is on non-accrual status."
        self.assertTrue(self.classify(short, share=0.40))         # wind-down fund: still a flag
        self.assertFalse(self.classify(short, share=0.60))        # linked to most lines: a note
        long_text = ("As of December 31, 2025, loans with a cost basis of $10 million and a fair value of $4 million "
                     "were classified as non-accrual. These loans have been accelerated from their original maturity dates.")
        self.assertFalse(self.classify(long_text, share=0.40))


def history(rows, tickers=(), exchanges=()):
    f = pd.DataFrame(rows, columns=["form", "date"])
    f["date"] = pd.to_datetime(f["date"])
    f.attrs.update(name="Test BDC", tickers=list(tickers), exchanges=list(exchanges))
    return f


class TestBDCType(unittest.TestCase):

    def at(self, row, date):
        return type_in_quarter(pd.Series(row), pd.Timestamp(date))

    def test_private_then_listed(self):
        # Morgan Stanley Direct Lending pattern: Form 10 in 2019, IPO in January 2024.
        row = classify(1, history([("10-12G", "2019-09-26"), ("N-2", "2023-12-15"), ("8-A12B", "2024-01-22")],
                                  ["MSDL"], ["NYSE"]))
        self.assertEqual(self.at(row, "2023-12-31"), "private")
        self.assertEqual(self.at(row, "2024-03-31"), "listed")

    def test_non_traded_from_first_prospectus(self):
        row = classify(2, history([("N-2", "2022-04-05")] + [("424B3", f"2023-0{m}-01") for m in range(4, 9)]))
        self.assertEqual(self.at(row, "2023-03-31"), "private")
        self.assertEqual(self.at(row, "2023-06-30"), "non-traded")

    def test_notes_exchange_offer_is_not_an_offering(self):
        row = classify(3, history([("10-12G", "2018-08-10"), ("N-2", "2021-03-16"), ("424B3", "2021-06-08"),
                                   ("424B3", "2021-07-01")]))
        self.assertEqual(self.at(row, "2024-12-31"), "private")

    def test_merged_away_listed_bdc(self):
        # No ticker any more, but a listing and a later delisting: listed until the delisting.
        row = classify(4, history([("8-A12B", "2007-06-25"), ("25-NSE", "2024-03-20")]))
        self.assertEqual(self.at(row, "2023-12-31"), "listed")


if __name__ == "__main__":
    unittest.main()
