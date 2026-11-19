"""
Tests for the early-warning scorecard (src/9_prediction): the borrower-quarter panel (same-loan price
changes, credit events within two quarters: non-accrual or a lender's concession, unlabelled exits
in four buckets, sample rules), term changes and loan exits (credit_events.py), filing dates, and
the estimation helpers (clustered standard errors, no look-ahead, equal-weighted ranking, deciles,
realized against predicted).
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

import common as MC
import config as CFG
import credit_events as CE
import evaluation as EV
import features as FT
import frames as FR
import models.logit as LG
import models.ranking as RK
import panel as MP
import targets as TG

PAR = 10_000_000.0
Q = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]


def tranche(chain, cik, borrower, q, price, link_rule, toggle=False, par=PAR, v1=np.nan, family=None, lien="1L",
            maturity="2029-06-30"):
    return {"chain_id": chain, "cik": cik, "family": family or f"F{cik}", "borrower_key": borrower, "cal_q": q,
            "qsort": MC.q_sort(q), "FV": price * par, "PAR": par, "COST": par, "FV_px": price * par, "PAR_px": par,
            "lien_group": lien, "link_rule": link_rule, "is_distress_amendment": toggle, "pik_share": 0.0,
            "V1_std": v1, "V1_range": v1, "sector": "Software", "maturity": pd.Timestamp(maturity),
            "price": price, "spread": 0.05, "IR_Final": 0.10, "PIK_Final": 0.0, "non_accrual": 0.0}


def toy_panel():
    """ACME: two BDCs' loans (chains 1 and 2) marked down over 2024; BDC 1's loan switches to PIK
    in 2024Q3 and BDC 1 flags it non-accrual in 2024Q4; BDC 2 lends a new loan at 70 in 2024Q4
    (chain 3). Four borrowers held 2024Q1-Q2 that then leave: BETA at par long before maturity,
    DELTA at par at maturity, EPSILON marked at 60, ZETA marked at 90. ETA: held all year at par,
    its maturity extended by a year in 2024Q3. GAMMA CLO 1: a CLO debt tranche, held all year."""
    p1 = [1.00, 0.98, 0.90, 0.85]
    p2 = [1.00, 1.00, 1.00, 0.95]
    t = []
    for i, q in enumerate(Q):
        t.append(tranche(1, 1, "ACME", q, p1[i], 0 if i == 0 else 1, toggle=(q == "2024Q3")))
        t.append(tranche(2, 2, "ACME", q, p2[i], 0 if i == 0 else 1))
    t.append(tranche(3, 2, "ACME", "2024Q4", 0.70, 0))
    for chain, name, price, kw in [(4, "BETA", 1.0, {}), (6, "DELTA", 1.0, {"maturity": "2024-09-15"}),
                                   (7, "EPSILON", 0.60, {}), (8, "ZETA", 0.90, {})]:
        t += [tranche(chain, 1, name, "2024Q1", price, 0, **kw), tranche(chain, 1, name, "2024Q2", price, 1, **kw)]
    t += [tranche(5, 1, "GAMMA CLO 1", q, 0.80, 0 if i == 0 else 1, lien="CLO") for i, q in enumerate(Q)]
    t += [tranche(9, 1, "ETA", q, 1.0, 0 if i == 0 else 1, maturity="2029-06-30" if i < 2 else "2030-06-30")
          for i, q in enumerate(Q)]
    tr = pd.DataFrame(t)
    rows = []
    for r in t:
        na = None
        if r["borrower_key"] == "ACME":
            na = (1.0 if r["cal_q"] == "2024Q4" else 0.0) if r["cik"] == 1 else None   # BDC 2: status unknown
        else:
            na = 0.0
        rows.append({"cik": r["cik"], "cal_q": r["cal_q"], "borrower_canonical": r["borrower_key"],
                     "PAR": r["PAR"], "FV": r["FV"], "COST": r["COST"], "instrument_seniority": "SENIOR_SECURED_1L",
                     "non_accrual": na})
    return MP.build_model_panel(tr, pl.DataFrame(rows, schema_overrides={"non_accrual": pl.Float64}))[0]


class TestPanel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = toy_panel().set_index(["borrower", "cal_q"])

    def test_same_loan_price_change_ignores_new_loans(self):
        # 2024Q4: chain 1 -0.05, chain 2 -0.05; the new loan at 70 has no previous quarter.
        a = self.m.loc["ACME"]
        self.assertAlmostEqual(a.loc["2024Q2", "d_price"], -0.01)
        self.assertAlmostEqual(a.loc["2024Q4", "d_price"], -0.05)

    def test_markdown_outcomes_are_next_quarter_same_loan_changes(self):
        # signals at 2024Q1: the next quarter's change in the mark (-0.01) and val_ret (FV -$0.2m on
        # $20m of prior fair value, cost unchanged); nothing after the last quarter
        a = self.m.loc["ACME"]
        self.assertAlmostEqual(a.loc["2024Q1", "y_dprice"], -0.01)
        self.assertAlmostEqual(a.loc["2024Q1", "y_valret"], -0.01)
        self.assertTrue(np.isnan(a.loc["2024Q4", "y_dprice"]) and np.isnan(a.loc["2024Q4", "y_valret"]))

    def test_rolling_max_drawdown(self):
        a = self.m.loc["ACME"]
        self.assertTrue(np.isnan(a.loc["2024Q1", "max_drawdown"]))     # no earlier quarter
        self.assertAlmostEqual(a.loc["2024Q2", "max_drawdown"], 0.01)
        self.assertAlmostEqual(a.loc["2024Q3", "max_drawdown"], 0.05)
        self.assertAlmostEqual(a.loc["2024Q4", "max_drawdown"], 0.10)

    def test_drawdown_window_is_four_quarters(self):
        d = pd.DataFrame({"borrower": "X", "qsort": range(1, 8),
                          "d_price": [np.nan, -0.10, 0.0, 0.0, 0.0, 0.0, 0.02]})
        dd = FT.rolling_max_drawdown(d).set_index("qsort")["dd"]
        self.assertAlmostEqual(dd[5], 0.10)        # quarters 1-5 include the fall in quarter 2
        self.assertAlmostEqual(dd[6], 0.0)         # quarters 2-6: the fall happened before the window
        self.assertAlmostEqual(dd[7], 0.0)

    def test_nonaccrual_entry_within_two_quarters(self):
        a = self.m.loc["ACME"]
        self.assertEqual(a.loc["2024Q1", "y_na"], 0)                  # accruing in 2024Q2 and 2024Q3
        self.assertEqual(a.loc["2024Q2", "y_na"], 1)                  # flagged in 2024Q4 = t + 2
        self.assertEqual(a.loc["2024Q2", "entry_q"], MC.q_sort("2024Q4"))
        self.assertEqual(a.loc["2024Q2", "outcome_q"], "2024Q3-2024Q4")
        self.assertEqual(a.loc["2024Q2", "y_na_same_lender"], 1)     # BDC 1 reported it accruing in 2024Q2
        self.assertTrue(np.isnan(a.loc["2024Q3", "y_na"]))           # window runs past the data
        self.assertEqual(a.loc["2024Q4", "na_status"], 1)
        self.assertFalse(a.loc["2024Q4", "na_risk_set"])             # already on non-accrual

    def test_credit_event_is_non_accrual_or_a_concession(self):
        # The outcome: non-accrual, or a lender's concession (PIK switch, debt-for-equity,
        # write-down), at t + 1 or t + 2. ACME's chain 1 switches to PIK in 2024Q3, so 2024Q1
        # (window 2024Q2-Q3) is an event although ACME is still accruing; non-accrual alone
        # (y_na, a robustness outcome) says 0.
        a = self.m.loc["ACME"]
        self.assertEqual(a.loc["2024Q3", "restr_pik"], 1)
        self.assertEqual((a.loc["2024Q1", "y_event"], a.loc["2024Q1", "y_na"]), (1, 0))
        self.assertEqual(a.loc["2024Q1", "event_q"], MC.q_sort("2024Q3"))
        self.assertEqual(a.loc["2024Q2", "y_event"], 1)
        self.assertEqual(a.loc["2024Q2", "event_q"], MC.q_sort("2024Q3"))    # the switch comes before the flag
        self.assertEqual(a.loc["2024Q2", "entry_q"], MC.q_sort("2024Q4"))
        self.assertTrue(np.isnan(a.loc["2024Q3", "y_event"]))                # window runs past the data
        self.assertEqual(set(CE.CONCESSIONS), {"pik", "equity", "write_down"})
        self.assertTrue(set(CE.CONCESSIONS) < set(CE.EVENT_TYPES))

    def test_extension_is_counted_beside_the_outcome(self):
        # ETA's maturity is extended in 2024Q3 while it stays at par and accruing: a term change,
        # not a credit event. Only the labelled robustness outcome y_all_terms counts it.
        e = self.m.loc["ETA"]
        self.assertEqual(e.loc["2024Q3", "restr_maturity"], 1)
        for q in ["2024Q1", "2024Q2"]:
            self.assertEqual((e.loc[q, "y_event"], e.loc[q, "y_na"], e.loc[q, "y_all_terms"]), (0, 0, 1))
            self.assertEqual((e.loc[q, "win_maturity"], e.loc[q, "win_pik"]), (1, 0))
        self.assertTrue(np.isnan(e.loc["2024Q1", "event_q"]))

    def test_exits_have_no_label_and_go_in_four_buckets(self):
        # The outcome comes from non-accrual flags and loan terms, never from a mark: a borrower
        # that leaves before the window closes without an event gets no label, at par (BETA, DELTA)
        # or deeply marked down (EPSILON at 60). Its bucket: repaid at par, impaired at 60,
        # unresolved at 90 (impaired at the second cutoff).
        m = self.m
        for q in ["2024Q1", "2024Q2"]:                               # gone in 2024Q3, inside both windows
            for name in ["BETA", "DELTA", "EPSILON", "ZETA"]:
                self.assertTrue(m.loc[(name, q), "exited"])
                self.assertTrue(np.isnan(m.loc[(name, q), "y_event"]))
                self.assertTrue(np.isnan(m.loc[(name, q), "y_na"]))
        got = {n: (m.loc[(n, "2024Q2"), "exit_class"], m.loc[(n, "2024Q2"), "exit_class_second"])
               for n in ["BETA", "DELTA", "EPSILON", "ZETA"]}
        self.assertEqual(got, {"BETA": ("repaid", "repaid"), "DELTA": ("repaid", "repaid"),
                               "EPSILON": ("impaired", "impaired"), "ZETA": ("unresolved", "impaired")})
        self.assertFalse(m.loc[("ACME", "2024Q3"), "exited"])                       # window runs past the data

    def test_event_counts_cover_every_borrower(self):
        m = self.m.reset_index()
        c = MP.event_counts(m).set_index("cal_q")
        parts = ["events", "no_event", "exited", "status_unknown"]
        exits = ["exit_" + k for k in CE.EXIT_CLASSES]
        self.assertTrue((c[parts].sum(axis=1) == c["at_risk"]).all())
        self.assertTrue((c[exits].sum(axis=1) == c["exited"]).all())
        self.assertTrue((c["event_non_accrual"] + c["event_concession_only"] == c["events"]).all())
        q2 = c.loc["2024Q2"]
        self.assertEqual([q2[k] for k in parts], [1, 1, 4, 0])                      # ACME; ETA; four exits
        self.assertEqual([q2[k] for k in exits], [2, 0, 1, 1])                      # BETA, DELTA / EPSILON / ZETA
        self.assertEqual((q2["labelled"], q2["events"]), (2, 1))                    # a low-marked exit is not an event
        self.assertEqual((q2["event_non_accrual"], q2["event_pik"]), (1, 1))        # ACME: flagged, and switched to PIK
        self.assertEqual(q2["no_event_other_change"], 1)                            # ETA's extension
        self.assertNotIn("2024Q3", c.index)                                         # window incomplete
        x = MP.unlabelled_exits(m).set_index("exit_class")
        self.assertEqual(x.loc["all", "exits_primary"], x.loc["all", "exits_second"])   # no exit dropped
        self.assertEqual(x.loc["impaired", "exits_second"], 2)                      # EPSILON and ZETA (2024Q2 in sample)

    def test_mark_uses_cost_where_par_is_missing(self):
        rows = pl.DataFrame({"cal_q": ["2024Q1"] * 3, "borrower_canonical": ["ACME"] * 3,
                             "instrument_seniority": ["SENIOR_SECURED_1L", "OTHER_UNKNOWN", "SENIOR_SECURED_1L"],
                             "PAR": [None, None, 5.0], "FV": [9.0, 50.0, 5.0], "COST": [10.0, 40.0, 5.0]})
        n = FT.no_par_debt(rows).iloc[0]                                            # only the first row: debt, no par
        self.assertEqual((n["borrower"], n["fv_nopar"], n["cost_nopar"]), ("ACME", 9.0, 10.0))

    def test_pik_switch_and_sample(self):
        a = self.m.loc["ACME"]
        self.assertTrue(np.isnan(a.loc["2024Q1", "pik_switch"]))     # first quarter: no switch defined
        self.assertEqual(a.loc["2024Q3", "pik_switch"], 1)
        self.assertFalse(a.loc["2024Q1", "in_sample"])
        self.assertTrue(a.loc["2024Q3", "in_sample"])
        self.assertEqual(a.loc["2024Q3", "has_dispersion"], 0)

    def test_valuation_gap_is_the_range(self):
        # Two lender families mark the same loan at 90 and 100: the gap is 10 points (std: 7.1).
        t = pd.DataFrame([tranche(1, 1, "ACME", "2024Q1", 0.95, 0)]).assign(V1_range=0.10, V1_std=0.10 / np.sqrt(2))
        b = FT.borrower_marks(FT.named_tranches(t)).iloc[0]
        self.assertAlmostEqual(b["dispersion"], 0.10)
        self.assertAlmostEqual(b["dispersion_std"], 0.10 / np.sqrt(2))
        self.assertEqual(b["has_dispersion"], 1)

    def test_clo_vehicle_out_of_sample(self):
        g = self.m.loc["GAMMA CLO 1"]
        self.assertEqual(g.loc["2024Q2", "clo_share"], 1.0)
        self.assertFalse(g["in_sample"].any())

    def test_event_rates_cover_every_sign(self):
        # Each sign alone: every borrower falls in one bucket of every sign, kept by the rule or not.
        import analysis as A
        n = 8
        d = pd.DataFrame({"borrower": [str(i) for i in range(n)], "qsort": 1, "na_risk_set": True,
                          CFG.OUTCOME: [1, 0, 0, 0, 1, 0, 0, 0], "price": np.linspace(0.7, 1.0, n),
                          "d_price": np.linspace(-0.1, 0.02, n), "max_drawdown": [0.2, 0.07, 0.03, 0, 0.12, 0, 0, 0.005],
                          "dispersion": 0.01, "has_dispersion": [1, 0] * 4, "crowding": [1, 2, 3, 6] * 2,
                          "pik_switch": [1, 0, 0, 0, 0, 0, 0, 0]})
        t = A.event_rates(d, 1)
        self.assertEqual(set(t["predictor"]), set(A.BUCKETS) | {"pik_switch"})
        self.assertTrue((t.groupby("predictor")["n"].sum() == n).all())
        dd = t[t["predictor"] == "max_drawdown"].set_index("bucket")
        self.assertEqual(list(dd.index), ["under 1", "1-5", "5-10", "over 10"])
        self.assertEqual(dd.loc["over 10", "events"], 2)


def loan(chain, cik, borrower, q, price=1.0, link=1, spread=0.05, rate=0.10, pik=0.0, toggle=False,
         maturity="2029-06-30", cost=1.0, par=PAR, na=0.0):
    """A tranche-panel row; price and cost per unit of par."""
    return {"chain_id": chain, "cik": cik, "borrower": borrower, "borrower_key": borrower, "qsort": q,
            "lien_group": "1L", "link_rule": link, "price": price, "FV": price * par, "PAR": par, "COST": cost * par,
            "spread": spread, "IR_Final": rate, "PIK_Final": pik, "is_distress_amendment": toggle,
            "maturity": pd.Timestamp(maturity), "non_accrual": na}


class TestCreditEvents(unittest.TestCase):

    def test_term_changes(self):
        rows = [loan(1, 1, "A", 1, link=0), loan(1, 1, "A", 2, spread=0.06, rate=0.11),          # +100 bp: both cutoffs
                loan(2, 1, "B", 1, link=0), loan(2, 1, "B", 2, spread=0.08, rate=0.08),          # prime -> SOFR relabel
                loan(3, 1, "C", 1, link=0), loan(3, 1, "C", 2, spread=0.056, rate=0.0995),       # +60 bp, base fell 65 bp
                loan(4, 1, "D", 1, link=0), loan(4, 1, "D", 2, maturity="2029-10-15"),           # +107 days
                loan(5, 1, "E", 1, link=0, price=0.60), loan(5, 1, "E", 2, price=0.60, cost=0.85),   # cost cut 15
                loan(6, 1, "F", 1, link=0), loan(6, 1, "F", 2, price=0.70, cost=0.70),           # revolver paid down
                loan(7, 1, "G", 1, link=0, pik=0.02), loan(7, 1, "G", 2, pik=0.04, toggle=True),  # PIK share up
                loan(8, 1, "H", 1, link=0), loan(8, 1, "H", 2, pik=0.03, toggle=True)]           # cash to PIK
        L = CE.loans(pd.DataFrame(rows))
        cols = ["pik", "spread", "maturity", "write_down"]
        for name, cut in [("primary", (1, 1, 1, 1)), ("second", (0, 0, 0, 1))]:
            ev = CE.term_changes(L, CE.CUTOFFS[name]).set_index(["borrower", "qsort"])[cols]
            self.assertEqual(ev.loc[("A", 1)].sum(), 0)                              # a chain's first filing
            self.assertEqual(ev.loc[("A", 2), "spread"], 1)
            self.assertEqual(ev.loc[("B", 2), "spread"], 0)
            self.assertEqual(ev.loc[("C", 2), "spread"], cut[0])
            self.assertEqual(ev.loc[("D", 2), "maturity"], cut[1])
            self.assertEqual(ev.loc[("E", 2), "write_down"], cut[3])
            self.assertEqual(ev.loc[("F", 2), "write_down"], 0)                      # marked at cost before
            self.assertEqual(ev.loc[("G", 2), "pik"], cut[2])
            self.assertEqual(ev.loc[("H", 2), "pik"], 1)

    def test_every_exit_in_one_bucket(self):
        # Loans at BDC 1 end in quarter 1 (exit in quarter 2); the data run to quarter 3.
        t = [loan(1, 1, "REPAID", 1, 0.99), loan(2, 1, "OTHER", 1, 0.99), loan(3, 2, "OTHER", 1), loan(3, 2, "OTHER", 2),
             loan(4, 1, "NEWTR", 1, 0.99), loan(5, 1, "NEWTR", 2, link=0), loan(6, 1, "LOW", 1, 0.70),
             loan(7, 1, "NA", 1, 0.99, na=1.0), loan(8, 1, "EQ", 1, 0.99), loan(9, 1, "MID", 1, 0.85),
             loan(10, 3, "NOFILE", 1, 0.99), loan(11, 1, "KEEP", 1), loan(11, 1, "KEEP", 2), loan(11, 1, "KEEP", 3),
             loan(12, 3, "KEEP3", 1), loan(12, 3, "KEEP3", 3)]
        t = pd.DataFrame(t)
        t.loc[t["chain_id"].isin([1, 2, 4, 6, 7, 8, 9, 10]) & (t["qsort"] == 1), "link_rule"] = 0
        L = CE.loans(t)
        d = t[["borrower", "qsort", "cik", "FV", "non_accrual"]]
        ls = TG.lender_status(d)
        eq = pd.DataFrame({"borrower": ["EQ", "KEEP"], "qsort": [2, 1], "cik": [1, 1]})   # BDC 1 reports equity at 1
        ex = CE.loan_exits(L, d, ls, eq, CE.CUTOFFS["primary"]).set_index("chain_id")
        self.assertEqual(sorted(ex.index), list(range(1, 11)))                        # every ended chain; 11, 12 run on
        want = {1: "repaid", 2: "refinanced", 4: "refinanced", 6: "impaired", 7: "impaired", 8: "impaired",
                9: "unresolved", 10: "unresolved", 3: "unresolved", 5: "repaid"}  # 3: BDC 2 skips quarter 3
        self.assertEqual(ex["exit_class"].to_dict(), want)
        self.assertEqual(ex.loc[2, "exit_reason"], "another BDC, already lending")
        self.assertEqual(ex.loc[4, "exit_reason"], "new tranche, same BDC")
        self.assertEqual(ex.loc[7, "exit_reason"], "non-accrual")                     # at 99: impaired comes first
        self.assertEqual(ex.loc[8, "exit_reason"], "replaced by equity")
        self.assertEqual(ex.loc[10, "exit_reason"], "BDC did not file next quarter")
        second = CE.loan_exits(L, d, ls, eq, CE.CUTOFFS["second"]).set_index("chain_id")
        self.assertEqual(second.loc[9, "exit_class"], "impaired")                     # 85 <= 90
        self.assertEqual(second.loc[1, "exit_class"], "repaid")                       # 99 >= 98

    def test_equity_rows_need_an_equity_word(self):
        rows = pl.DataFrame({"cik": [1, 1, 1, 1], "cal_q": ["2024Q1"] * 4,
                             "borrower_canonical": ["A", "B", "C", "D"],
                             "instrument_seniority": ["OTHER_UNKNOWN", "OTHER_UNKNOWN", "SENIOR_SECURED_1L", "OTHER_UNKNOWN"],
                             "PAR": [None, None, None, 5.0], "FV": [3.0, 3.0, 3.0, 3.0], "COST": [2.0, 2.0, 2.0, 2.0],
                             "investment_identifier": ["A Holdings, Series A preferred units", "B Inc 1 | Non-Affiliated",
                                                       "C LLC, Class A units", "D LP, common stock"],
                             "InvestmentOwnedBalanceShares": [100.0, 100.0, None, None]})
        self.assertEqual(CE.equity_holdings(rows)["borrower"].tolist(), ["A"])        # B: shares only; C: debt; D: par


class TestFilingLags(unittest.TestCase):

    def test_days_from_quarter_end_to_filing(self):
        # Two 2024Q1 filings (a 10-Q after 40 days, a 10-K after 91) and one 2024Q2 filing; the
        # accession in the dataset has lost its dashes and leading zeros.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            final, meta = Path(tmp) / "final.csv", Path(tmp) / "meta.csv"
            pd.DataFrame({"accession": ["119312524000001", "119312524000001", "95017024000002", "119312524000003"],
                          "cal_q": ["2024Q1", "2024Q1", "2024Q1", "2024Q2"]}).to_csv(final, index=False)
            pd.DataFrame({"accessionNumber": ["0001193125-24-000001", "0000950170-24-000002", "0001193125-24-000003"],
                          "filingDate": ["2024-05-10", "2024-06-30", "2024-08-09"],
                          "form": ["10-Q", "10-K", "10-Q"]}).to_csv(meta, index=False)
            t = MP.filing_lags(final, meta).set_index("cal_q")
        q1 = t.loc["2024Q1"]
        self.assertEqual((q1["filings"], q1["with_filing_date"], q1["annual_reports"]), (2, 2, 1))
        self.assertEqual((q1["median_days"], q1["max_days"]), (65.5, 91))
        self.assertEqual((q1["within_45_days"], q1["within_90_days"]), (0.5, 0.5))
        self.assertEqual(str(q1["last_filing"]), "2024-06-30")                    # when 2024Q1 can be scored
        self.assertEqual(t["filed_before_quarter_end"].sum(), 0)


class TestFit(unittest.TestCase):

    def test_logit_matches_sklearn(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(1)
        X = rng.normal(size=(3000, 2))
        y = (rng.random(3000) < 1 / (1 + np.exp(-(-3 + X @ [1.0, -0.5])))).astype(float)
        r = LG.logit_cluster(X, y, np.arange(3000))
        sk = LogisticRegression(penalty=None, tol=1e-10, max_iter=10_000).fit(X, y)
        np.testing.assert_allclose(r["beta"], np.r_[sk.intercept_, sk.coef_[0]], atol=1e-5)

    def test_cluster_se_one_obs_per_cluster_is_robust(self):
        # One row per cluster: the clustered covariance is the heteroskedasticity-robust sandwich.
        rng = np.random.default_rng(2)
        n = 2000
        X = rng.normal(size=(n, 2))
        y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + X @ [1.0, -0.5])))).astype(float)
        r = LG.logit_cluster(X, y, np.arange(n))
        X1 = np.column_stack([np.ones(n), X])
        p = r["fitted"]
        bread = np.linalg.inv(X1.T @ (X1 * (p * (1 - p))[:, None]))
        robust = n / (n - 1) * bread @ (X1.T * (y - p) ** 2) @ X1 @ bread
        np.testing.assert_allclose(r["cov"], robust, rtol=1e-8)

    def test_clustering_widens_se_with_repeated_borrowers(self):
        rng = np.random.default_rng(3)
        g = np.repeat(np.arange(300), 10)
        xb = rng.normal(size=300)                                  # predictor and outcome constant within borrower
        x, y = np.repeat(xb, 10), np.repeat((rng.random(300) < 1 / (1 + np.exp(-xb))).astype(float), 10)
        se_cl = np.sqrt(LG.logit_cluster(x[:, None], y, g)["cov"][1, 1])
        se_iid = np.sqrt(LG.logit_cluster(x[:, None], y, np.arange(3000))["cov"][1, 1])
        self.assertGreater(se_cl, 2 * se_iid)

    def test_fixed_caps(self):
        d = pd.DataFrame({"price": [0.30, 0.85, 1.02], "d_price": [-0.45, -0.10, 0.30], "pik_switch": [1, 0, 0]})
        X = FR.design(d, ["price", "d_price", "pik_switch"])
        np.testing.assert_allclose(X[:, 0], [0.70, 0.85, 1.02])      # a mark below 70 counts as 70
        np.testing.assert_allclose(X[:, 1], [-0.20, -0.10, 0.20])    # a move of more than 20 points counts as 20
        X = FR.design(d, ["price", "d_price", "pik_switch"], caps={})  # the uncapped robustness check
        np.testing.assert_allclose(X[:, :2], d[["price", "d_price"]].to_numpy())

    def test_design_fills_missing_dispersion_with_zero(self):
        d = pd.DataFrame({"dispersion": [np.nan, 0.01, 0.5], "has_dispersion": [0, 1, 1]})
        X = FR.design(d, ["dispersion", "has_dispersion"])
        np.testing.assert_allclose(X[:, 0], [0.0, 0.01, 0.5])        # single lender: 0, indicator carries it

    @staticmethod
    def _frame(seed, n=6000, quarters=10):
        rng = np.random.default_rng(seed)
        i = np.arange(n)                                        # every borrower once in every quarter
        d = pd.DataFrame({"borrower": (i % (n // quarters)).astype(str), "qsort": i // (n // quarters) + 1,
                          "x": rng.normal(size=n), "has_dispersion": 0})
        d["y"] = (rng.random(n) < 1 / (1 + np.exp(-(-3 + d["x"])))).astype(float)
        d["cal_q"] = d["outcome_q"] = d["qsort"].astype(str)
        return d

    def test_no_look_ahead(self):
        # Test-quarter outcomes must not change the fitted model: hide them and the scores are identical.
        d = self._frame(4)
        d.loc[d["qsort"] >= 9, "y"] = 1.0                       # outcomes the model must not see
        seen = EV.split_predictions(d, ["x"], 7, [9, 10])
        hidden = d.assign(y=d["y"].where(d["qsort"] <= 7))      # quarter 8's window is still open at 9
        unseen = EV.split_predictions(hidden, ["x"], 7, [9, 10])
        np.testing.assert_allclose(seen["pred"].to_numpy(), unseen["pred"].to_numpy())
        self.assertEqual(set(seen["qsort"]), {9, 10})

    def test_training_windows_must_close_before_the_test(self):
        # Quarter 8's two-quarter window ends at 10: fitting on it to score 9 would look ahead.
        with self.assertRaises(ValueError):
            EV.split_predictions(self._frame(4), ["x"], 8, [9, 10])

    def test_fit_is_deterministic(self):
        d = self._frame(6)
        tr, te = d[d["qsort"] <= 7], d[d["qsort"] > 7]
        np.testing.assert_array_equal(LG.fit_predict(tr, te, ["x"]), LG.fit_predict(tr, te, ["x"]))

    def test_rolling_folds_use_only_earlier_quarters(self):
        p = EV.rolling_predictions(self._frame(5), ["x"])
        self.assertTrue((p["train_last_q"] <= p["qsort"] - CFG.HORIZON).all())     # windows closed by q
        self.assertEqual(p["qsort"].min(), CFG.HORIZON + CFG.MIN_TRAIN_QUARTERS)

    def test_equal_weight_score(self):
        # Lower mark, larger fall and a PIK switch each raise the rank; ranks are taken within the quarter.
        d = pd.DataFrame({"qsort": [1, 1, 1, 2, 2], "price": [0.6, 0.9, 1.0, 0.99, 1.0],
                          "d_price": [-0.10, -0.02, 0.0, 0.0, -0.01], "pik_switch": [1, 0, 0, 0, 0]})
        s = RK.equal_weight_score(d, ["price", "d_price", "pik_switch"])
        self.assertAlmostEqual(s[0], 1.0)                          # riskiest on all three signs
        self.assertGreater(s[1], s[2])
        self.assertAlmostEqual(s[3], s[4])                         # one sign each, PIK tied

    @staticmethod
    def _scores():
        """Two quarters of 20 borrowers; the 4 events sit near the top of each quarter's ranking."""
        return pd.DataFrame({"qsort": [1] * 20 + [2] * 20, "cal_q": ["1"] * 20 + ["2"] * 20,
                             "borrower": [str(i) for i in range(40)], "pred": list(np.linspace(0.5, 0.01, 20)) * 2,
                             "y": ([1, 1] + [0] * 18) + ([1, 0, 1] + [0] * 17)})

    def test_decile_table(self):
        p = self._scores()
        t = EV.decile_table(p).set_index("decile")
        self.assertTrue((t["n"] == 4).all())
        self.assertEqual(t.loc[10, "events"], 3)                   # top two of each quarter
        self.assertEqual(t.loc[9, "events"], 1)
        self.assertAlmostEqual(t.loc[10, "share_of_events"], 0.75)
        top = p.sort_values("pred")["pred"].iloc[-4:]
        self.assertAlmostEqual(t.loc[10, "predicted_rate"], top.mean())
        self.assertAlmostEqual(t.loc[10, "realized_to_predicted"], 3 / top.sum())
        lo, hi = t.loc[10, ["rate_ci_low", "rate_ci_high"]]
        self.assertTrue(0 <= lo < 0.75 < hi <= 1)
        self.assertNotIn("realized_to_predicted", EV.decile_table(p, probabilities=False))   # ranks, not probabilities

    def test_realized_to_predicted(self):
        # 4 events against 10.2 predicted: the model predicts too many, and says so with an interval.
        p = self._scores()
        perf = EV.performance(p)
        self.assertAlmostEqual(perf["realized_to_predicted"], 4 / p["pred"].sum())
        self.assertTrue(perf["ratio_lo"] <= perf["realized_to_predicted"] <= perf["ratio_hi"] < 1)
        self.assertTrue(np.isnan(EV.performance(p, probabilities=False)["realized_to_predicted"]))

    def test_rate_interval_counts_a_borrower_once(self):
        # The same 40 rows and 6 events, first as 40 borrowers, then as 20 borrowers seen twice
        # with each event repeated: the repeated borrowers carry less information, so a wider interval.
        y = np.array([1, 1, 1, 1, 1, 1] + [0] * 34)
        lo1, hi1 = EV.ratio_ci(y, np.ones(40), np.arange(40))
        lo2, hi2 = EV.ratio_ci(y, np.ones(40), np.arange(40) // 2)
        self.assertTrue(lo1 <= y.mean() <= hi1)
        self.assertGreater(hi2 - lo2, hi1 - lo1)
        self.assertTrue(np.isnan(EV.ratio_ci([0, 0], [0, 0], [1, 2])[0]))       # no denominator

    def test_auc_gain_of_a_model_over_itself_is_zero(self):
        d = self._frame(7)
        p = EV.split_predictions(d, ["x"], 7, [9, 10])
        self.assertEqual(EV.auc_gain(p, p), (0.0, 0.0, 0.0))
        auc, lo, hi = EV.auc_ci(p)
        self.assertTrue(lo < auc < hi)

    def test_ranks_by_outcome_scores_every_borrower(self):
        # Every accruing borrower of the test quarter is ranked, labelled or not, and falls in one
        # group; events are split by kind; refinanced is listed though empty (no exit dropped).
        import analysis as A
        rng = np.random.default_rng(1)
        n_b, quarters = 80, 8
        i = np.arange(n_b * quarters)
        m = pd.DataFrame({"borrower": (i % n_b).astype(str), "qsort": i // n_b + 1, "na_risk_set": True,
                          "window_complete": True, "has_dispersion": 1, "exited": False, "exit_class": ""})
        m["cal_q"] = m["outcome_q"] = m["qsort"].astype(str)
        for c in ["price", "d_price", "max_drawdown", "dispersion"]:
            m[c] = rng.normal(size=len(m))
        m["pik_switch"] = (rng.random(len(m)) < 0.1).astype(float)
        m[CFG.OUTCOME] = (rng.random(len(m)) < 0.3).astype(float)
        m["flag_in_window"] = (m[CFG.OUTCOME] == 1) & (rng.random(len(m)) < 0.5)
        gone = rng.random(len(m)) < 0.1
        m.loc[gone, [CFG.OUTCOME, "exited", "exit_class", "flag_in_window"]] = [np.nan, True, "impaired", False]
        t = A.ranks_by_outcome(m, 6, [8]).set_index("what_happened")
        self.assertEqual(t.loc["all", "borrower_quarters"], n_b)
        self.assertEqual(t.drop("all")["borrower_quarters"].sum(), n_b)
        self.assertEqual(t.loc["left the books: refinanced", "borrower_quarters"], 0)
        self.assertEqual(t.loc["all", "in_riskiest_tenth"], n_b // 10)
        events = t.loc[["credit event: non-accrual", "credit event: concession only"], "borrower_quarters"].sum()
        last = m[m["qsort"] == 8]
        self.assertEqual(events, (last[CFG.OUTCOME] == 1).sum())
        self.assertAlmostEqual(t.loc["all", "labelled_share"], last[CFG.OUTCOME].notna().mean())
        self.assertTrue(np.isnan(t.loc["no credit event", "roc_auc_vs_no_event"]))

    def test_pr_auc_and_recall_at_budget(self):
        # 4 events, 3 of them in the top 10% (2 of 20 per quarter) and all 4 in the top 20%.
        p = self._scores()
        self.assertAlmostEqual(EV.recall_at(p, 0.10)[0], 0.75)
        self.assertAlmostEqual(EV.recall_at(p, 0.20)[0], 1.0)
        ap, lo, hi = EV.pr_auc_ci(p)
        self.assertTrue(p["y"].mean() < ap <= 1.0)              # better than the event rate (random)
        self.assertTrue(lo <= ap <= hi)


if __name__ == "__main__":
    unittest.main()
