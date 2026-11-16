"""
Tests for the early-warning scorecard (src/9_prediction): the borrower-quarter panel (same-loan price
changes, non-accrual entries within two quarters, unlabelled exits typed by maturity, sample rules)
and the estimation helpers (clustered standard errors, no look-ahead, equal-weighted ranking,
deciles).
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
import frames as FR
import models.logit as LG
import models.ranking as RK
import evaluation as EV
import features as FT
import panel as MP

PAR = 10_000_000.0
Q = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]


def tranche(chain, cik, borrower, q, price, link_rule, toggle=False, par=PAR, v1=np.nan, family=None, lien="1L",
            maturity="2029-06-30"):
    return {"chain_id": chain, "cik": cik, "family": family or f"F{cik}", "borrower_key": borrower, "cal_q": q,
            "qsort": MC.q_sort(q), "FV": price * par, "PAR": par, "COST": par, "FV_px": price * par, "PAR_px": par,
            "lien_group": lien, "link_rule": link_rule, "is_distress_amendment": toggle, "pik_share": 0.0,
            "V1_std": v1, "V1_range": v1, "sector": "Software", "maturity": pd.Timestamp(maturity)}


def toy_panel():
    """ACME: two BDCs' loans (chains 1 and 2) marked down over 2024; BDC 1 flags it non-accrual in
    2024Q4; BDC 2 lends a new loan at 70 in 2024Q4 (chain 3). Four borrowers held 2024Q1-Q2 that
    then leave: BETA at par long before maturity, DELTA at par at maturity, EPSILON marked at 60,
    ZETA marked at 90. GAMMA CLO 1: a CLO debt tranche, held all year."""
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
    return MP.build_model_panel(tr, pl.DataFrame(rows, schema_overrides={"non_accrual": pl.Float64}))


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

    def test_exits_have_no_label_and_are_typed_by_maturity_not_mark(self):
        # The outcome comes from non-accrual flags only: a borrower that leaves before the window
        # closes gets no label, at par (BETA, DELTA) or deeply marked down (EPSILON at 60). The
        # exit type comes from the scheduled maturity: DELTA's loan was due in 2024Q3.
        m = self.m
        for q in ["2024Q1", "2024Q2"]:                               # gone in 2024Q3, inside both windows
            for name in ["BETA", "DELTA", "EPSILON", "ZETA"]:
                self.assertTrue(m.loc[(name, q), "exited"])
                self.assertTrue(np.isnan(m.loc[(name, q), "y_na"]))
        self.assertEqual(m.loc[("DELTA", "2024Q2"), "exit_type"], "at scheduled maturity")
        for name in ["BETA", "EPSILON", "ZETA"]:
            self.assertEqual(m.loc[(name, "2024Q2"), "exit_type"], "before scheduled maturity")
        self.assertFalse(m.loc[("ACME", "2024Q3"), "exited"])                       # window runs past the data

    def test_event_counts_cover_every_borrower(self):
        m = self.m.reset_index()
        c = MP.event_counts(m).set_index("cal_q")
        parts = ["entered_non_accrual", "still_accruing", "exited", "status_unknown"]
        exits = ["exit_at_scheduled_maturity", "exit_before_scheduled_maturity", "exit_maturity_not_reported"]
        self.assertTrue((c[parts].sum(axis=1) == c["at_risk"]).all())
        self.assertTrue((c[exits].sum(axis=1) == c["exited"]).all())
        q2 = c.loc["2024Q2"]
        self.assertEqual([q2[k] for k in parts], [1, 0, 4, 0])                      # ACME enters, four exits
        self.assertEqual([q2[k] for k in exits], [1, 3, 0])                         # DELTA at maturity
        self.assertEqual((q2["labelled"], q2["events"]), (1, 1))                    # a low-marked exit is not an event
        self.assertNotIn("2024Q3", c.index)                                         # window incomplete
        x = MP.exit_table(m).set_index("exit_type")
        self.assertEqual(x.loc["all", "exits"], 4)
        self.assertEqual(x.loc["before scheduled maturity", "mark at t below 80"], 1)   # EPSILON: context only

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

    def test_clo_vehicle_out_of_sample(self):
        g = self.m.loc["GAMMA CLO 1"]
        self.assertEqual(g.loc["2024Q2", "clo_share"], 1.0)
        self.assertFalse(g["in_sample"].any())


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

    def test_decile_table(self):
        # Two quarters of 20 borrowers; the events sit at the top of each quarter's ranking.
        p = pd.DataFrame({"qsort": [1] * 20 + [2] * 20, "borrower": [str(i) for i in range(40)],
                          "pred": list(np.linspace(0.5, 0.01, 20)) * 2, "y": ([1, 1] + [0] * 18) + ([1, 0, 1] + [0] * 17)})
        t = EV.decile_table(p).set_index("decile")
        self.assertTrue((t["n"] == 4).all())
        self.assertEqual(t.loc[10, "events"], 3)                   # top two of each quarter
        self.assertEqual(t.loc[9, "events"], 1)
        self.assertAlmostEqual(t.loc[10, "share_of_events"], 0.75)
        self.assertAlmostEqual(t.loc[10, "predicted_rate"], p.sort_values("pred")["pred"].iloc[-4:].mean())
        lo, hi = EV.wilson(3, 4)
        self.assertAlmostEqual(t.loc[10, "rate_ci_low"], lo)
        self.assertTrue(lo < 0.75 < hi)

    def test_auc_gain_of_a_model_over_itself_is_zero(self):
        d = self._frame(7)
        p = EV.split_predictions(d, ["x"], 7, [9, 10])
        self.assertEqual(EV.auc_gain(p, p), (0.0, 0.0, 0.0))
        auc, lo, hi = EV.auc_ci(p)
        self.assertTrue(lo < auc < hi)

    def test_pr_auc_and_recall_at_budget(self):
        # Same toy as the decile table: 4 events, 3 of them in the top 10% (2 of 20 per quarter)
        # and all 4 in the top 20% (4 of 20 per quarter).
        p = pd.DataFrame({"qsort": [1] * 20 + [2] * 20, "borrower": [str(i) for i in range(40)],
                          "pred": list(np.linspace(0.5, 0.01, 20)) * 2, "y": ([1, 1] + [0] * 18) + ([1, 0, 1] + [0] * 17)})
        self.assertAlmostEqual(EV.recall_at(p, 0.10)[0], 0.75)
        self.assertAlmostEqual(EV.recall_at(p, 0.20)[0], 1.0)
        ap, lo, hi = EV.pr_auc_ci(p)
        self.assertTrue(p["y"].mean() < ap <= 1.0)              # better than the event rate (random)
        self.assertTrue(lo <= ap <= hi)


if __name__ == "__main__":
    unittest.main()
