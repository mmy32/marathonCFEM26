"""
Tests for the early-warning scorecard (src/9_model): the borrower-quarter panel (same-loan price
changes, non-accrual entries, exit classification, sample rules) and the estimation helpers
(clustered standard errors, no look-ahead, equal-weighted ranking, deciles).
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

import model_fit as F
import model_panel as MP

PAR = 10_000_000.0
Q = ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]


def tranche(chain, cik, borrower, q, price, link_rule, toggle=False, par=PAR, v1=np.nan, family=None, lien="1L",
            maturity="2029-06-30"):
    return {"chain_id": chain, "cik": cik, "family": family or f"F{cik}", "borrower_key": borrower, "cal_q": q,
            "qsort": MP.q_sort(q), "FV": price * par, "PAR": par, "COST": par, "FV_px": price * par, "PAR_px": par,
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

    def test_rolling_max_drawdown(self):
        a = self.m.loc["ACME"]
        self.assertTrue(np.isnan(a.loc["2024Q1", "max_drawdown"]))     # no earlier quarter
        self.assertAlmostEqual(a.loc["2024Q2", "max_drawdown"], 0.01)
        self.assertAlmostEqual(a.loc["2024Q3", "max_drawdown"], 0.05)
        self.assertAlmostEqual(a.loc["2024Q4", "max_drawdown"], 0.10)

    def test_drawdown_window_is_four_quarters(self):
        d = pd.DataFrame({"borrower": "X", "qsort": range(1, 8),
                          "d_price": [np.nan, -0.10, 0.0, 0.0, 0.0, 0.0, 0.02]})
        dd = MP.rolling_max_drawdown(d).set_index("qsort")["dd"]
        self.assertAlmostEqual(dd[5], 0.10)        # quarters 1-5 include the fall in quarter 2
        self.assertAlmostEqual(dd[6], 0.0)         # quarters 2-6: the fall happened before the window
        self.assertAlmostEqual(dd[7], 0.0)

    def test_nonaccrual_entry(self):
        a = self.m.loc["ACME"]
        self.assertEqual(a.loc["2024Q1", "y_na"], 0)
        self.assertEqual(a.loc["2024Q3", "y_na"], 1)
        self.assertEqual(a.loc["2024Q3", "y_na_same_lender"], 1)     # BDC 1 reported it accruing in 2024Q3
        self.assertEqual(a.loc["2024Q4", "na_status"], 1)
        self.assertTrue(np.isnan(a.loc["2024Q4", "y_na"]))           # no next quarter in the data
        self.assertFalse(a.loc["2024Q4", "na_risk_set"])             # already on non-accrual

    def test_exits_are_classified(self):
        m = self.m
        self.assertEqual(m.loc[("BETA", "2024Q2"), "exit_type"], "refinanced")      # left at par before maturity
        self.assertEqual(m.loc[("DELTA", "2024Q2"), "exit_type"], "repaid")         # left at par at maturity
        self.assertEqual(m.loc[("EPSILON", "2024Q2"), "exit_type"], "impaired")     # left marked at 60
        self.assertEqual(m.loc[("ZETA", "2024Q2"), "exit_type"], "unclassified")    # left marked at 90
        self.assertTrue(pd.isna(m.loc[("BETA", "2024Q1"), "exit_type"]))            # still held next quarter
        self.assertTrue(pd.isna(m.loc[("ACME", "2024Q4"), "exit_type"]))            # last quarter of the data

    def test_exit_outcomes(self):
        m = self.m
        self.assertEqual(m.loc[("BETA", "2024Q1"), "y_na"], 0)
        self.assertEqual(m.loc[("BETA", "2024Q2"), "y_na"], 0)                      # paid off: not an event
        self.assertEqual(m.loc[("DELTA", "2024Q2"), "y_na"], 0)
        self.assertEqual(m.loc[("EPSILON", "2024Q2"), "y_na"], 1)                   # impaired: an event ...
        self.assertTrue(np.isnan(m.loc[("EPSILON", "2024Q2"), "y_na_impaired_unlabelled"]))   # ... no label here
        self.assertTrue(np.isnan(m.loc[("ZETA", "2024Q2"), "y_na"]))                # unclassified: no label

    def test_event_counts_cover_every_borrower(self):
        c = MP.event_counts(self.m.reset_index()).set_index("cal_q")
        parts = ["entered_non_accrual", "still_accruing", "exit_repaid", "exit_refinanced", "exit_impaired", "exit_unclassified",
                 "status_unknown"]
        self.assertTrue((c[parts].sum(axis=1) == c["at_risk"]).all())
        q2 = c.loc["2024Q2"]
        self.assertEqual([q2[k] for k in parts], [0, 1, 1, 1, 1, 1, 0])             # ACME accruing, four exits
        self.assertEqual(q2["labelled"], 4)                                         # ACME, repaid, refinanced, impaired
        self.assertEqual(q2["events"], 1)                                           # the impaired exit
        self.assertEqual(c.loc["2024Q3", "events"], 1)

    def test_mark_uses_cost_where_par_is_missing(self):
        rows = pl.DataFrame({"cal_q": ["2024Q1"] * 3, "borrower_canonical": ["ACME"] * 3,
                             "instrument_seniority": ["SENIOR_SECURED_1L", "OTHER_UNKNOWN", "SENIOR_SECURED_1L"],
                             "PAR": [None, None, 5.0], "FV": [9.0, 50.0, 5.0], "COST": [10.0, 40.0, 5.0]})
        n = MP.no_par_debt(rows).iloc[0]                                            # only the first row: debt, no par
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
        r = F.logit_cluster(X, y, np.arange(3000))
        sk = LogisticRegression(penalty=None, tol=1e-10, max_iter=10_000).fit(X, y)
        np.testing.assert_allclose(r["beta"], np.r_[sk.intercept_, sk.coef_[0]], atol=1e-5)

    def test_cluster_se_one_obs_per_cluster_is_robust(self):
        # One row per cluster: the clustered covariance is the heteroskedasticity-robust sandwich.
        rng = np.random.default_rng(2)
        n = 2000
        X = rng.normal(size=(n, 2))
        y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + X @ [1.0, -0.5])))).astype(float)
        r = F.logit_cluster(X, y, np.arange(n))
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
        se_cl = np.sqrt(F.logit_cluster(x[:, None], y, g)["cov"][1, 1])
        se_iid = np.sqrt(F.logit_cluster(x[:, None], y, np.arange(3000))["cov"][1, 1])
        self.assertGreater(se_cl, 2 * se_iid)

    def test_fixed_caps(self):
        d = pd.DataFrame({"price": [0.30, 0.85, 1.02], "d_price": [-0.45, -0.10, 0.30], "pik_switch": [1, 0, 0]})
        X = F.design(d, F.SCORECARD)
        np.testing.assert_allclose(X[:, 0], [0.70, 0.85, 1.02])      # a mark below 70 counts as 70
        np.testing.assert_allclose(X[:, 1], [-0.20, -0.10, 0.20])    # a move of more than 20 points counts as 20

    def test_design_fills_missing_dispersion_with_zero(self):
        d = pd.DataFrame({"dispersion": [np.nan, 0.01, 0.5], "has_dispersion": [0, 1, 1]})
        X = F.design(d, ["dispersion", "has_dispersion"])
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
        seen = F.split_predictions(d, ["x"], 8, [9, 10])
        hidden = d.assign(y=d["y"].where(d["qsort"] < 9))
        unseen = F.split_predictions(hidden, ["x"], 8, [9, 10])
        np.testing.assert_allclose(seen["pred"].to_numpy(), unseen["pred"].to_numpy())
        self.assertEqual(set(seen["qsort"]), {9, 10})

    def test_fit_is_deterministic(self):
        d = self._frame(6)
        tr, te = d[d["qsort"] <= 7], d[d["qsort"] > 7]
        np.testing.assert_array_equal(F.fit_predict(tr, te, ["x"]), F.fit_predict(tr, te, ["x"]))

    def test_rolling_folds_use_only_earlier_quarters(self):
        p = F.rolling_predictions(self._frame(5), ["x"])
        self.assertTrue((p["train_last_q"] < p["qsort"]).all())
        self.assertEqual(p["qsort"].min(), 1 + F.MIN_TRAIN_QUARTERS)

    def test_equal_weight_score(self):
        # Lower mark, larger fall and a PIK switch each raise the rank; ranks are taken within the quarter.
        d = pd.DataFrame({"qsort": [1, 1, 1, 2, 2], "price": [0.6, 0.9, 1.0, 0.99, 1.0],
                          "d_price": [-0.10, -0.02, 0.0, 0.0, -0.01], "pik_switch": [1, 0, 0, 0, 0]})
        s = F.equal_weight_score(d, F.SCORECARD)
        self.assertAlmostEqual(s[0], 1.0)                          # riskiest on all three signs
        self.assertGreater(s[1], s[2])
        self.assertAlmostEqual(s[3], s[4])                         # one sign each, PIK tied

    def test_decile_table(self):
        # Two quarters of 20 borrowers; the events sit at the top of each quarter's ranking.
        p = pd.DataFrame({"qsort": [1] * 20 + [2] * 20, "borrower": [str(i) for i in range(40)],
                          "pred": list(np.linspace(0.5, 0.01, 20)) * 2, "y": ([1, 1] + [0] * 18) + ([1, 0, 1] + [0] * 17)})
        t = F.decile_table(p).set_index("decile")
        self.assertTrue((t["n"] == 4).all())
        self.assertEqual(t.loc[10, "events"], 3)                   # top two of each quarter
        self.assertEqual(t.loc[9, "events"], 1)
        self.assertAlmostEqual(t.loc[10, "share_of_events"], 0.75)
        self.assertAlmostEqual(t.loc[10, "predicted_rate"], p.sort_values("pred")["pred"].iloc[-4:].mean())
        lo, hi = F.wilson(3, 4)
        self.assertAlmostEqual(t.loc[10, "rate_ci_low"], lo)
        self.assertTrue(lo < 0.75 < hi)

    def test_auc_gain_of_a_model_over_itself_is_zero(self):
        d = self._frame(7)
        p = F.split_predictions(d, ["x"], 8, [9, 10])
        self.assertEqual(F.auc_gain(p, p), (0.0, 0.0, 0.0))
        auc, lo, hi = F.auc_ci(p)
        self.assertTrue(lo < auc < hi)


if __name__ == "__main__":
    unittest.main()
