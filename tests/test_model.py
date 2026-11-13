"""
Tests for the early-warning models (src/9_model): the borrower-quarter panel (same-loan price
changes, rolling max drawdown, non-accrual entries, sample rules) and the estimation helpers
(clustered standard errors, threshold, no look-ahead).
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


def tranche(chain, cik, borrower, q, price, link_rule, toggle=False, par=PAR, v1=np.nan, family=None, lien="1L"):
    return {"chain_id": chain, "cik": cik, "family": family or f"F{cik}", "borrower_key": borrower, "cal_q": q,
            "qsort": MP.q_sort(q), "FV": price * par, "PAR": par, "COST": par, "FV_px": price * par, "PAR_px": par,
            "lien_group": lien, "link_rule": link_rule, "is_distress_amendment": toggle, "pik_share": 0.0,
            "V1_std": v1, "V1_range": v1, "sector": "Software"}


def toy_panel():
    """ACME: two BDCs' loans (chains 1 and 2) marked down over 2024; BDC 1 flags it non-accrual in
    2024Q4; BDC 2 lends a new loan at 70 in 2024Q4 (chain 3). BETA: held 2024Q1-Q2, then exits.
    GAMMA CLO 1: a CLO debt tranche, held all year."""
    p1 = [1.00, 0.98, 0.90, 0.85]
    p2 = [1.00, 1.00, 1.00, 0.95]
    t = []
    for i, q in enumerate(Q):
        t.append(tranche(1, 1, "ACME", q, p1[i], 0 if i == 0 else 1, toggle=(q == "2024Q3")))
        t.append(tranche(2, 2, "ACME", q, p2[i], 0 if i == 0 else 1))
    t.append(tranche(3, 2, "ACME", "2024Q4", 0.70, 0))
    t += [tranche(4, 1, "BETA", "2024Q1", 1.0, 0), tranche(4, 1, "BETA", "2024Q2", 1.0, 1)]
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
                     "PAR": r["PAR"], "FV": r["FV"], "non_accrual": na})
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
        self.assertAlmostEqual(a.loc["2024Q3", "price_change"], -0.05)
        # the outline's borrower-level difference picks up the new loan
        self.assertAlmostEqual(a.loc["2024Q3", "price_change_raw"], (0.85 + 0.95 + 0.70) / 3 - 0.95)

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

    def test_exit_leaves_outcomes_missing(self):
        b = self.m.loc["BETA"]
        self.assertTrue(np.isnan(b.loc["2024Q2", "y_na"]))
        self.assertTrue(np.isnan(b.loc["2024Q2", "val_ret"]))
        self.assertEqual(b.loc["2024Q1", "y_na"], 0)

    def test_pik_switch_and_sample(self):
        a = self.m.loc["ACME"]
        self.assertTrue(np.isnan(a.loc["2024Q1", "pik_switch"]))     # first quarter: no switch defined
        self.assertEqual(a.loc["2024Q3", "pik_switch"], 1)
        self.assertFalse(a.loc["2024Q1", "in_sample"])
        self.assertTrue(a.loc["2024Q3", "in_sample"])
        self.assertEqual(a.loc["2024Q3", "has_dispersion"], 0)

    def test_horizon_outcomes(self):
        a = self.m.loc["ACME"]                                       # BDC 1 flags it in 2024Q4
        self.assertEqual(a.loc["2024Q1", "y_na_h2"], 0)              # accruing through 2024Q3
        self.assertEqual(a.loc["2024Q2", "y_na_h2"], 1)              # entry two quarters later
        self.assertEqual(a.loc["2024Q2", "first_entry_k"], 2)
        # 2024Q3's two-quarter window runs past the data: unlabelled even though the entry is seen
        self.assertTrue(np.isnan(a.loc["2024Q3", "y_na_h2"]))
        self.assertTrue(a["y_na_h4"].isna().all())                   # no four-quarter window fits in 2024

    def test_clo_vehicle_out_of_sample(self):
        g = self.m.loc["GAMMA CLO 1"]
        self.assertEqual(g.loc["2024Q2", "clo_share"], 1.0)
        self.assertFalse(g["in_sample"].any())

    def test_val_ret(self):
        a = self.m.loc["ACME"]
        self.assertAlmostEqual(a.loc["2024Q2", "val_ret"], (0.90 + 1.00) / (0.98 + 1.00) - 1)


class TestFit(unittest.TestCase):

    def test_logit_matches_sklearn(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(1)
        X = rng.normal(size=(3000, 2))
        y = (rng.random(3000) < 1 / (1 + np.exp(-(-3 + X @ [1.0, -0.5])))).astype(float)
        r = F.logit_cluster(X, y, np.arange(3000))
        sk = LogisticRegression(penalty=None, tol=1e-10, max_iter=10_000).fit(X, y)
        np.testing.assert_allclose(r["beta"], np.r_[sk.intercept_, sk.coef_[0]], atol=1e-5)

    def test_cluster_se_one_obs_per_cluster_is_hc1(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(500, 2))
        y = X @ [1.0, 2.0] + rng.normal(size=500) * (1 + np.abs(X[:, 0]))
        r = F.ols_cluster(X, y, np.arange(500))
        X1 = np.column_stack([np.ones(500), X])
        e = y - X1 @ r["beta"]
        bread = np.linalg.inv(X1.T @ X1)
        hc1 = 500 / (500 - 3) * bread @ (X1.T * e ** 2) @ X1 @ bread
        np.testing.assert_allclose(r["cov"], hc1, rtol=1e-8)

    def test_clustering_widens_se_with_repeated_borrowers(self):
        rng = np.random.default_rng(3)
        g = np.repeat(np.arange(100), 10)
        x = np.repeat(rng.normal(size=100), 10)[:, None]          # predictor and error constant within borrower
        y = 0.5 * x[:, 0] + np.repeat(rng.normal(size=100), 10) + 0.1 * rng.normal(size=1000)
        se_cl = np.sqrt(F.ols_cluster(x, y, g)["cov"][1, 1])
        se_iid = np.sqrt(F.ols_cluster(x, y, np.arange(1000))["cov"][1, 1])
        self.assertGreater(se_cl, 2 * se_iid)

    def test_recall_threshold(self):
        y = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        p = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.5, 0.2, 0.05])
        thr = F.recall_threshold(y, p, 0.8)
        self.assertEqual(thr, 0.6)
        self.assertEqual(((p >= thr) & (y == 1)).sum(), 4)

    def test_fixed_caps(self):
        d = pd.DataFrame({"price": [0.30, 0.85, 1.02], "max_drawdown": [0.45, 0.10, 0.0], "pik_switch": [1, 0, 0]})
        cols = ["price", "max_drawdown", "pik_switch"]
        X = F.design(d, cols)
        np.testing.assert_allclose(X[:, 0], [0.70, 0.85, 1.02])      # a mark below 70 counts as 70
        np.testing.assert_allclose(X[:, 1], [0.20, 0.10, 0.0])       # a fall above 20 points counts as 20

    def test_design_fills_missing_dispersion_with_zero(self):
        d = pd.DataFrame({"dispersion": [np.nan, 0.01, 0.5], "has_dispersion": [0, 1, 1]})
        X = F.design(d, ["dispersion", "has_dispersion"])
        np.testing.assert_allclose(X[:, 0], [0.0, 0.01, 0.5])        # single lender: 0, indicator carries it

    def test_no_look_ahead(self):
        # Test-quarter outcomes must not change the fitted model: hide them and the scores are identical.
        rng = np.random.default_rng(4)
        n = 4000
        d = pd.DataFrame({"borrower": rng.integers(0, 800, n).astype(str), "qsort": rng.integers(1, 9, n),
                          "x": rng.normal(size=n), "has_dispersion": 0})
        d["y"] = (rng.random(n) < 1 / (1 + np.exp(-(-3 + d["x"])))).astype(float)
        d["cal_q"] = d["outcome_q"] = d["qsort"].astype(str)
        d.loc[d["qsort"] == 8, "y"] = 1.0                       # an outcome the model must not see
        _, seen = F.evaluate("binary", "logit", d, ["x"], 7, 8)
        hidden = d.assign(y=d["y"].where(d["qsort"] < 8))
        _, unseen = F.evaluate("binary", "logit", hidden, ["x"], 7, 8)
        np.testing.assert_allclose(seen["pred"].to_numpy(), unseen["pred"].to_numpy())

    def test_models_are_deterministic(self):
        # Every model gives the same scores when refitted (the lasso's saga solver shuffles the data).
        rng = np.random.default_rng(6)
        n = 3000
        d = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "has_dispersion": 0})
        d["y"] = (rng.random(n) < 1 / (1 + np.exp(-(-2 + d["x1"])))).astype(float)
        tr, te = d.iloc[:2000], d.iloc[2000:]
        for name, params in [("lasso", {"C": 0.1}), ("ridge", {"C": 0.1}), ("gbt", {"max_depth": 2, "learning_rate": 0.1}),
                             ("mlp", {"alpha": 0.1})]:
            p1, _, _ = F.fit_predict("binary", name, params, tr, te, ["x1", "x2"])
            p2, _, _ = F.fit_predict("binary", name, params, tr, te, ["x1", "x2"])
            np.testing.assert_array_equal(p1, p2, err_msg=name)

    def test_rolling_folds_use_only_observed_outcomes(self):
        rng = np.random.default_rng(5)
        n = 6000
        d = pd.DataFrame({"borrower": rng.integers(0, 900, n).astype(str), "qsort": rng.integers(1, 11, n),
                          "x": rng.normal(size=n), "has_dispersion": 0})
        d["y"] = (rng.random(n) < 1 / (1 + np.exp(-(-3 + d["x"])))).astype(float)
        d["cal_q"] = d["outcome_q"] = d["qsort"].astype(str)
        for h in (1, 2, 4):
            p = F.rolling_predictions(d, ["x"], "binary", h)
            self.assertTrue((p["train_last_q"] <= p["qsort"] - h).all())        # labels filed by the test quarter
            self.assertEqual(p["qsort"].min(), 1 + h + F.ROLL_MIN_TRAIN_QUARTERS - 1)

    def test_review_capture(self):
        p = pd.DataFrame({"qsort": [1] * 10 + [2] * 10, "borrower": [str(i) for i in range(20)],
                          "pred": list(np.linspace(1, 0.1, 10)) * 2,
                          "y": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0] + [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]})
        c = F.review_capture(p, shares=(0.1, 0.2)).set_index("share_flagged")
        self.assertEqual(c.loc[0.1, "caught"], 1)                  # top 1 per quarter: the first quarter's entry
        self.assertEqual(c.loc[0.2, "caught"], 2)                  # top 2: the second quarter's too
        self.assertAlmostEqual(c.loc[0.2, "precision"], 0.5)
        self.assertAlmostEqual(c.loc[0.2, "capture"], 0.5)


if __name__ == "__main__":
    unittest.main()
