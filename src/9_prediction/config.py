"""Every tunable constant of the early-warning model (src/9_prediction) in one place: the outcome window, the
sample rules, the definitions of the signs, the time split, and the settings of the tests. Nothing
here imports from the rest of the model."""
from __future__ import annotations

# ---- outcome and sample
OUTCOME = "y_event"         # the one classification outcome: an accruing borrower has a credit event at t + 1 or t + 2
                            # (non-accrual, or a lender's concession: PIK switch, debt-for-equity, write-down)
HORIZON = 2                 # the outcome window: a credit event at t + 1 or t + 2
MIN_PRIOR_FV = 1_000_000    # sample: the borrower's debt fair value at t is at least $1m
MIN_PRICED_SHARE = 0.5      # sample: priceable par is at least half of the borrower's funded par at t
MAX_CLO_SHARE = 0.5         # sample: CLO / structured tranches under half of par (CLO vehicles are not operating borrowers)

# ---- definitions of the signs
DD_WINDOW = 4               # rolling max drawdown over the preceding four quarters
PIK_LOOKBACK = 3            # past-year PIK switch: a switch in any of the three quarters before t (with t, a year)
LOW_MARK = 0.90             # PIK-switch hypotheses: mark below 90% of par ...
MARKDOWN = 0.05             # ... or marked down at least five points in the quarter
JUNIOR_LIENS = {"2L", "SUB", "UNSEC"}

# ---- time split. Signs at t are scored against credit events at t + 1 or t + 2. Three stages, in time
# order; a stage's training windows all close before the next stage's first signal quarter:
TRAIN_END = "2025Q2"        # development: signs up to 2025Q2 (windows close by 2025Q4) choose the signs
TEST_QS = ["2025Q4"]        # test: one fit on the development quarters, scored on one quarter of
                            # signals, so it looks two quarters ahead like every other stage
                            # (events in 2026Q1-2026Q2, the first half of 2026). Was 2025Q3-Q4 until
                            # 2026-11-16 and 2025Q3 (with development up to 2025Q1) until 2026-11-18;
                            # the rule keeps the same signs either way.
SCORE_Q = "2026Q2"          # scorecard quarter (window 2026Q3-2026Q4)
RESERVED_Q = "2026Q3"       # reserved: nothing is fitted or chosen on it. Once filed, signals from
                            # 2026Q1 (window 2026Q2-2026Q3) are the blind test: the only test no
                            # choice has seen (the 2025Q4 test was seen in the rolling check before
                            # this split was chosen). Change nothing after running it

# ---- fitting
HALF_LIFE = None            # recency weighting of the training rows: a quarter's rows weigh 0.5 ** (age / HALF_LIFE),
                            # age in quarters before the last training quarter; None = every quarter the same.
                            # Half-lives of 8, 4 and 2 were tested on the development quarters on 2026-11-23
                            # under the sign rule (+MIN_GAIN ROC-AUC): none changed the ranking (best +0.0001),
                            # so none is used; they only lift calibration a little (0.89 -> 0.91 at 2)

# ---- tests and intervals
MIN_TRAIN_QUARTERS = 3      # rolling: a test quarter needs at least three labelled training quarters
N_BOOT = 500
SEED = 0
# A sign is kept if it adds at least half a point of ROC-AUC to the baseline, out of sample on the
# development quarters only. This threshold was set on 2026-11-13 after the development and test
# results had been seen (a rule of "any gain" would also have kept the one-year drawdown); it is
# fixed from here on, and the reserved quarter (2026Q3) is the test that no choice has seen.
MIN_GAIN = 0.005
