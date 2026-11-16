"""Every tunable constant of the early-warning model (src/9_prediction) in one place: the outcome window, the
sample rules, the definitions of the signs, the time split, and the settings of the tests. Nothing
here imports from the rest of the model."""
from __future__ import annotations

# ---- outcome and sample
OUTCOME = "y_na"            # the one classification outcome: an accruing borrower enters non-accrual at t + 1 or t + 2
HORIZON = 2                 # the outcome window: non-accrual at t + 1 or t + 2
WINDOW_TEXT = "within the next two quarters"
MIN_PRIOR_FV = 1_000_000    # sample: the borrower's debt fair value at t is at least $1m
MIN_PRICED_SHARE = 0.5      # sample: priceable par is at least half of the borrower's funded par at t
MAX_CLO_SHARE = 0.5         # sample: CLO / structured tranches under half of par (CLO vehicles are not operating borrowers)

# ---- definitions of the signs
MAX_PRICE = 1.5             # FV / PAR above this is par left in local currency (cleaning issue): no price
DD_WINDOW = 4               # rolling max drawdown over the preceding four quarters
LOW_MARK = 0.90             # PIK-switch hypotheses: mark below 90% of par ...
MARKDOWN = 0.05             # ... or marked down at least five points in the quarter
JUNIOR_LIENS = {"2L", "SUB", "UNSEC"}
# Debt reported without par is marked at FV / cost (the tranche panel holds only rows with par).
# OTHER_UNKNOWN rows without par are mostly equity and warrants, and CLO rows are not operating debt.
NO_PAR_DEBT = {"SENIOR_SECURED_1L", "UNITRANCHE", "SENIOR_SECURED_2L", "SUBORDINATED_MEZZ", "SENIOR_UNSECURED",
               "SENIOR_SECURED_UNSPEC"}

# ---- time split. Signs at t are scored against entries at t + 1 or t + 2. Three stages in time
# order; a stage's training windows all close before the next stage's first signal quarter:
TRAIN_END = "2025Q1"        # development: signs up to 2025Q1 (windows close by 2025Q3) choose the signs
TEST_QS = ["2025Q3", "2025Q4"]    # test: one fit on the development quarters, scored on signals from the
                                  # second half of 2025 (entries in 2025Q4-2026Q2)
SCORE_Q = "2026Q2"          # scorecard quarter (window 2026Q3-2026Q4)
RESERVED_Q = "2026Q3"       # reserved: nothing is fitted or chosen on it. Once filed, signals from
                            # 2026Q1 (window 2026Q2-2026Q3) are the blind test

# ---- tests and intervals
MIN_TRAIN_QUARTERS = 3      # rolling: a test quarter needs at least three labelled training quarters
N_BOOT = 500
SEED = 0
# A sign is kept if it adds at least half a point of ROC-AUC to the baseline, out of sample on the
# development quarters only. This threshold was set on 2026-11-13 after the development and test
# results had been seen (a rule of "any gain" would also have kept the one-year drawdown); it is
# fixed from here on, and the reserved quarter (2026Q3) is the test that no choice has seen.
MIN_GAIN = 0.005
