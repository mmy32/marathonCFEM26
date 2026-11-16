"""Which inputs go in which model (src/9_prediction): the baseline, the candidate signs, the scorecard, the
specifications compared in the analysis, the robustness variants and the one-input baselines."""
from __future__ import annotations

from config import OUTCOME

# The baseline: the mark and its change over the quarter. Each other sign is added to it alone and
# kept only if it improves the ranking out of sample (analysis.sign_selection).
BASELINE = ["price", "d_price"]
# Dispersion exists only for borrowers with a tranche held by 2+ manager families; it is 0
# elsewhere and comes with its indicator.
CANDIDATES = {"pik_switch": ["pik_switch"], "dispersion": ["dispersion", "has_dispersion"],
              "crowding": ["crowding"], "max_drawdown": ["max_drawdown"]}
# The baseline and the signs the frozen rule kept on the development quarters (2026-11-14, outcome
# within two quarters): PIK switch, fall over the past year, valuation gap; crowding was dropped.
# run_model.py stops if sign_selection.csv ever keeps a different set.
SCORECARD = BASELINE + CANDIDATES["pik_switch"] + CANDIDATES["max_drawdown"] + CANDIDATES["dispersion"]
SIGN_NAMES = {"pik_switch": "PIK switch", "dispersion": "valuation gap", "crowding": "crowding",
              "max_drawdown": "fall over the past year"}
# The four inputs of the one-input baselines, and the markdown outcomes (next quarter)
INPUTS = {"mark level": ["price"], "rolling max drawdown": ["max_drawdown"], "PIK switch": ["pik_switch"],
          "valuation dispersion": ["dispersion", "has_dispersion"]}
MARKDOWN_OUTCOMES = {"price change (y_dprice)": "y_dprice", "val_ret (y_valret)": "y_valret"}


def specs() -> dict:
    """The mark alone, the baseline (mark and its change), and the baseline with each other sign."""
    return {"mark alone": ["price"], "baseline": BASELINE,
            **{f"baseline + {SIGN_NAMES[k]}": BASELINE + cols for k, cols in CANDIDATES.items()},
            "scorecard": SCORECARD}


def _swap(cols: list[str], old: str, new: list[str]) -> list[str]:
    i = cols.index(old)
    return cols[:i] + new + cols[i + 1:]


ROBUSTNESS = [  # (variant, outcome, predictors, multi-lender borrowers only); the scorecard varied
    ("scorecard", OUTCOME, SCORECARD, False),
    ("PIK: change in PIK share of coupon (pp)", OUTCOME, _swap(SCORECARD, "pik_switch", ["d_pik_pp"]), False),
    ("market-adjusted mark and change", OUTCOME,
     _swap(_swap(SCORECARD, "price", ["price_rel"]), "d_price", ["d_price_adj"]), False),
    ("+ seniority mix (junior share of par)", OUTCOME, SCORECARD + ["junior_share"], False),
    ("+ borrower size (log BDC-held par)", OUTCOME, SCORECARD + ["log_par"], False),
    ("valuation gap as the range of family marks", OUTCOME, _swap(SCORECARD, "dispersion", ["dispersion_range"]), False),
    ("borrowers with 2+ lender families only", OUTCOME, [c for c in SCORECARD if c != "has_dispersion"], True),
    ("without the two weakest signs (mark, change, PIK switch)", OUTCOME, BASELINE + ["pik_switch"], False),
    ("outcome: first-ever non-accrual", "y_na_first", SCORECARD, False),
    ("outcome: same-lender entries only", "y_na_same_lender", SCORECARD, False),
]
