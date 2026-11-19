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
# The baseline and the signs the frozen rule kept on the development quarters (2026-11-16, outcome:
# a credit event within two quarters): PIK switch and fall over the past year; the valuation gap
# (+0.004, under the rule's +0.005) and crowding were dropped. With non-accrual alone as the
# outcome the rule had also kept the valuation gap. run_model.py stops if sign_selection.csv ever
# keeps a different set.
SCORECARD = BASELINE + CANDIDATES["pik_switch"] + CANDIDATES["max_drawdown"]
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


NO_CAPS = "no caps: mark, its change and the fall over the year as reported"
NON_ACCRUAL_SIGNS = BASELINE + CANDIDATES["pik_switch"] + CANDIDATES["max_drawdown"] + CANDIDATES["dispersion"]


def _robustness() -> list[tuple]:
    """(variant, outcome, predictors, multi-lender borrowers only): the scorecard varied one way
    at a time. The last rows change the outcome and say so in their name."""
    v = [("scorecard", OUTCOME, SCORECARD, False), (NO_CAPS, OUTCOME, SCORECARD, False)]
    if "pik_switch" in SCORECARD:
        v.append(("PIK: change in PIK share of coupon (pp)", OUTCOME, _swap(SCORECARD, "pik_switch", ["d_pik_pp"]), False))
    v += [("market-adjusted mark and change", OUTCOME,
           _swap(_swap(SCORECARD, "price", ["price_rel"]), "d_price", ["d_price_adj"]), False),
          ("+ seniority mix (junior share of par)", OUTCOME, SCORECARD + ["junior_share"], False),
          ("+ borrower size (log BDC-held par)", OUTCOME, SCORECARD + ["log_par"], False)]
    if "dispersion" in SCORECARD:
        v.append(("valuation gap as the std of family marks", OUTCOME, _swap(SCORECARD, "dispersion", ["dispersion_std"]), False))
    else:
        v.append(("+ valuation gap (dropped by the sign rule)", OUTCOME, SCORECARD + CANDIDATES["dispersion"], False))
    v += [("borrowers with 2+ lender families only", OUTCOME, [c for c in SCORECARD if c != "has_dispersion"], True),
          ("mark, change and PIK switch only", OUTCOME, BASELINE + ["pik_switch"], False),
          ("outcome: non-accrual alone (the outcome until 2026-11-16)", "y_na", SCORECARD, False),
          ("outcome: non-accrual alone, with the five signs the rule kept for it", "y_na", NON_ACCRUAL_SIGNS, False),
          ("outcome: non-accrual alone, first-ever", "y_na_first", SCORECARD, False),
          ("outcome: non-accrual alone, same-lender entries only", "y_na_same_lender", SCORECARD, False),
          ("outcome: non-accrual or any of the five term changes", "y_all_terms", SCORECARD, False)]
    return v


ROBUSTNESS = _robustness()
