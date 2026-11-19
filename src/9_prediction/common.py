"""Small helpers shared by the model modules (src/9_prediction): quarter arithmetic and borrower-name
normalisation. Quarters are labelled 'YYYYQn' and sorted as year * 4 + quarter (qsort)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HORIZON


def q_label(qsort) -> list[str]:
    """qsort (year * 4 + quarter) back to 'YYYYQn'."""
    q = np.asarray(qsort, dtype=int)
    return [f"{y}Q{n}" for y, n in zip((q - 1) // 4, (q - 1) % 4 + 1)]


def q_sort(label: str) -> int:
    return int(label[:4]) * 4 + int(label[-1])


def q_next(label: str, k: int = 1) -> str:
    return q_label([q_sort(label) + k])[0]


def window_label(qsort) -> list[str]:
    """The outcome window of signal quarter t: 't+1-t+HORIZON', e.g. '2026Q1-2026Q2'."""
    q = np.asarray(qsort, dtype=int)
    return [f"{a}-{b}" for a, b in zip(q_label(q + 1), q_label(q + HORIZON))]


def norm_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()
