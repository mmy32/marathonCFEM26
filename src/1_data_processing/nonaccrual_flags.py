"""Turn the scanned XBRL footnotes (nonaccrual_scan.py) into a row-level non-accrual flag.

A footnote counts as a non-accrual flag when its wording says the marked investment is on
non-accrual (POSITIVE) and nothing in it says the opposite or makes it a general note
(NEGATIVE): "excluding those on non-accrual", "there were no investments on non-accrual",
"all debt investments are income producing except ...". A footnote linked to more than half of
a filing's rows, or a long one (> LONG_TEXT characters) linked to more than a quarter, is a
general note, not a flag, and is dropped.

Row value (`non_accrual` in FINAL_DATASET):
    1     the row's context is linked to a flag footnote
    0     not linked, in a filing whose non-accrual status is known (see load_flags)
    blank the filing carries no non-accrual footnote in its XBRL: the BDC marks non-accrual in
          the HTML only, or says nothing; unknown, not zero
"""
from __future__ import annotations

import re

import pandas as pd

from paths import PARSED_DIR

SCAN_DIR = PARSED_DIR / "nonaccrual"
MAX_LINKED_SHARE = 0.25

POSITIVE = re.compile(
    r"\b(?:is|was|are|were|been|being|remains?|currently|placed|classified|reported|investments?|loans?|debt|security|asset|positions?)\b"
    r"[\w\s,()-]{0,40}?\bnon[\s-]?accru"
    r"|^\W*\(?\w?\)?\W*(?:partial\s+)?(?:pik\s+|payment[\s-]in[\s-]kind\s*\(?pik\)?\s+)?(?:partial\s+)?non[\s-]?accru(?:al|ing)\b",
    re.I)
NEGATIVE = re.compile(
    r"exclud|except|\bno (?:portfolio company )?investments?\b|there (?:were|are|was|is) no|had no\b|held no\b|"
    r"\bnone\b|net of non|composition|affiliated person|\bspv\b|unfunded|undrawn|gross (?:additions|reductions)|"
    r"weighted average|yield|not on non|nor non|no non[\s-]?accru|held for collateral|co-?investments|"
    r"pays,? or has the option to pay|cost value reflects|holds investments in",
    re.I)
LONG_TEXT = 150   # characters: a long footnote linked to many rows is a general note, not a flag


def classify_texts(texts: pd.DataFrame, linked_share: pd.Series) -> pd.DataFrame:
    """Add is_flag and why to footnote_texts.csv rows. linked_share: (accession, footnote) ->
    share of the filing's FINAL_DATASET rows whose context the footnote is linked to."""
    t = texts.copy()
    share = pd.Series([linked_share.get((a, f), 0.0) for a, f in zip(t["accession"], t["footnote"])], index=t.index)
    pos = t["text"].str.contains(POSITIVE)
    neg = t["text"].str.contains(NEGATIVE)
    # Short explicit labels ("investment is on non-accrual status") stay flags up to half of a
    # filing's rows (wind-down funds); long texts linked to over a quarter of rows are notes.
    general = (share > 2 * MAX_LINKED_SHARE) | ((share > MAX_LINKED_SHARE) & (t["text"].str.len() > LONG_TEXT))
    t["linked_share"] = share
    t["is_flag"] = pos & ~neg & ~general
    t["why"] = (pos.map({True: "positive wording", False: "no positive wording"})
                + neg.map({True: "; negative wording", False: ""})
                + general.map({True: "; general note (linked to too many rows)", False: ""}))
    return t


def linked_shares(rows: pd.DataFrame, ctx: pd.DataFrame) -> pd.Series:
    """(accession, footnote) -> share of the filing's rows (accession, context_id) linked to it."""
    n = rows.groupby("accession").size()
    hit = ctx.merge(rows[["accession", "context_id"]], on=["accession", "context_id"])
    k = hit.groupby(["accession", "footnote"]).size()
    return k / k.index.get_level_values("accession").map(n).to_numpy()


EXPLICIT_NONE = re.compile(r"(?:were|are|had|held) no (?:portfolio company )?investments?\b[^.]{0,40}non[\s-]?accru|"
                           r"there (?:were|are) no (?:investments|loans)[^.]{0,60}non[\s-]?accru|no non[\s-]?accru", re.I)


def load_flags(rows: pd.DataFrame) -> tuple[set, set, pd.DataFrame]:
    """rows: FINAL_DATASET accession (str) and context_id. Returns ({(accession, context_id)}
    on non-accrual, {accession} whose non-accrual status is known, classified texts).

    A filing's status is known when it carries a flag-type footnote (linked to holdings or,
    in a quarter with none on non-accrual, defined but unused) or states that no investment
    is on non-accrual. Otherwise its rows are unknown: the BDC marks non-accrual in the HTML
    only, which the XBRL does not carry."""
    texts = pd.read_csv(SCAN_DIR / "footnote_texts.csv", dtype={"accession": str})
    ctx = pd.read_csv(SCAN_DIR / "footnote_contexts.csv", dtype={"accession": str})
    texts = classify_texts(texts, linked_shares(rows, ctx))
    flagged = ctx.merge(texts.loc[texts["is_flag"], ["accession", "footnote"]], on=["accession", "footnote"])
    pairs = set(zip(flagged["accession"], flagged["context_id"]))
    flag_wording = texts["text"].str.contains(POSITIVE) & ~texts["text"].str.contains(NEGATIVE)
    known = set(texts.loc[texts["is_flag"] | flag_wording | texts["text"].str.contains(EXPLICIT_NONE), "accession"])
    return pairs, known, texts
