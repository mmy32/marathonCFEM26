"""Scan the XBRL instances for non-accrual flags on schedule-of-investments facts.

BDCs mark non-accrual loans with a footnote in the schedule of investments. In the XBRL
instance the footnote is a link:footnote, attached to facts by link:footnoteArc (fact -> loc
-> footnote). Each fact names its context, and FINAL_DATASET keeps every row's context_id, so
a row is on non-accrual when one of its facts links to a non-accrual footnote.

For every filing in FINAL_DATASET this writes (data/parsed/nonaccrual/):

    footnote_texts.csv   cik, accession, footnote label, text, n linked facts, for every footnote
                         whose text mentions non-accrual (to classify the wording)
    footnote_contexts.csv cik, accession, context_id, footnote label: contexts linked to those
                         footnotes
    filing_totals.csv    undimensioned facts whose concept name mentions non-accrual, plus net
                         assets (us-gaap:StockholdersEquity / AssetsNet), per filing: the BDC's
                         own totals, for the reconciliation and the NAV-based benchmark
    scan_log.csv         per filing: facts, footnotes, arcs, non-accrual footnotes, seconds

    python src/1_data_processing/nonaccrual_scan.py
"""
from __future__ import annotations

import csv
import re
import sys
import time

from lxml import etree

from paths import FINAL_DIR, IXBRL_RAW_XML_DIR, PARSED_DIR

OUT_DIR = PARSED_DIR / "nonaccrual"
XL = "{http://www.w3.org/1999/xlink}"
NA_TEXT = re.compile(r"non[\s-]?accru", re.I)
NA_CONCEPT = re.compile(r"non.?accru", re.I)
NAV_CONCEPTS = {"StockholdersEquity", "AssetsNet"}


def scan_file(path):
    fact_ctx, fn_text, locs, arcs, totals = {}, {}, {}, [], []
    contexts_with_dims = set()
    for _, el in etree.iterparse(str(path), events=("end",), huge_tree=True, recover=True):
        if not isinstance(el.tag, str):
            continue
        ln = etree.QName(el.tag).localname
        if ln == "context":
            if any(etree.QName(c.tag).localname in ("explicitMember", "typedMember") for c in el.iter() if isinstance(c.tag, str)):
                contexts_with_dims.add(el.get("id"))
            period = [c.text for c in el.iter() if isinstance(c.tag, str) and etree.QName(c.tag).localname in ("instant", "endDate")]
            el.attrib["_period"] = period[-1] if period else ""
            ctx_period[el.get("id")] = el.attrib["_period"]
            el.clear()
        elif ln == "footnote":
            fn_text[el.get(XL + "label")] = " ".join("".join(el.itertext()).split())
            el.clear()
        elif ln == "loc":
            locs[el.get(XL + "label")] = el.get(XL + "href", "").split("#")[-1]
            el.clear()
        elif ln == "footnoteArc":
            arcs.append((el.get(XL + "from"), el.get(XL + "to")))
            el.clear()
        elif el.get("contextRef") is not None:
            ctx = el.get("contextRef")
            if el.get("id"):
                fact_ctx[el.get("id")] = ctx
            if NA_CONCEPT.search(ln) or ln in NAV_CONCEPTS:
                totals.append((etree.QName(el.tag).namespace, ln, ctx, (el.text or "").strip(),
                               el.get("decimals"), el.get("unitRef")))
            el.clear()
    na_fn = {k for k, v in fn_text.items() if NA_TEXT.search(v)}
    linked = {}
    for fr, to in arcs:
        if to in na_fn and fr in locs and locs[fr] in fact_ctx:
            linked.setdefault(to, set()).add(fact_ctx[locs[fr]])
    totals = [t for t in totals if t[2] not in contexts_with_dims]
    return fact_ctx, fn_text, arcs, na_fn, linked, totals


ctx_period: dict = {}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = set()
    with open(FINAL_DIR / "FINAL_DATASET.csv", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        h = next(r)
        ic, ia = h.index("cik"), h.index("accession")
        for row in r:
            keys.add((row[ic], row[ia]))
    keys = sorted(keys)
    print(f"Scanning {len(keys)} filings ...")
    w = {name: csv.writer(open(OUT_DIR / f"{name}.csv", "w", encoding="utf-8", newline=""), lineterminator="\n")
         for name in ("footnote_texts", "footnote_contexts", "filing_totals", "scan_log")}
    w["footnote_texts"].writerow(["cik", "accession", "footnote", "n_contexts", "text"])
    w["footnote_contexts"].writerow(["cik", "accession", "context_id", "footnote"])
    w["filing_totals"].writerow(["cik", "accession", "namespace", "concept", "context_id", "period", "value", "decimals", "unit"])
    w["scan_log"].writerow(["cik", "accession", "status", "facts", "footnotes", "arcs", "na_footnotes", "na_contexts", "seconds"])
    t_all = time.time()
    for i, (cik, acc) in enumerate(keys):
        path = IXBRL_RAW_XML_DIR / f"{cik}_{acc.zfill(18)}.xml"
        if not path.exists():
            w["scan_log"].writerow([cik, acc, "missing", 0, 0, 0, 0, 0, 0])
            continue
        t0 = time.time()
        ctx_period.clear()
        try:
            fact_ctx, fn_text, arcs, na_fn, linked, totals = scan_file(path)
        except Exception as e:  # noqa: BLE001 - log and move on
            w["scan_log"].writerow([cik, acc, f"error: {e}"[:200], 0, 0, 0, 0, 0, round(time.time() - t0, 1)])
            continue
        for fn in sorted(na_fn):
            w["footnote_texts"].writerow([cik, acc, fn, len(linked.get(fn, ())), fn_text[fn][:2000]])
            for c in sorted(linked.get(fn, ())):
                w["footnote_contexts"].writerow([cik, acc, c, fn])
        for ns, concept, ctx, val, dec, unit in totals:
            w["filing_totals"].writerow([cik, acc, ns, concept, ctx, ctx_period.get(ctx, ""), val, dec, unit])
        n_ctx = len(set().union(*linked.values())) if linked else 0
        w["scan_log"].writerow([cik, acc, "ok", len(fact_ctx), len(fn_text), len(arcs), len(na_fn), n_ctx,
                                round(time.time() - t0, 1)])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(keys)}  {time.time() - t_all:.0f}s", flush=True)
    print(f"Done in {time.time() - t_all:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
