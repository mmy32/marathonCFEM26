"""Step 7b: validated labels for identifiers that labels_by_identifier.csv does not cover yet (a new quarter).

Reads step 7's output (data/enriched/data_private_credit_FINAL_enriched_final.csv), takes every
investment_identifier with no validated label, and asks Claude (default claude-haiku-5-5) to confirm,
correct or fill borrower, seniority and sector under the rules in final_validation_prompt.md. Batches
are grouped by BDC and carry that BDC's existing borrower names ("known_borrowers") so a borrower
keeps one spelling across quarters. Answers are checked against the allowed lists and the 0.70 rule,
cached in data/final_dataset/_new_label_answers.jsonl (a re-run only asks for what is missing), then
appended to labels_by_identifier.csv and labels_validated.json with the next ids. A borrower that
already has a validated sector gets that sector (and its evidence), as in the original validation,
which set one sector per borrower. Existing labels are never changed.

The API key is read from ANTHROPIC_API_KEY, or from the file named in ANTHROPIC_API_KEY_FILE.

    python src/1_data_processing/label_new_identifiers.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ENRICHED_DIR, FINAL_DIR

ENRICHED_CSV = ENRICHED_DIR / "data_private_credit_FINAL_enriched_final.csv"
LABELS_CSV = FINAL_DIR / "labels_by_identifier.csv"
LABELS_JSON = FINAL_DIR / "labels_validated.json"
ANSWERS = FINAL_DIR / "_new_label_answers.jsonl"
PROMPT_MD = Path(__file__).resolve().parent / "final_validation_prompt.md"

MODEL = "claude-haiku-5-5"
BATCH = 25
WORKERS = 8
MIN_CONF = 0.70

SENIORITY = {"SENIOR_SECURED_1L", "SENIOR_SECURED_2L", "SENIOR_SECURED_UNSPEC", "UNITRANCHE", "SENIOR_UNSECURED",
             "SUBORDINATED_MEZZ", "CLO_STRUCTURED_EQUITY", "OTHER_UNKNOWN"}
SECTORS = {"Healthcare", "Technology", "Financials & Insurance", "Transportation", "Consumer", "Industrials",
           "Energy & Utilities", "Materials", "Real Estate", "Communication & Media", "Other / Unknown",
           "Other / Unknown / Unresolved"}
EVIDENCE_TYPES = {"reported_industry", "identifier_text", "general_knowledge", "none"}
UNRESOLVED = "Other / Unknown / Unresolved"
LABEL_COLS = ["id", "investment_identifier", "rows_in_file", "n_bdcs", "borrower_prev", "borrower_canonical",
              "borrower_confidence", "borrower_action", "borrower_reason", "seniority_prev", "instrument_seniority",
              "seniority_confidence", "seniority_evidence", "seniority_action", "sector_prev", "sector",
              "sector_confidence", "sector_evidence_type", "sector_evidence", "sector_action"]


def _client():
    """Key from ANTHROPIC_API_KEY, or from the file named in ANTHROPIC_API_KEY_FILE: either just the key, or
    lines "API KEY: ..." and "Workspace ID: ..." (an identity-linked key needs the workspace id)."""
    import anthropic
    key, ws = os.getenv("ANTHROPIC_API_KEY", ""), os.getenv("ANTHROPIC_WORKSPACE_ID", "")
    if not key and os.getenv("ANTHROPIC_API_KEY_FILE"):
        text = Path(os.environ["ANTHROPIC_API_KEY_FILE"]).read_text().strip()
        m = re.search(r"api key\s*:\s*(\S+)", text, re.I)
        key = m.group(1) if m else text
        m = re.search(r"workspace id\s*:\s*(\S+)", text, re.I)
        ws = ws or (m.group(1) if m else "")
    if not key:
        raise SystemExit("Set ANTHROPIC_API_KEY, or ANTHROPIC_API_KEY_FILE to a file holding the key.")
    return anthropic.Anthropic(api_key=key, default_headers={"anthropic-workspace-id": ws} if ws else None)


def _problems(o: dict, item: dict) -> list[str]:
    """Why an answer is unusable (empty list: fine)."""
    bad = []
    if o.get("id") != item["id"]:
        bad.append("id")
    if o.get("instrument_seniority") not in SENIORITY:
        bad.append("seniority code")
    if o.get("sector") not in SECTORS:
        bad.append("sector label")
    if o.get("sector_evidence_type") not in EVIDENCE_TYPES:
        bad.append("evidence type")
    for k in ("borrower_confidence", "seniority_confidence", "sector_confidence"):
        if not isinstance(o.get(k), (int, float)) or not 0 <= o[k] <= 1:
            bad.append(k)
    if bad:
        return bad
    if o["borrower_canonical"] and o["borrower_confidence"] < MIN_CONF:
        bad.append("borrower kept below 0.70")
    if o["instrument_seniority"] != "OTHER_UNKNOWN" and o["seniority_confidence"] < MIN_CONF:
        bad.append("seniority kept below 0.70")
    if o["sector"] != UNRESOLVED and o["sector_confidence"] < MIN_CONF:
        bad.append("sector kept below 0.70")
    return bad


def _ask(client, system: str, items: list[dict], known: list[str]) -> list[dict]:
    payload = {"identifiers": items}
    if known:
        payload["known_borrowers"] = known
    msg = ("Validate this batch. Return only the JSON array.\n" + json.dumps(payload, ensure_ascii=False))
    last = "answer length differs from the batch"
    for attempt in range(4):
        try:
            r = client.messages.create(model=MODEL, max_tokens=16000, system=system,
                                       messages=[{"role": "user", "content": msg}])
            text = "".join(b.text for b in r.content if b.type == "text")
            text = text[text.index("["): text.rindex("]") + 1]
            out = json.loads(text)
            if len(out) == len(items):
                return out
        except Exception as e:  # rate limit, bad JSON, overload: retry
            last = repr(e)[:200]
            time.sleep(2 + 4 * attempt)
    raise RuntimeError(f"batch failed after retries: {last}")


def main(limit: int | None = None):
    enr = pd.read_csv(ENRICHED_CSV, usecols=["cik", "cal_q", "investment_identifier", "borrower_canonical",
                                             "instrument_seniority", "sector"], dtype=str, low_memory=False)
    labels = pd.read_csv(LABELS_CSV, dtype=str, keep_default_na=False)
    have = set(labels["investment_identifier"])
    enr = enr.dropna(subset=["investment_identifier"])
    new = enr[~enr["investment_identifier"].isin(have)]
    print(f"{new['investment_identifier'].nunique():,} new identifiers in {len(new):,} rows")

    g = new.groupby("investment_identifier", sort=False)
    ident = pd.DataFrame({"rows_in_file": g.size(), "n_bdcs": g["cik"].nunique(), "cik": g["cik"].first(),
                          "borrower_prev": g["borrower_canonical"].first().fillna(""),
                          "seniority_prev": g["instrument_seniority"].first().fillna("OTHER_UNKNOWN"),
                          "sector_prev": g["sector"].first().fillna(UNRESOLVED)}).reset_index()
    if limit:
        ident = ident.head(limit)
    next_id = int(pd.to_numeric(labels["id"]).max()) + 1
    ident["id"] = range(next_id, next_id + len(ident))

    # existing borrower names per BDC, for reuse
    cik_by_name = enr[enr["investment_identifier"].isin(have)].dropna(subset=["borrower_canonical"])
    lab_b = labels.merge(cik_by_name[["investment_identifier", "cik"]].drop_duplicates(), on="investment_identifier")
    known_by_cik = {c: sorted(set(x for x in d["borrower_canonical"] if x)) for c, d in lab_b.groupby("cik")}

    done = {}
    if ANSWERS.exists():
        for line in ANSWERS.read_text(encoding="utf-8").splitlines():
            a = json.loads(line)
            done[a["investment_identifier"]] = a
    todo = ident[~ident["investment_identifier"].isin(done)]
    print(f"{len(done):,} already answered, {len(todo):,} to ask")

    system = PROMPT_MD.read_text(encoding="utf-8")
    client = _client()
    lock = threading.Lock()
    stats = Counter()

    def run(batch: pd.DataFrame):
        items = [{"id": int(r.id), "investment_identifier": r.investment_identifier,
                  "borrower_canonical": r.borrower_prev, "instrument_seniority": r.seniority_prev,
                  "sector": r.sector_prev} for r in batch.itertuples()]
        words = {w for it in items for w in re.findall(r"[A-Za-z0-9]+", it["investment_identifier"].upper())}
        known = [n for n in known_by_cik.get(batch["cik"].iloc[0], []) if n.split(" ")[0] in words][:150]
        try:
            out = _ask(client, system, items, known)
        except Exception as e:
            with lock:
                stats["batch_failed"] += 1
            print("batch failed:", e)
            return
        lines = []
        for it, o in zip(items, out):
            p = _problems(o, it)
            if p:  # unusable answer: leave it for a re-run
                with lock:
                    stats["rejected"] += 1
                continue
            o["investment_identifier"] = it["investment_identifier"]
            lines.append(json.dumps(o, ensure_ascii=False))
        with lock:
            stats["ok"] += len(lines)
            with open(ANSWERS, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            print(dict(stats), flush=True)

    batches = []
    for _, d in todo.groupby("cik", sort=False):
        for i in range(0, len(d), BATCH):
            batches.append(d.iloc[i:i + BATCH])
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(run, batches))
    print("asked:", dict(stats))
    if limit:
        raise SystemExit("--limit: answers cached, nothing appended")

    # ---- assemble
    done = {}
    for line in ANSWERS.read_text(encoding="utf-8").splitlines():
        a = json.loads(line)
        done[a["investment_identifier"]] = a
    missing = [x for x in ident["investment_identifier"] if x not in done]
    if missing:
        raise SystemExit(f"{len(missing):,} identifiers still unanswered; re-run to retry them")

    # sector already settled for a borrower: reuse (one sector per borrower, as in the original validation)
    lw = labels.assign(w=pd.to_numeric(labels["rows_in_file"]))
    lw = lw[(lw["borrower_canonical"] != "") & (lw["sector"] != UNRESOLVED)]
    best = lw.sort_values("w", ascending=False).drop_duplicates("borrower_canonical").set_index("borrower_canonical")

    rows, js = [], []
    nid = {}
    for r in ident.itertuples():
        a = done[r.investment_identifier]
        b = a["borrower_canonical"].strip()
        sec, st, se, sc = a["sector"], a["sector_evidence_type"], a["sector_evidence"], a["sector_confidence"]
        if b and b in best.index:
            h = best.loc[b]
            sec, st, se, sc = h["sector"], h["sector_evidence_type"], h["sector_evidence"], float(h["sector_confidence"])
        b_act = ("blank" if not b else "keep" if b == r.borrower_prev else "fill" if not r.borrower_prev else "replace")
        s_act = "keep" if sec == r.sector_prev else "replace"
        rows.append({"id": r.id, "investment_identifier": r.investment_identifier, "rows_in_file": r.rows_in_file,
                     "n_bdcs": r.n_bdcs, "borrower_prev": r.borrower_prev, "borrower_canonical": b,
                     "borrower_confidence": a["borrower_confidence"], "borrower_action": b_act,
                     "borrower_reason": {"keep": "current value confirmed", "fill": "filled from identifier",
                                         "replace": "identifier names a different entity", "blank": "no single borrower"}[b_act],
                     "seniority_prev": r.seniority_prev, "instrument_seniority": a["instrument_seniority"],
                     "seniority_confidence": a["seniority_confidence"], "seniority_evidence": a.get("seniority_evidence", ""),
                     "seniority_action": "keep" if a["instrument_seniority"] == r.seniority_prev else "replace",
                     "sector_prev": r.sector_prev, "sector": sec, "sector_confidence": sc, "sector_evidence_type": st,
                     "sector_evidence": se, "sector_action": s_act})
        js.append({"id": r.id, "borrower_canonical": b, "borrower_confidence": a["borrower_confidence"],
                   "instrument_seniority": a["instrument_seniority"], "seniority_confidence": a["seniority_confidence"],
                   "seniority_evidence": a.get("seniority_evidence", ""), "sector": sec, "sector_confidence": sc,
                   "sector_evidence_type": st, "sector_evidence": se})
    add = pd.DataFrame(rows)[LABEL_COLS]
    add.to_csv(LABELS_CSV, mode="a", header=False, index=False)
    text = LABELS_JSON.read_text(encoding="utf-8").rstrip()
    assert text.endswith("]")
    body = text[:-1].rstrip() + ",\n" + ",\n".join(json.dumps(o, ensure_ascii=False) for o in js) + "\n]\n"
    LABELS_JSON.write_text(body, encoding="utf-8")
    print(f"appended {len(add):,} labels (ids {add['id'].min()}-{add['id'].max()})")
    print("sector:", add["sector"].value_counts().to_dict())
    print("seniority:", add["instrument_seniority"].value_counts().to_dict())
    print("borrower blank:", int((add["borrower_canonical"] == "").sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    main(ap.parse_args().limit)
