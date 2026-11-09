"""Parse one filing's iXBRL instance (from SEC EDGAR) into a wide table: one row per schedule-of-
investments context on the report date, one column per fact. Used by scripts/run_ixbrl_parser.py."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_data_processing"))  # for paths.py

import re
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from lxml import etree

from paths import ensure_dirs, IXBRL_RAW_XML_DIR, IXBRL_PER_FILE_DIR, IXBRL_WIDE_ALL_CSV

logger = logging.getLogger("ixbrl_parser")


# -------------------------
# Config container
# -------------------------
@dataclass(frozen=True)
class IXBRLConfig:
    user_agent: str
    request_timeout: int = 20
    sleep_between: float = 0.6
    retry: int = 2

    # output controls
    write_raw_xml: bool = True
    write_per_file_wide: bool = False     # keep default clean
    write_per_file_contexts: bool = False # keep default clean


# -------------------------
# Core helpers
# -------------------------
SEC_HOST = "https://www.sec.gov"


def _abs_href(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    return href if href.startswith("http") else SEC_HOST + href


def parse_accession_from_index_url(index_url: str) -> str:
    m = re.search(r"/Archives/edgar/data/\d+/([0-9]{14,})/", index_url)
    return m.group(1) if m else "noacc"


def _namespace_from_root(root) -> dict:
    ns = {(k or "xbrli"): v for k, v in root.nsmap.items()}
    ns.setdefault("xbrli", "http://www.xbrl.org/2003/instance")
    ns.setdefault("xbrldi", "http://xbrl.org/2006/xbrldi")
    return ns


def _context_period_text(ctx, ns):
    inst = ctx.find(".//xbrli:period/xbrli:instant", namespaces=ns)
    return inst.text.strip() if inst is not None and inst.text else None


def extract_context_members_both(xml_file: Path, root=None, ns: Optional[dict] = None) -> pd.DataFrame:
    if root is None:
        root = etree.parse(str(xml_file)).getroot()
    if ns is None:
        ns = _namespace_from_root(root)

    rows = []
    for ctx in root.findall(".//xbrli:context", namespaces=ns):
        ctx_id = ctx.get("id") or ""
        period_text = _context_period_text(ctx, ns)
        if not period_text:
            continue

        for em in ctx.findall(".//xbrldi:explicitMember", namespaces=ns):
            rows.append((ctx_id, em.get("dimension", ""), (em.text or "").strip(), period_text, "explicit"))

        for tm in ctx.findall(".//xbrldi:typedMember", namespaces=ns):
            child = next(iter(tm), None)
            val = child.text.strip() if child is not None and child.text else ""
            rows.append((ctx_id, tm.get("dimension", ""), val, period_text, "typed"))

    return pd.DataFrame(rows, columns=["context_id", "dimension", "value", "period", "member_kind"])


def filter_contexts_by_report_date(df_ctx: pd.DataFrame, report_date) -> pd.DataFrame:
    target = pd.to_datetime(report_date).date()
    period_dates = pd.to_datetime(df_ctx["period"], errors="coerce").dt.date
    return df_ctx[period_dates == target].copy()


def find_extracted_instance_url(index_url: str, session: requests.Session, cfg: IXBRLConfig) -> Optional[str]:
    """
    Find the 'EXTRACTED XBRL INSTANCE DOCUMENT' link on the SEC filing index page.
    """
    for attempt in range(cfg.retry + 1):
        try:
            resp = session.get(index_url, timeout=cfg.request_timeout)
            resp.raise_for_status()
            break
        except Exception:
            if attempt >= cfg.retry:
                return None
            time.sleep(1.0)

    soup = BeautifulSoup(resp.text, "html.parser")
    for tr in soup.select("table tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        desc = " ".join((tds[1].get_text() or "").lower().split())
        if "extracted xbrl instance document" in desc:
            a = tr.find("a")
            href = a.get("href") if a else None
            return _abs_href(href)
    return None


def download_xml(xml_url: str, out_path: Path, session: requests.Session, cfg: IXBRLConfig) -> bool:
    for attempt in range(cfg.retry + 1):
        try:
            resp = session.get(xml_url, timeout=cfg.request_timeout)
            resp.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(resp.content)
            return True
        except Exception:
            if attempt >= cfg.retry:
                return False
            time.sleep(1.0)
    return False


def parse_facts_wide(xml_file: Path, kept_contexts: pd.DataFrame, root=None, ns: Optional[dict] = None) -> pd.DataFrame:
    """
    Build a wide table keyed by context_id from all facts with contextRef,
    restricted to contexts in kept_contexts.
    """
    if root is None:
        root = etree.parse(str(xml_file)).getroot()
    if ns is None:
        ns = _namespace_from_root(root)

    base = kept_contexts[["context_id", "period"]].drop_duplicates()
    rows = {r.context_id: {"context_id": r.context_id, "period": r.period} for r in base.itertuples()}

    skip_ns = {v for v in (ns.get("xbrli"), ns.get("xbrldi"), ns.get("link")) if v}

    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if "contextRef" not in el.attrib:
            continue

        # skip structural namespaces
        uri = el.tag[1:].split("}", 1)[0] if el.tag.startswith("{") else None
        if uri in skip_ns:
            continue

        ctx_id = el.attrib.get("contextRef", "")
        if ctx_id not in rows:
            continue

        local = el.tag.split("}")[-1]
        col = f"{el.prefix}:{local}" if el.prefix else local

        # de-dupe repeated concepts in same context
        if col in rows[ctx_id]:
            i = 2
            while f"{col}#{i}" in rows[ctx_id]:
                i += 1
            col = f"{col}#{i}"

        rows[ctx_id][col] = (el.text or "").strip()
        rows[ctx_id][f"{col}-unitRef"] = el.attrib.get("unitRef", "")

    return pd.DataFrame(rows.values())


# -------------------------
# Public API: parse one filing row
# -------------------------
def process_one_filing(
    row: pd.Series,
    session: requests.Session,
    cfg: IXBRLConfig,
) -> Tuple[Optional[pd.DataFrame], bool, str]:
    """
    Parse one metadata row into a wide context table.

    Returns: (df_wide, ok, status)
      - df_wide includes: cik, accession, reportDate, context_id, period, investment_identifier, plus fact columns
      - ok False if cannot parse
      - status: 'success' or a reason code
    """
    cik = str(row.get("cik", "")).strip()
    index_url = str(row.get("index_url", "")).strip()
    report_date = row.get("reportDate", "")

    if not cik or not index_url or not report_date:
        return None, False, "missing_key_fields"

    xml_url = find_extracted_instance_url(index_url, session, cfg)
    if not xml_url:
        return None, False, "no_instance"

    accession = parse_accession_from_index_url(index_url)
    xml_path = IXBRL_RAW_XML_DIR / f"{cik}_{accession}.xml"

    if cfg.write_raw_xml and not xml_path.exists():
        # The SEC drops requests fired back-to-back: a short gap between the index-page fetch and
        # this download (cfg.sleep_between is already applied between filings by the caller).
        time.sleep(min(cfg.sleep_between, 0.3))
        ok = download_xml(xml_url, xml_path, session, cfg)
        if not ok:
            return None, False, "download_fail"

    if not xml_path.exists():
        return None, False, "xml_missing"

    try:
        root = etree.parse(str(xml_path)).getroot()
    except etree.XMLSyntaxError:
        # A cached file can be truncated (e.g. a disk-full crash mid-write): re-download once.
        xml_path.unlink(missing_ok=True)
        if not download_xml(xml_url, xml_path, session, cfg):
            return None, False, "download_fail"
        try:
            root = etree.parse(str(xml_path)).getroot()
        except etree.XMLSyntaxError:
            return None, False, "corrupt_xml"

    ns = _namespace_from_root(root)

    ctx_all = extract_context_members_both(xml_path, root=root, ns=ns)
    ctx_typed = ctx_all[ctx_all["member_kind"] == "typed"].copy()
    if ctx_typed.empty:
        return None, False, "no_typed"

    ctx_typed = filter_contexts_by_report_date(ctx_typed, report_date)
    if ctx_typed.empty:
        return None, False, "no_contexts_on_date"

    kept = ctx_typed[["context_id", "period"]].drop_duplicates()
    id_map = ctx_typed.groupby("context_id")["value"].first().rename("investment_identifier")

    df_wide = parse_facts_wide(xml_path, kept, root=root, ns=ns)
    df_wide = df_wide.merge(id_map.reset_index(), on="context_id", how="left")

    # add keys
    df_wide.insert(0, "cik", cik)
    df_wide.insert(1, "accession", accession)
    df_wide.insert(2, "reportDate", str(report_date))

    # optional per-file outputs (off by default)
    if cfg.write_per_file_wide:
        out_dir = IXBRL_PER_FILE_DIR / cik
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{accession}_wide.csv").write_text(df_wide.to_csv(index=False), encoding="utf-8")

    if cfg.write_per_file_contexts:
        out_dir = IXBRL_PER_FILE_DIR / cik
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{accession}_contexts.csv").write_text(ctx_all.to_csv(index=False), encoding="utf-8")

    return df_wide, True, "success"


def append_wide_frames_bounded(
    frames: list[pd.DataFrame],
    out_csv: Path = IXBRL_WIDE_ALL_CSV,
    overwrite: bool = False,
    chunksize: int = 10_000,
) -> None:
    """Append parsed frames to the combined wide table without loading the whole file. New columns
    (each filer has its own XBRL extension columns) are added by rewriting the file in chunks of
    `chunksize` rows, read as text so values are unchanged. Only within-batch duplicates are
    removed; the caller's resume logic keeps already-parsed filings out of `frames`."""
    if not frames:
        return

    batch = pd.concat(frames, ignore_index=True)

    dedup_key = [c for c in ["cik", "accession", "context_id"] if c in batch.columns]
    if dedup_key:
        for c in dedup_key:
            batch[c] = batch[c].astype(str)
        batch = batch.drop_duplicates(subset=dedup_key).reset_index(drop=True)

    if overwrite or not out_csv.exists():
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        batch.to_csv(out_csv, index=False)
        return

    existing_cols = pd.read_csv(out_csv, nrows=0).columns.tolist()
    new_cols = [c for c in batch.columns if c not in existing_cols]

    if not new_cols:
        batch.reindex(columns=existing_cols).to_csv(out_csv, mode="a", header=False, index=False)
        return

    # Schema widening: rewrite the existing file chunk-by-chunk with the new
    # columns added (NaN for prior rows), then append the new batch.
    full_cols = existing_cols + new_cols
    tmp_path = out_csv.with_name(out_csv.stem + ".tmp" + out_csv.suffix)
    first_write = True
    for chunk in pd.read_csv(out_csv, chunksize=chunksize, dtype=str, low_memory=False):
        chunk.reindex(columns=full_cols).to_csv(
            tmp_path, mode="w" if first_write else "a", header=first_write, index=False
        )
        first_write = False
    batch.reindex(columns=full_cols).to_csv(
        tmp_path, mode="w" if first_write else "a", header=first_write, index=False
    )
    tmp_path.replace(out_csv)


def init_session(user_agent: str) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": user_agent})
    return sess


def init_ixbrl_dirs() -> None:
    ensure_dirs()   # includes the raw_xml, per_file and combined folders
