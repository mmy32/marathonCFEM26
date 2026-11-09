"""Maturity dates for the 76 BDCs that show them in the HTML schedule of investments but do not tag them in
XBRL, joined to panel rows on context_id. `download` caches the documents in cache/html_gz/; `extract`
writes data/parsed/maturity_html/maturity_from_html.csv. Rules in docs/6_final_dataset.md.
Usage: SEC_USER_AGENT="Name email" python src/1_data_processing/maturity_html.py download|extract"""
from __future__ import annotations

import collections
import gzip
import html as htmllib
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for paths.py
import pandas as pd

from paths import CACHE_DIR, ENRICHED_DIR, METADATA_CSV, PARSED_DIR

HTML_GZ_DIR = CACHE_DIR / "html_gz"
OUT_DIR = PARSED_DIR / "maturity_html"
OUT_CSV = OUT_DIR / "maturity_from_html.csv"
ENRICHED_CSV = ENRICHED_DIR / "data_private_credit_FINAL_enriched_final.csv"

_TAG = re.compile(r"<[^>]+>")
_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_TD = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
_CTX = re.compile(r"<ix:nonfraction[^>]*?contextref=\"([^\"]+)\"", re.I)
_MON = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTH_RE = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
# Order matters: full dates before month-only ones, so "5/7/2027" is not read as "7/2027".
_DATE = re.compile(
    r"(?<![\d/])(?P<mdy>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{4}|\d{2})(?![\d/])"
    r"|(?<![\d/])(?P<my_m>\d{1,2})/(?P<my_y>\d{4})(?![\d/])"
    rf"|(?P<mon>{_MONTH_RE})\s*(?P<mon_d>\d{{1,2}}),\s*(?P<mon_y>\d{{4}})"
    rf"|(?P<mon2>{_MONTH_RE})\s+(?P<mon2_y>\d{{4}})"
    r"|(?P<iso_y>\d{4})-(?P<iso_m>\d{2})-(?P<iso_d>\d{2})",
    re.I)


def _ts(y: int, m: int, d: int):
    if y < 100:
        y += 2000
    if not (1990 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31):
        return None
    try:
        return pd.Timestamp(y, m, d)
    except ValueError:
        return None


def find_dates(text: str) -> list:
    """Every date in a cell's text, in order."""
    out = []
    for m in _DATE.finditer(text):
        g = m.groupdict()
        if g["mdy"]:
            t = _ts(int(g["y"]), int(g["mdy"]), int(g["d"]))
        elif g["my_m"]:
            t = _ts(int(g["my_y"]), int(g["my_m"]), 15)
        elif g["mon"]:
            t = _ts(int(g["mon_y"]), _MON[g["mon"][:3].lower()], int(g["mon_d"]))
        elif g["mon2"]:
            t = _ts(int(g["mon2_y"]), _MON[g["mon2"][:3].lower()], 15)
        else:
            t = _ts(int(g["iso_y"]), int(g["iso_m"]), int(g["iso_d"]))
        if t is not None:
            out.append(t)
    return out


def _text(cell: str) -> str:
    return re.sub(r"\s+", " ", htmllib.unescape(_TAG.sub(" ", cell))).strip()


_MM_YY = re.compile(r"(\d{1,2})/(\d{2})")
_KEYWORD = re.compile(r"\b(?:due|matur\w*)\b\W{0,3}(?:date\W{0,3})?", re.I)


def _cell_dates(cell: str) -> list:
    """Dates in a cell under a maturity header. A cell that is only "MM/YY" is a month-year."""
    found = find_dates(cell)
    if not found:
        m = _MM_YY.fullmatch(cell)
        if m:
            t = _ts(int(m.group(2)), int(m.group(1)), 15)
            found = [t] if t is not None else []
    return found


def _keyword_dates(cell: str) -> list:
    """Dates right after "due" / "maturity" in free text ("Term Debt (S + 10%, Due 11/2026)")."""
    out = []
    for m in _KEYWORD.finditer(cell):
        date = _DATE.match(cell, m.end())
        if date is not None:
            out += find_dates(date.group(0))
    return out


def extract_maturities(html: str, report_date: pd.Timestamp | None = None) -> dict:
    """context_id -> (maturity Timestamp, cell text) for every schedule row with a date.

    Under a header with a maturity column, every date in the row counts. In tables without
    one, only dates right after "due" / "maturity" count. report_date (the filing's period
    end) decides rows with fewer dates than the header's date columns: their latest date is
    the maturity only if it is after report_date, since an acquisition date never is."""
    out = {}
    date_cols = None          # header cells naming a date, for the current schedule table
    for tr in _TR.findall(html):
        ctxs = _CTX.findall(tr)
        cells = [c for c in (_text(c) for c in _TD.findall(tr)) if c]
        if not ctxs:
            low = [c.lower() for c in cells]
            # A schedule header names cost, principal or par; a summary table ("Maturity Year |
            # Fair Value") does not. A schedule header without a maturity column resets it.
            if len(cells) >= 3 and any(w in c for c in low for w in ("cost", "principal", "par ", "par/", "par amount")):
                date_cols = ([c for c in low if "date" in c or "matur" in c]
                             if any("matur" in c for c in low) else None)
            continue
        # (cell, its dates) for every cell holding a date
        if date_cols is None:
            dated = [(c, d) for c in cells if (d := _keyword_dates(c))]
        else:
            dated = [(c, d) for c in cells if (d := _cell_dates(c))]
        if not dated:
            continue
        mpos = [i for i, h in enumerate(date_cols or []) if "matur" in h]
        if mpos and len(dated) == len(date_cols):
            cell, ds = dated[mpos[0]]
            mat = max(ds)
        elif date_cols and len(dated) < len(date_cols):
            # an equity row with only its acquisition date, or a loan with the acquisition
            # date left blank: only a date after the report date can be a maturity
            mat, cell = max((d, c) for c, ds in dated for d in ds)
            if report_date is None or mat <= report_date:
                continue
        else:
            mat, cell = max((d, c) for c, ds in dated for d in ds)
        ctx = collections.Counter(ctxs).most_common(1)[0][0]
        out.setdefault(ctx, (mat, cell))
    return out


# ---------------------------------------------------------------- download

def target_filings() -> pd.DataFrame:
    """Filings with at least one blank maturity in the enriched panel, with their HTML url."""
    p = pd.read_csv(ENRICHED_CSV, usecols=["cik", "accession", "period", "InvestmentMaturityDate"], dtype=str)
    report = p.groupby("accession")["period"].max().rename("report_date")   # latest context period
    need = p[p["InvestmentMaturityDate"].isna()][["cik", "accession"]].drop_duplicates()
    need = need.join(report, on="accession")
    meta = pd.read_csv(METADATA_CSV, dtype=str)
    meta["accession"] = meta["accessionNumber"].str.replace("-", "").str.lstrip("0")
    need["accession_key"] = need["accession"].str.lstrip("0")
    out = need.merge(meta[["accession", "url"]].rename(columns={"accession": "accession_key"}),
                     on="accession_key", how="left")
    missing = out["url"].isna().sum()
    if missing:
        print(f"[warn] {missing} filings have no url in {METADATA_CSV.name}")
    return out.dropna(subset=["url"]).drop(columns="accession_key")


def gz_path(cik: str, accession: str) -> Path:
    return HTML_GZ_DIR / str(cik) / f"{accession}.htm.gz"


def download(user_agent: str, sleep_between: float = 0.11) -> None:
    import requests
    from filings import throttled_get
    t = target_filings()
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    done = failed = 0
    for i, r in enumerate(t.itertuples(), 1):
        out = gz_path(r.cik, r.accession)
        if out.exists() or _old_cache_file(r.cik, r.accession) is not None:
            continue
        try:
            resp = throttled_get(s, r.url, timeout=120, sleep_between=sleep_between)
            resp.raise_for_status()
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            with gzip.open(tmp, "wb", compresslevel=6) as f:
                f.write(resp.content)
            tmp.replace(out)
            done += 1
        except Exception as e:  # keep going; rerun to retry
            failed += 1
            print(f"[fail] {r.cik} {r.accession}: {e!r}", flush=True)
        if i % 25 == 0:
            print(f"{i}/{len(t)} downloaded={done} failed={failed}", flush=True)
    print(f"done: {len(t)} filings, downloaded={done}, failed={failed}", flush=True)


# ---------------------------------------------------------------- extract

def _old_cache_file(cik: str, accession: str) -> Path | None:
    """The downloader's uncompressed cache: cache/{cik}/{0001234567-YY-NNNNNN}.htm."""
    d = CACHE_DIR / str(cik)
    if d.is_dir():
        for f in d.glob("*.htm"):
            if f.stem.replace("-", "").lstrip("0") == accession.lstrip("0"):
                return f
    return None


def read_html(cik: str, accession: str) -> str | None:
    gz = gz_path(cik, accession)
    if gz.exists():
        with gzip.open(gz, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")
    old = _old_cache_file(cik, accession)
    return old.read_text(encoding="utf-8", errors="ignore") if old else None


def _extract_one(args):
    cik, accession, report_date = args
    h = read_html(cik, accession)
    if h is None:
        return None
    report = pd.Timestamp(report_date) if isinstance(report_date, str) else None
    return [(accession, ctx, mat.strftime("%Y-%m-%d"), cell[:200])
            for ctx, (mat, cell) in extract_maturities(h, report).items()]


def extract(filings: pd.DataFrame | None = None, workers: int | None = None) -> pd.DataFrame:
    """filings: cik, accession, report_date (YYYY-MM-DD) per filing; default target_filings()."""
    from multiprocessing import Pool
    filings = target_filings() if filings is None else filings
    jobs = list(zip(filings["cik"], filings["accession"], filings["report_date"]))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with Pool(workers) as pool:
        results = pool.map(_extract_one, jobs, chunksize=4)
    missing = sum(r is None for r in results)
    rows = [x for r in results if r for x in r]
    out = pd.DataFrame(rows, columns=["accession", "context_id", "maturity_html", "maturity_cell"])
    print(f"{len(filings) - missing} filings read ({missing} without HTML), {len(out):,} contexts with a date")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if cmd == "download":
        ua = os.environ.get("SEC_USER_AGENT")
        if not ua:
            sys.exit("set SEC_USER_AGENT to 'Name email@domain' (the SEC requires a contact)")
        download(ua)
    elif cmd == "extract":
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        extract().to_csv(OUT_CSV, index=False)
        print(f"wrote {OUT_CSV}")
    else:
        sys.exit("usage: maturity_html.py download | extract")
