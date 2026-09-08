from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union, List, Dict, Tuple

import pandas as pd
import requests
import time

from paths import CACHE_DIR, METADATA_CSV, FAILURES_CSV


# =========================
# Small helpers
# =========================

def normalize_cik(cik: str) -> str:
    """Normalize to no-leading-zero numeric string (SEC url uses no-leading-zero in /data/{cik}/...)."""
    return str(int(str(cik).strip()))

def to_timestamp(x: Optional[Union[str, datetime]]) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    if pd.isna(x):
        return None
    return pd.to_datetime(x)

def accession_dir(acc: str) -> str:
    return str(acc).replace("-", "")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def throttled_get(session: requests.Session, url: str, timeout: int = 30, sleep_between: float = 0.11, **kwargs):
    time.sleep(sleep_between)
    return session.get(url, timeout=timeout, **kwargs)


# =========================
# Output layout
# =========================

def cik_cache_dir(cik: str) -> Path:
    """cache/{cik}/"""
    cik = normalize_cik(cik)
    return CACHE_DIR / cik

def local_primary_path(cik: str, accession: str, primary_doc: str) -> Path:
    """
    cache/{cik}/{accession}{ext}
    (keeps your old behavior)
    """
    ext = Path(str(primary_doc)).suffix
    return cik_cache_dir(cik) / f"{accession}{ext}"


# =========================
# SEC submissions
# =========================

def _get_json_with_retry(
    url: str,
    headers: dict,
    timeout: int = 30,
    sleep_between: float = 0.25,
    retry: int = 2,
) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(retry + 1):
        time.sleep(sleep_between)
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_exc = e
    raise last_exc


def fetch_all_submissions(
    cik: str,
    user_agent: str,
    sleep_between: float = 0.25,
    retry: int = 2,
) -> pd.DataFrame:
    """
    Fetch SEC submissions (recent + older fragments under filings.files) and return a DataFrame.
    """
    cik_norm = normalize_cik(cik)
    headers = {"User-Agent": user_agent}
    base_url = "https://data.sec.gov/submissions/"
    url = f"{base_url}CIK{cik_norm.zfill(10)}.json"

    data = _get_json_with_retry(url, headers, sleep_between=sleep_between, retry=retry)

    submission = pd.DataFrame(data["filings"]["recent"])

    # older fragments
    files = data.get("filings", {}).get("files", [])
    for f in files:
        file_url = f"{base_url}{f['name']}"
        frag = _get_json_with_retry(file_url, headers, sleep_between=sleep_between, retry=retry)
        # NOTE: SEC fragments format can vary; keep your current approach.
        sub_df = pd.DataFrame(frag)
        submission = pd.concat([submission, sub_df], ignore_index=True)

    return submission


# =========================
# Build metadata
# =========================

def build_metadata(
    submission: pd.DataFrame,
    cik: str,
    form_types: Optional[Union[str, Iterable[str]]] = None,
    date_from: Optional[Union[str, datetime]] = None,
    date_to: Optional[Union[str, datetime]] = None,
    index_ext: str = "htm",
) -> pd.DataFrame:
    """
    Convert raw submissions DF into a clean metadata DF, filtered by forms + filingDate range.
    """
    df = submission.copy()

    # parse dates
    if "filingDate" in df.columns:
        df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
    if "reportDate" in df.columns:
        df["reportDate"] = pd.to_datetime(df["reportDate"], errors="coerce")

    # forms filter
    if form_types is not None:
        if isinstance(form_types, str):
            form_types = [form_types]
        df = df[df["form"].isin(list(form_types))]

    # date filter (on filingDate)
    sd = to_timestamp(date_from)
    ed = to_timestamp(date_to)
    if sd is not None and "filingDate" in df.columns:
        df = df[df["filingDate"] >= sd]
    if ed is not None and "filingDate" in df.columns:
        df = df[df["filingDate"] <= ed]

    keep_cols = [c for c in [
        "accessionNumber", "filingDate", "reportDate", "form",
        "isXBRL", "isInlineXBRL", "primaryDocument"
    ] if c in df.columns]

    meta = df[keep_cols].copy()

    # urls
    cik_nolead = normalize_cik(cik)
    base = "https://www.sec.gov/Archives/edgar/data"

    meta["url"] = (
        f"{base}/{cik_nolead}/"
        + meta["accessionNumber"].apply(accession_dir)
        + "/"
        + meta["primaryDocument"].astype(str)
    )

    meta["index_url"] = (
        f"{base}/{cik_nolead}/"
        + meta["accessionNumber"].apply(accession_dir)
        + "/"
        + meta["accessionNumber"].astype(str)
        + f"-index.{index_ext}"
    )

    meta.insert(0, "cik", cik_nolead)

    if "filingDate" in meta.columns:
        meta = meta.sort_values("filingDate", ascending=False).reset_index(drop=True)

    return meta


# =========================
# Download
# =========================

@dataclass
class DownloadResult:
    downloaded: int
    skipped: int
    failed: int
    failures: pd.DataFrame  # cik, accessionNumber, url, error


def download_filing_htmls(
    metadata: pd.DataFrame,
    user_agent: str,
    overwrite: bool = False,
    timeout: int = 30,
    sleep_between: float = 0.11,
) -> DownloadResult:
    """
    Download primary HTML documents for rows in metadata into cache/{cik}/{accession}{ext}.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    downloaded = 0
    skipped = 0
    failed = 0
    failure_rows: List[Dict[str, str]] = []

    for _, row in metadata.iterrows():
        cik = str(row["cik"])
        acc = str(row["accessionNumber"]).strip()
        url = str(row["url"])
        primary_doc = str(row.get("primaryDocument", ""))

        out_dir = cik_cache_dir(cik)
        ensure_dir(out_dir)

        out_path = local_primary_path(cik, acc, primary_doc)

        if (not overwrite) and out_path.exists():
            skipped += 1
            continue

        try:
            resp = throttled_get(session, url, timeout=timeout, sleep_between=sleep_between)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            downloaded += 1
        except requests.RequestException as e:
            failed += 1
            failure_rows.append({
                "cik": cik,
                "accessionNumber": acc,
                "url": url,
                "error": repr(e),
            })

    failures_df = pd.DataFrame(failure_rows)
    return DownloadResult(downloaded, skipped, failed, failures_df)


# =========================
# Metadata store
# =========================

def load_existing_metadata(path: Path = METADATA_CSV) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()

def save_metadata(df: pd.DataFrame, path: Path = METADATA_CSV) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)

def append_and_dedup_metadata(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old.empty:
        out = new.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)
    # safer key
    if {"cik", "accessionNumber"}.issubset(out.columns):
        out = out.drop_duplicates(subset=["cik", "accessionNumber"])
    else:
        out = out.drop_duplicates()
    return out

def save_failures(df: pd.DataFrame, path: Path = FAILURES_CSV) -> None:
    if df is None or df.empty:
        return
    ensure_dir(path.parent)
    if path.exists() and path.stat().st_size > 0:
        try:
            old = pd.read_csv(path)
            out = pd.concat([old, df], ignore_index=True).drop_duplicates()
        except pd.errors.EmptyDataError:
            out = df
    else:
        out = df
    out.to_csv(path, index=False)


# =========================
# High-level convenience runner
# =========================

def download_for_cik(
    cik: str,
    user_agent: str,
    form_types: Optional[Iterable[str]] = ("10-K", "10-Q"),
    date_from: Optional[Union[str, datetime]] = None,
    date_to: Optional[Union[str, datetime]] = None,
    overwrite: bool = False,
    only_new: bool = True,
    update_metadata: bool = True,
    test_mode: bool = False,
) -> Tuple[pd.DataFrame, DownloadResult]:
    """
    High-level: fetch submissions -> build metadata -> (optionally) filter to only-new -> download.

    test_mode=True:
      - still downloads files, but forces update_metadata=False (no writes to cache/all_metadata.csv)
      - intended for quick smoke tests on a single CIK/time range
    """
    if test_mode:
        update_metadata = False

    # 1) fetch submissions
    submission = fetch_all_submissions(cik, user_agent=user_agent)

    # 2) build metadata for this cik
    meta = build_metadata(
        submission,
        cik=cik,
        form_types=form_types,
        date_from=date_from,
        date_to=date_to,
    )

    # 3) only-new filter
    if only_new and update_metadata:
        old = load_existing_metadata(METADATA_CSV)
        if not old.empty and {"cik", "accessionNumber"}.issubset(old.columns):
            old_keys = set(zip(old["cik"].astype(str), old["accessionNumber"].astype(str)))
            meta_keys = list(zip(meta["cik"].astype(str), meta["accessionNumber"].astype(str)))
            mask_new = [k not in old_keys for k in meta_keys]
            meta_to_download = meta.loc[mask_new].copy()
        else:
            meta_to_download = meta
    else:
        meta_to_download = meta

    # 4) download
    result = download_filing_htmls(meta_to_download, user_agent=user_agent, overwrite=overwrite)

    # 5) persist metadata/failures (optional)
    if update_metadata:
        old = load_existing_metadata(METADATA_CSV)
        full = append_and_dedup_metadata(old, meta)
        save_metadata(full, METADATA_CSV)

    save_failures(result.failures, FAILURES_CSV)

    return meta, result

def download_failures_only(
    failures_csv: Path = FAILURES_CSV,
    user_agent: str = "Your Name your.email@domain.com",
    overwrite: bool = True,
    timeout: int = 30,
    sleep_between: float = 0.11,
) -> DownloadResult:
    """
    Re-download only the failed (cik, accessionNumber, url) rows recorded in failures_csv.
    This bypasses submissions/metadata building and directly retries the exact failed URLs.
    """
    if not failures_csv.exists():
        return DownloadResult(downloaded=0, skipped=0, failed=0, failures=pd.DataFrame())

    df = pd.read_csv(failures_csv)
    if df.empty:
        return DownloadResult(downloaded=0, skipped=0, failed=0, failures=pd.DataFrame())

    # Build a minimal metadata-like DF needed by download_filing_htmls:
    # columns: cik, accessionNumber, url, primaryDocument
    # We don't know primaryDocument; infer ext from url if possible, else default ".htm".
    def _infer_primary_doc(url: str) -> str:
        # try to use url suffix
        suf = Path(str(url)).suffix
        return f"primary{suf if suf else '.htm'}"

    meta = pd.DataFrame({
        "cik": df["cik"].astype(str),
        "accessionNumber": df["accessionNumber"].astype(str),
        "url": df["url"].astype(str),
        "primaryDocument": df["url"].astype(str).apply(_infer_primary_doc),
    })

    # retry downloads
    result = download_filing_htmls(
        meta,
        user_agent=user_agent,
        overwrite=overwrite,
        timeout=timeout,
        sleep_between=sleep_between,
    )

    # overwrite failures file with *new* failures only (optional behavior)
    # If you prefer to append, comment this out.
    if result.failures is None or result.failures.empty:
        failures_csv.unlink(missing_ok=True)
    else:
        result.failures.to_csv(failures_csv, index=False)

    return result

