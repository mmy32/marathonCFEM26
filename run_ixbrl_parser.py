from __future__ import annotations

import os
import sys
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # final/
sys.path.append(str(ROOT))

from paths import METADATA_CSV, IXBRL_WIDE_ALL_CSV
from ixbrl_parser import parse_accession_from_index_url
from ixbrl_parser import (
    IXBRLConfig,
    init_ixbrl_dirs,
    init_session,
    process_one_filing,
    combine_wide_frames,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_ixbrl_parser")


@dataclass
class RunResult:
    processed: int = 0
    success: int = 0
    failed: int = 0


def load_parsed_keys_from_wide(out_csv: Path) -> set[tuple[str, str]]:
    if not out_csv.exists():
        return set()
    try:
        df = pd.read_csv(out_csv, usecols=["cik", "accession"], dtype=str, low_memory=False)
        df = df.dropna(subset=["cik", "accession"]).drop_duplicates()
        return set(map(tuple, df[["cik", "accession"]].values.tolist()))
    except Exception as e:
        logger.warning(f"Unable to read parsed keys from {out_csv}: {e}")
        return set()


def load_failures_list(fail_csv: Path) -> pd.DataFrame:
    if not fail_csv.exists():
        raise FileNotFoundError(f"resume_failed_only=True but failures file not found: {fail_csv}")
    df = pd.read_csv(fail_csv, dtype=str, low_memory=False)
    need = {"cik", "index_url", "reportDate"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"failures file missing required columns: {missing}")
    return df


def load_and_filter_metadata(
    metadata_csv: Path,
    cik_list: Optional[Iterable[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    need = {"cik", "index_url", "reportDate"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"metadata missing required columns: {missing}")

    if "isXBRL" in df.columns:
        df = df[df["isXBRL"].astype(int) == 1]
    if "isInlineXBRL" in df.columns:
        df = df[df["isInlineXBRL"].astype(int) == 1]

    if cik_list is not None:
        want = {str(c) for c in cik_list}
        df = df[df["cik"].astype(str).isin(want)]

    if (date_from or date_to) and "filingDate" in df.columns:
        fd = pd.to_datetime(df["filingDate"], errors="coerce")
        if date_from:
            df = df[fd >= pd.to_datetime(date_from)]
        if date_to:
            df = df[fd <= pd.to_datetime(date_to)]

    df = df.dropna(subset=["cik", "index_url", "reportDate"]).copy()

    if limit is not None and limit > 0:
        df = df.head(int(limit)).copy()

    return df.reset_index(drop=True)


def run_ixbrl_parse(
    user_agent: str,
    metadata_csv: Path = METADATA_CSV,
    out_csv: Path = IXBRL_WIDE_ALL_CSV,
    cik_list: Optional[Iterable[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: Optional[int] = None,
    append_if_exists: bool = False,
    resume_parsed_only: bool = True,
    resume_failed_only: bool = False,
    sleep_between: float = 0.6,
    retry: int = 2,
    request_timeout: int = 20,
    write_per_file_wide: bool = False,
    write_per_file_contexts: bool = False,
) -> RunResult:
    init_ixbrl_dirs()

    df_meta = load_and_filter_metadata(
        metadata_csv=metadata_csv,
        cik_list=cik_list,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
    )

    # ===== Resume logic =====
    if resume_failed_only:
        fail_csv = out_csv.parent / "ixbrl_parse_failures.csv"
        df_fail = load_failures_list(fail_csv)

        key_cols = ["cik", "index_url", "reportDate"]
        df_meta[key_cols] = df_meta[key_cols].astype(str)
        df_fail[key_cols] = df_fail[key_cols].astype(str)

        df_meta = df_meta.merge(df_fail[key_cols].drop_duplicates(), on=key_cols, how="inner")
        logger.info(f"resume_failed_only=True -> to process: {len(df_meta)} filings from failures list")

    elif resume_parsed_only:
        parsed_keys = load_parsed_keys_from_wide(out_csv)  # set[(cik, accession)]
        if parsed_keys:
            df_meta = df_meta.copy()
            df_meta["_cik"] = df_meta["cik"].astype(str)
            df_meta["_accession"] = df_meta["index_url"].astype(str).map(parse_accession_from_index_url)

            df_parsed = pd.DataFrame(list(parsed_keys), columns=["_cik", "_accession"])

            before = len(df_meta)
            tmp = df_meta.merge(df_parsed, on=["_cik", "_accession"], how="left", indicator=True)
            df_meta = tmp[tmp["_merge"] == "left_only"].drop(columns=["_merge", "_cik", "_accession"])
            logger.info(f"resume_parsed_only=True -> skipped {before - len(df_meta)} already-parsed filings")

    logger.info(f"To process: {len(df_meta)} filings")

    cfg = IXBRLConfig(
        user_agent=user_agent,
        request_timeout=request_timeout,
        sleep_between=sleep_between,
        retry=retry,
        write_raw_xml=True,
        write_per_file_wide=write_per_file_wide,
        write_per_file_contexts=write_per_file_contexts,
    )

    sess = init_session(user_agent=user_agent)

    frames = []
    failures = []
    stats = RunResult()

    for i, row in df_meta.iterrows():
        stats.processed += 1
        cik = str(row["cik"])

        try:
            df_wide, ok, status = process_one_filing(row, sess, cfg)
            if ok and df_wide is not None:
                frames.append(df_wide)
                stats.success += 1
            else:
                stats.failed += 1
                failures.append({"cik": cik, "index_url": row["index_url"], "reportDate": row["reportDate"], "status": status})
        except Exception as e:
            stats.failed += 1
            failures.append({"cik": cik, "index_url": row["index_url"], "reportDate": row["reportDate"], "status": f"exception:{e}"})

        time.sleep(cfg.sleep_between)

        if (i + 1) % 50 == 0:
            logger.info(f"Progress {i+1}/{len(df_meta)} | success={stats.success} failed={stats.failed}")

    if failures:
        fail_csv = out_csv.parent / "ixbrl_parse_failures.csv"
        pd.DataFrame(failures).to_csv(fail_csv, index=False)
        logger.warning(f"Wrote ixbrl parse failures -> {fail_csv} (n={len(failures)})")

    if not frames:
        logger.warning("No successful parses; nothing to combine.")
        return stats

    combined = combine_wide_frames(frames, out_csv=out_csv, append_if_exists=append_if_exists)
    logger.info(f"Wrote combined wide -> {out_csv} (rows={len(combined)}, cols={len(combined.columns)})")
    logger.info(f"Done. processed={stats.processed} success={stats.success} failed={stats.failed}")
    return stats


def main():
    user_agent = "Your Name your.email@domain.com"
    mode = "full"  # "test" | "full" | "retry_failed"

    if mode == "test":
        run_ixbrl_parse(
            user_agent=user_agent,
            cik_list=["0001287750"],
            date_from="2024-01-01",
            date_to="2024-12-31",
            limit=20,
            append_if_exists=False,
            resume_parsed_only=False,  # IMPORTANT in tests
            resume_failed_only=False,
            write_per_file_wide=False,
            write_per_file_contexts=False,
        )
    elif mode == "full":
        run_ixbrl_parse(
            user_agent=user_agent,
            append_if_exists=True,
            resume_parsed_only=True,
            resume_failed_only=False,
            write_per_file_wide=False,
            write_per_file_contexts=False,
        )
    elif mode == "retry_failed":
        run_ixbrl_parse(
            user_agent=user_agent,
            append_if_exists=True,
            resume_failed_only=True,
            resume_parsed_only=False,
            write_per_file_wide=False,
            write_per_file_contexts=False,
        )
    else:
        raise ValueError("mode must be one of: test | full | retry_failed")


if __name__ == "__main__":
    main()
