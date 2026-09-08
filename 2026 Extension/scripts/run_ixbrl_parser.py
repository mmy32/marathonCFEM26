from __future__ import annotations

import os
import sys
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[0]  # final/
sys.path.append(str(ROOT))

from paths import METADATA_CSV, IXBRL_WIDE_ALL_CSV
from ixbrl_parser import parse_accession_from_index_url
from ixbrl_parser import (
    IXBRLConfig,
    init_ixbrl_dirs,
    init_session,
    process_one_filing,
    append_wide_frames_bounded,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_ixbrl_parser")


@dataclass
class RunResult:
    processed: int = 0
    success: int = 0
    failed: int = 0


def load_parsed_keys_from_wide(out_csv: Path, chunksize: int = 10_000) -> set[tuple[str, str]]:
    if not out_csv.exists():
        return set()
    try:
        keys: set[tuple[str, str]] = set()
        # usecols does NOT avoid the cost of tokenizing every column on every row -
        # the C engine still parses full rows before dropping unwanted columns, so a
        # wide (hundreds of columns) file can OOM here even though only 2 columns are
        # kept. That silently happened on this file (270k rows x 790 cols) and made
        # resume_parsed_only think nothing had been parsed yet, re-parsing and
        # duplicating filings we already had. Reading in chunks bounds memory to one
        # chunk's full-width rows at a time instead of the whole file.
        for chunk in pd.read_csv(out_csv, usecols=["cik", "accession"], dtype=str, chunksize=chunksize):
            chunk = chunk.dropna(subset=["cik", "accession"])
            keys.update(map(tuple, chunk[["cik", "accession"]].values.tolist()))
        return keys
    except Exception as e:
        logger.error(f"Unable to read parsed keys from {out_csv}: {e}")
        raise


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
        mask = pd.Series(True, index=df.index)
        if date_from:
            mask &= fd >= pd.to_datetime(date_from)
        if date_to:
            mask &= fd <= pd.to_datetime(date_to)
        df = df[mask]

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
    flush_every: int = 100,
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

    # Filings come from hundreds of different BDC filers, each with its own XBRL
    # extension-taxonomy column names (prefix-merging into a common schema only
    # happens later, in preprocessor.py). Holding every parsed filing's frame in
    # memory for the whole run and concatenating once at the end does not scale:
    # a ~2,300-filing run OOM'd on the final concat after successfully parsing
    # everything. Flushing to disk periodically helped but still isn't enough on
    # this machine (~8GB RAM total) - a later flush OOM'd too, because
    # combine_wide_frames() reloads the *entire* accumulated file every call.
    # append_wide_frames_bounded() never loads the full file, so peak memory
    # stays bounded regardless of how large the combined table gets.
    wrote_any = False

    def _flush():
        nonlocal frames, wrote_any
        if not frames:
            return
        overwrite = (not wrote_any) and (not append_if_exists)
        append_wide_frames_bounded(frames, out_csv=out_csv, overwrite=overwrite)
        frames = []
        wrote_any = True

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

        # stats.processed, not the iterrows() label `i`: after resume-filtering,
        # df_meta's index keeps its original (pre-filter) labels rather than a clean
        # 0..N-1 range, so `i` is not a position counter and printed nonsense like
        # "Progress 2300/1739" once the label values ran past the true row count.
        if stats.processed % 50 == 0:
            logger.info(f"Progress {stats.processed}/{len(df_meta)} | success={stats.success} failed={stats.failed}")

        if len(frames) >= flush_every:
            logger.info(f"Flushing {len(frames)} parsed frames -> {out_csv} ...")
            _flush()

    if failures:
        fail_csv = out_csv.parent / "ixbrl_parse_failures.csv"
        pd.DataFrame(failures).to_csv(fail_csv, index=False)
        logger.warning(f"Wrote ixbrl parse failures -> {fail_csv} (n={len(failures)})")

    _flush()

    if not wrote_any:
        logger.warning("No successful parses; nothing to combine.")
        return stats

    logger.info(f"Wrote combined wide -> {out_csv}")
    logger.info(f"Done. processed={stats.processed} success={stats.success} failed={stats.failed}")
    return stats


def main():
    user_agent = "Cornell ORIE5220 Capstone sl3627@cornell.edu"
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
