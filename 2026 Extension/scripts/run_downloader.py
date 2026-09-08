import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # 2026 Extension/
sys.path.append(str(ROOT / "src"))

from filings import download_for_cik, download_failures_only
from paths import PROCESSED_DIR


def run_download(
    user_agent: str,
    mode: str = "full",  # "test" | "full" | "retry_failed"
    form_types=("10-K", "10-Q"),
    overwrite: bool = False,
):
    intervals_csv = PROCESSED_DIR / "BDC_intervals.csv"

    if mode == "retry_failed":
        result = download_failures_only(
            user_agent=user_agent,
            overwrite=True,  # retry usually forces overwrite
        )
        print(f"[retry_failed] downloaded={result.downloaded}, skipped={result.skipped}, failed={result.failed}")
        if result.failed:
            print(result.failures.head())
        return result

    if mode == "test":
        cik = "0001287750"
        date_from = "2024-01-01"
        date_to = "2024-12-31"

        meta, result = download_for_cik(
            cik=cik,
            user_agent=user_agent,
            form_types=form_types,
            date_from=date_from,
            date_to=date_to,
            overwrite=overwrite,
            only_new=False,
            update_metadata=False,  # test run: don't update all_metadata.csv
            test_mode=True,
        )
        print(f"[TEST {cik}] metadata rows={len(meta)} | downloaded={result.downloaded}, skipped={result.skipped}, failed={result.failed}")
        if result.failed:
            print(result.failures.head())
        return result

    if mode != "full":
        raise ValueError("mode must be one of: test | full | retry_failed")

    # ===== full universe =====
    if not intervals_csv.exists():
        raise FileNotFoundError(f"Missing intervals file: {intervals_csv}")

    intervals = pd.read_csv(intervals_csv)
    cik_col = "CIK" if "CIK" in intervals.columns else "cik"
    intervals[cik_col] = intervals[cik_col].astype(str)

    if "start_date" not in intervals.columns or "end_date" not in intervals.columns:
        raise KeyError("BDC_intervals.csv must contain start_date and end_date columns")

    intervals["start_date"] = pd.to_datetime(intervals["start_date"], errors="coerce")
    intervals["end_date"] = pd.to_datetime(intervals["end_date"], errors="coerce")

    groups = list(intervals.groupby(cik_col))
    print(f"[FULL] total CIKs={len(groups)} from {intervals_csv}")

    cik_failures = []
    for i, (cik, g) in enumerate(groups, 1):
        start = g["start_date"].min()
        end = g["end_date"].max()  # NaT => active

        print(f"\n[{i}/{len(groups)}] CIK={cik} | window: {start} -> {end}")

        try:
            meta, result = download_for_cik(
                cik=cik,
                user_agent=user_agent,
                form_types=form_types,
                date_from=start if pd.notna(start) else None,
                date_to=end if pd.notna(end) else None,
                overwrite=overwrite,
                only_new=True,
                update_metadata=True,
                test_mode=False,
            )
            print(f"[{cik}] metadata rows={len(meta)} | downloaded={result.downloaded}, skipped={result.skipped}, failed={result.failed}")
        except Exception as e:
            print(f"[{cik}] CIK-level failure, skipping: {e!r}")
            cik_failures.append({"cik": cik, "error": repr(e)})

    if cik_failures:
        fail_path = PROCESSED_DIR / "cik_level_download_failures.csv"
        pd.DataFrame(cik_failures).to_csv(fail_path, index=False)
        print(f"\n[FULL] {len(cik_failures)} CIK-level failures written to {fail_path}")

    print("\n[FULL] done.")
    return None


def main():
    user_agent = "Cornell ORIE5220 Capstone sl3627@cornell.edu"
    mode = "full"  # "test" | "full" | "retry_failed"
    run_download(user_agent=user_agent, mode=mode)


if __name__ == "__main__":
    main()
