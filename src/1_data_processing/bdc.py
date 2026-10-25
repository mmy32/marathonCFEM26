from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import requests


# =========================
# Paths / outputs
# =========================

@dataclass(frozen=True)
class BDCPaths:
    """Output locations for BDC universe build."""
    out_dir: Path

    @property
    def filings_csv(self) -> Path:
        return self.out_dir / "BDC_filings_2001_present.csv"

    @property
    def intervals_csv(self) -> Path:
        return self.out_dir / "BDC_intervals.csv"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =========================
# master.idx -> filings
# =========================

HEADER_LINE = "CIK|Company Name|Form Type|Date Filed|Filename"


def _compile_form_pattern(forms_regex: str, include_amendments: bool) -> re.Pattern:
    """
    Compile the form matching regex. If include_amendments=True and the regex is the default,
    accept optional '/A' suffix.
    """
    if include_amendments and forms_regex == r"^(N-54A|N-54C)$":
        forms_regex = r"^(N-54A|N-54C)(/A)?$"
    return re.compile(forms_regex, re.IGNORECASE)


def _fetch_master_idx_text(
    session: requests.Session,
    url: str,
    timeout: int,
    sleep_between: float = 0.25,
    retry: int = 2,
) -> Optional[str]:
    for attempt in range(retry + 1):
        time.sleep(sleep_between)
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200 and r.text:
                return r.text
        except requests.RequestException:
            pass
    return None


def _find_data_start(lines: List[str], header_line: str = HEADER_LINE) -> int:
    """Return the index where data rows start (after header line)."""
    for i, line in enumerate(lines):
        if line.strip().startswith(header_line):
            return i + 1
    return 0


def _parse_master_idx_lines(
    lines: List[str],
    form_pat: re.Pattern,
    archives_prefix: str,
) -> List[Dict[str, str]]:
    """Parse master.idx lines into rows for matching forms."""
    start_idx = _find_data_start(lines)
    out: List[Dict[str, str]] = []

    for line in lines[start_idx:]:
        parts = line.split("|")
        if len(parts) < 5:
            continue

        cik, company, form, date, filename = [p.strip() for p in parts[:5]]
        if form_pat.match(form):
            out.append(
                {
                    "Company": company,
                    "CIK": cik,
                    "Form": form,
                    "Date": date,
                    "Link": archives_prefix + filename,
                }
            )
    return out


def _finalize_filings_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and deduplicate filings DataFrame."""
    if df.empty:
        return df

    df = df.copy()
    df["CIK"] = df["CIK"].astype(str).str.zfill(10)
    df = (
        df.drop_duplicates(subset=["CIK", "Form", "Date", "Link"])
        .sort_values(["Date", "CIK"])
        .reset_index(drop=True)
    )
    return df


def build_bdc_filings_from_masteridx(
    start_year: int = 2001,
    end_year: int = 2026,
    user_agent: str = "Your Name your.email@domain.com",
    timeout: int = 30,
    forms_regex: str = r"^(N-54A|N-54C)$",
    include_amendments: bool = False,
    base_url: str = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx",
    archives_prefix: str = "https://www.sec.gov/Archives/",
    out_dir: Optional[str | Path] = None,
    out_csv: Optional[str | Path] = None,
    sleep_between: float = 0.25,
    retry: int = 2,
) -> pd.DataFrame:
    """
    Build a raw panel of BDC status filings (N-54A / N-54C) by scanning EDGAR full-index master.idx files.

    Output control:
    - If out_dir is provided, writes to <out_dir>/BDC_filings_2001_present.csv
    - If out_csv is provided, writes exactly to that path
    - If both provided, out_csv takes precedence
    """
    # Decide output path (optional)
    out_path: Optional[Path] = None
    if out_csv is not None:
        out_path = Path(out_csv)
        ensure_dir(out_path.parent)
    elif out_dir is not None:
        p = BDCPaths(Path(out_dir))
        ensure_dir(p.out_dir)
        out_path = p.filings_csv

    headers = {"User-Agent": user_agent}
    form_pat = _compile_form_pattern(forms_regex, include_amendments)

    rows: List[Dict[str, str]] = []
    session = requests.Session()
    session.headers.update(headers)

    for year in range(start_year, end_year + 1):
        for q in range(1, 5):
            url = base_url.format(year=year, q=q)
            text = _fetch_master_idx_text(session, url, timeout, sleep_between=sleep_between, retry=retry)
            if text is None:
                continue

            lines = text.splitlines()
            rows.extend(_parse_master_idx_lines(lines, form_pat, archives_prefix))

    df = _finalize_filings_df(pd.DataFrame(rows))

    if out_path is not None:
        df.to_csv(out_path, index=False)

    return df


# =========================
# filings -> intervals
# =========================

def _normalize_form(x: object) -> str:
    """Normalize form string for interval logic (handles case/whitespace)."""
    return str(x).strip().upper()


def build_bdc_intervals_from_filings(
    df_filings: pd.DataFrame,
    out_dir: Optional[str | Path] = None,
    out_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Convert N-54A/N-54C filings into continuous BDC status intervals.

    Output control:
    - If out_dir is provided, writes to <out_dir>/BDC_intervals.csv
    - If out_csv is provided, writes exactly to that path
    - If both provided, out_csv takes precedence
    """
    # Decide output path (optional)
    out_path: Optional[Path] = None
    if out_csv is not None:
        out_path = Path(out_csv)
        ensure_dir(out_path.parent)
    elif out_dir is not None:
        p = BDCPaths(Path(out_dir))
        ensure_dir(p.out_dir)
        out_path = p.intervals_csv

    df = df_filings.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    rows = []
    for cik, g in df.sort_values(["CIK", "Date"]).groupby("CIK"):
        company = None
        if "Company" in g.columns:
            s = g["Company"].dropna()
            company = s.iloc[0] if not s.empty else None

        start_date, start_link = None, None

        for _, r in g.iterrows():
            f = _normalize_form(r.get("Form", ""))

            if f == "N-54A":
                start_date = r["Date"]
                start_link = r.get("Link", None)

            elif f == "N-54C":
                end_date = r["Date"]
                end_link = r.get("Link", None)

                if start_date is not None:
                    rows.append(
                        {
                            "CIK": cik,
                            "Company": company,
                            "start_date": start_date,
                            "end_date": end_date,
                            "Link_A": start_link,
                            "Link_C": end_link,
                        }
                    )
                    start_date, start_link = None, None
                else:
                    rows.append(
                        {
                            "CIK": cik,
                            "Company": company,
                            "start_date": pd.NaT,
                            "end_date": end_date,
                            "Link_A": None,
                            "Link_C": end_link,
                        }
                    )

        # still active
        if start_date is not None:
            rows.append(
                {
                    "CIK": cik,
                    "Company": company,
                    "start_date": start_date,
                    "end_date": pd.NaT,
                    "Link_A": start_link,
                    "Link_C": None,
                }
            )

    out = pd.DataFrame(rows)

    # Keep stable ordering (helps diff/debug)
    if not out.empty:
        out = out.sort_values(["CIK", "start_date", "end_date"], na_position="last").reset_index(drop=True)

    if out_path is not None:
        out.to_csv(out_path, index=False)

    return out
