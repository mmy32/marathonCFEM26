"""BDC type by quarter (listed, non-traded or private) from each BDC's EDGAR filing history (cached in
cache/sec_submissions/) -> data/reference/bdc_type.csv. Rules in docs/6_final_dataset.md; cases the
rules cannot settle are in OVERRIDES. Run: python src/1_data_processing/bdc_type.py"""
from __future__ import annotations

import json
import sys
import time

import pandas as pd
import requests

from paths import CACHE_DIR, ENRICHED_DIR, REFERENCE_DIR

UA = {"User-Agent": "Cornell CFEM capstone sl3627@cornell.edu"}
SUB_DIR = CACHE_DIR / "sec_submissions"
OUT_CSV = REFERENCE_DIR / "bdc_type.csv"
MIN_PROSPECTUSES = 3
# cik -> (type, reason): replaces the rules for that BDC in every quarter
OVERRIDES: dict[int, tuple[str, str]] = {
    845385: ("listed", "Princeton Capital Corp: trades over the counter (PIAC), no exchange registration"),
    1495584: ("listed", "Firsthand Technology Value Fund: Nasdaq-listed, trades over the counter after its 2025 delisting"),
}


def _get(url: str) -> dict:
    for attempt in range(4):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(1 + attempt)
    r.raise_for_status()


def fetch(cik: int) -> pd.DataFrame:
    """All filings (form, date) of one CIK, cached as JSON. Also returns tickers/exchanges in attrs."""
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    path = SUB_DIR / f"CIK{cik:010d}.json"
    if not path.exists():
        main = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        pages = [_get(f"https://data.sec.gov/submissions/{f['name']}") for f in main["filings"].get("files", [])]
        time.sleep(0.15)
        path.write_text(json.dumps({"main": main, "pages": pages}))
    blob = json.loads(path.read_text())
    main = blob["main"]
    parts = [main["filings"]["recent"]] + blob["pages"]
    f = pd.concat([pd.DataFrame({"form": p["form"], "date": p["filingDate"]}) for p in parts], ignore_index=True)
    f["date"] = pd.to_datetime(f["date"])
    f.attrs.update(name=main["name"], tickers=main.get("tickers", []), exchanges=main.get("exchanges", []))
    return f


def classify(cik: int, f: pd.DataFrame) -> dict:
    forms = f.groupby("form")["date"].agg(["min", "max", "size"])
    first = lambda form: forms.loc[form, "min"] if form in forms.index else pd.NaT
    has_ticker = any(e for e in f.attrs["exchanges"] if e and e.upper() not in ("OTC", ""))
    delist = f[f["form"].isin(["25-NSE", "25"])]["date"]
    listing = first("8-A12B")
    was_listed = has_ticker or (pd.notna(listing) and len(delist) > 0)
    listed_from = listing if was_listed else pd.NaT
    if was_listed and pd.isna(listed_from):
        listed_from = pd.Timestamp("1900-01-01")          # listed before the filing history starts
    listed_to = pd.NaT
    if was_listed and not has_ticker and len(delist[delist > listed_from]):
        listed_to = delist[delist > listed_from].max()
    pre_end = listed_from if pd.notna(listed_from) else pd.Timestamp("2100-01-01")
    pre = f[f["date"] < pre_end]
    reg_n2 = pre.loc[pre["form"] == "N-2", "date"].min()
    is_prospectus = pre["form"].str.startswith("424B") | (pre["form"] == "497")   # 497: HMS Income Fund / MSIF
    prospectus = pre.loc[is_prospectus & (pre["date"] > reg_n2), "date"] if pd.notna(reg_n2) else pd.Series(dtype="datetime64[ns]")
    reg_10 = pre.loc[pre["form"] == "10-12G", "date"].min()
    non_traded_from = prospectus.min() if len(prospectus) >= MIN_PROSPECTUSES else pd.NaT
    if pd.notna(non_traded_from):
        why = f"N-2 {reg_n2.date()}, {len(prospectus)} prospectuses from {non_traded_from.date()}"
        if pd.notna(reg_10) and reg_10 < non_traded_from:
            why = f"Form 10 {reg_10.date()}; " + why
    elif pd.notna(reg_10) or pd.notna(reg_n2):
        why = (f"Form 10 {reg_10.date()}" if pd.notna(reg_10) else f"N-2 {reg_n2.date()}") + \
              f", {len(prospectus)} prospectuses: private offering"
    else:
        why = "no Form 10 or N-2 in the filing history"
    override = OVERRIDES.get(cik)
    return dict(cik=cik, name=f.attrs["name"], tickers=",".join(f.attrs["tickers"]),
                exchanges=",".join(e or "" for e in f.attrs["exchanges"]),
                listed_from=listed_from, listed_to=listed_to, non_traded_from=non_traded_from,
                override=override[0] if override else None,
                evidence=override[1] if override else ("listed; " if was_listed else "") + why,
                first_filing=f["date"].min())


def type_in_quarter(row: pd.Series, quarter_end: pd.Timestamp) -> str:
    if isinstance(row.get("override"), str):
        return row["override"]
    listed = pd.notna(row["listed_from"]) and row["listed_from"] <= quarter_end and (
        pd.isna(row["listed_to"]) or quarter_end < row["listed_to"])
    if listed:
        return "listed"
    return "non-traded" if pd.notna(row["non_traded_from"]) and row["non_traded_from"] <= quarter_end else "private"


def build(ciks) -> pd.DataFrame:
    out = []
    for i, cik in enumerate(sorted(ciks)):
        out.append(classify(cik, fetch(cik)))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ciks)}")
    return pd.DataFrame(out)


if __name__ == "__main__":
    # step 7 output: same BDCs as FINAL_DATASET, and it exists before step 8 needs this table
    ciks = pd.read_csv(ENRICHED_DIR / "data_private_credit_FINAL_enriched_final.csv", usecols=["cik"])["cik"].unique()
    print(f"Fetching EDGAR filing histories for {len(ciks)} BDCs (cached in {SUB_DIR}) ...")
    table = build(ciks)
    table.to_csv(OUT_CSV, index=False)
    print(table.assign(type_2026Q2=[type_in_quarter(r, pd.Timestamp("2026-06-30")) for _, r in table.iterrows()])
          ["type_2026Q2"].value_counts().to_string())
    print(f"Wrote {OUT_CSV}")
    sys.exit(0)
