"""
Augment SOFR.csv with EURIBOR (3M), SONIA, CORRA and BBSW (3M) quarterly averages.

Every series is pulled from the official publisher (or a US-government mirror of it)
and aggregated the same way as the SOFR column: the simple average of daily
observations within each calendar quarter.

  EURIBOR 3M : ECB Data Portal, FM.Q.U2.EUR.RT.MM.EURIBOR3MD_.HSTA
               (same dataflow/method as the SOFR series; already a quarterly average)
  SONIA      : FRED (St. Louis Fed) IUDSOIA, source Bank of England;
               falls back to the Bank of England IADB if FRED fails
  CORRA      : Bank of Canada Valet API, series AVG.INTWO
  BBSW 3M    : Reserve Bank of Australia table F1, series FIRMMBAB90D

Usage:
    pip install pandas requests
    python augment_rates.py SOFR.csv SOFR_augmented.csv
"""
import csv
import io
import sys

import pandas as pd
import requests

START = "2023-01-01"                    # panel starts 2023Q1 (preprocessor quarter_from)
UA = {"User-Agent": "Mozilla/5.0 (research data download)"}


def _get(url, **kw):
    r = requests.get(url, headers=UA, timeout=60, **kw)
    r.raise_for_status()
    return r


def quarterly_mean(s: pd.Series) -> pd.Series:
    """Average of daily observations per calendar quarter, indexed by Period[Q]."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s.index >= START]
    return s.groupby(s.index.to_period("Q")).mean()


# ---------- parsers (separate from fetching so they can be tested offline) ----------

def parse_ecb(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    idx = pd.PeriodIndex(df["TIME_PERIOD"].str.replace("-", ""), freq="Q")
    return pd.Series(df["OBS_VALUE"].astype(float).values, index=idx)


def parse_fred(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", "value"]
    return pd.Series(df["value"].values, index=pd.to_datetime(df["date"]))


def parse_boe(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    return pd.Series(df.iloc[:, 1].values,
                     index=pd.to_datetime(df.iloc[:, 0], format="%d %b %Y"))


def parse_boc(js: dict) -> pd.Series:
    obs = js["observations"]
    return pd.Series([o["AVG.INTWO"]["v"] for o in obs],
                     index=pd.to_datetime([o["d"] for o in obs]))


def parse_rba(text: str, series_id: str = "FIRMMBAB90D") -> pd.Series:
    # The file opens with a one-field title line, so pandas would size every row to 1 column;
    # read with csv (ragged rows allowed) and pad to the widest row instead.
    rows = list(csv.reader(io.StringIO(text)))
    width = max(len(r) for r in rows)
    raw = pd.DataFrame([r + [None] * (width - len(r)) for r in rows], dtype=str)
    hdr = raw.index[raw.iloc[:, 0].str.strip().eq("Series ID")][0]
    cols = raw.iloc[hdr].str.strip().tolist()
    data = raw.iloc[hdr + 1:].copy()
    data.columns = cols
    data = data[data["Series ID"].notna() & data[series_id].notna()]
    idx = pd.to_datetime(data["Series ID"].str.strip(), format="%d-%b-%Y")
    return pd.Series(data[series_id].values, index=idx)


# ---------- fetchers ----------

def euribor3m():
    url = ("https://data-api.ecb.europa.eu/service/data/FM/"
           "Q.U2.EUR.RT.MM.EURIBOR3MD_.HSTA?format=csvdata&startPeriod=2023-Q1")
    return parse_ecb(_get(url).text)


def sonia():
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=IUDSOIA&cosd={START}"
        return quarterly_mean(parse_fred(_get(url).text))
    except Exception as e:  # fall back to the Bank of England directly
        print(f"  FRED failed ({e}); using Bank of England IADB")
        url = ("https://www.bankofengland.co.uk/boeapps/database/"
               "_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/2023&Dateto=now"
               "&SeriesCodes=IUDSOIA&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")
        return quarterly_mean(parse_boe(_get(url).text))


def corra():
    url = f"https://www.bankofcanada.ca/valet/observations/AVG.INTWO/json?start_date={START}"
    return quarterly_mean(parse_boc(_get(url).json()))


def bbsw3m():
    url = "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv"
    return quarterly_mean(parse_rba(_get(url).content.decode("latin-1")))


COLUMNS = {
    "EUR EURIBOR 3-month - ECB - Historical close, average of observations through period "
    "(FM.Q.U2.EUR.RT.MM.EURIBOR3MD_.HSTA)": euribor3m,
    "GBP Sterling Overnight Index Average SONIA - Bank of England via FRED - "
    "average of daily observations through period (IUDSOIA)": sonia,
    "CAD Canadian Overnight Repo Rate Average CORRA - Bank of Canada - "
    "average of daily observations through period (AVG.INTWO)": corra,
    "AUD Bank Bill Swap Rate BBSW 3-month - RBA - "
    "average of daily observations through period (F1 FIRMMBAB90D)": bbsw3m,
}


def augment(sofr_path, out_path, fetchers=COLUMNS):
    base = pd.read_csv(sofr_path)
    base = base[pd.PeriodIndex(base["TIME PERIOD"], freq="Q") >= pd.Period(START, "Q")].reset_index(drop=True)
    q = pd.PeriodIndex(base["TIME PERIOD"], freq="Q")
    for name, fn in fetchers.items():
        print(f"Fetching {name.split(' - ')[0]} ...")
        s = fn()
        base[name] = s.reindex(q).round(4).values  # keep SOFR's quarters only
        missing = base.loc[base[name].isna(), "TIME PERIOD"].tolist()
        if missing:
            print(f"  WARNING: no data for {missing}")
    base.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(base.drop(columns="DATE").to_string(index=False))


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "SOFR.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "SOFR_augmented.csv"
    augment(src, dst)
