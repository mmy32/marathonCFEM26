"""
Augment SOFR.csv with EUR, GBP, CAD, AUD, SEK, JPY, CHF, NOK, KRW, DKK, CNY, NZD and SGD base
rates by calendar quarter.

Every series is pulled from the official publisher (or a US-government mirror of it)
and aggregated the same way as the SOFR column: the simple average of daily
observations within each calendar quarter.

  EURIBOR 3M : ECB Data Portal, FM.Q.U2.EUR.RT.MM.EURIBOR3MD_.HSTA
               (same dataflow/method as the SOFR series; already a quarterly average)
  SONIA      : FRED (St. Louis Fed) IUDSOIA, source Bank of England;
               falls back to the Bank of England IADB if FRED fails
  CORRA      : Bank of Canada Valet API, series AVG.INTWO
  BBSW 3M    : Reserve Bank of Australia table F1, series FIRMMBAB90D
  STIBOR 3M  : OECD Main Economic Indicators via FRED, IR3TIB01SEM156N (monthly averages)
  TONA       : Bank of Japan Time-Series API, FM01 STRDCLUCON (uncollateralized O/N call rate)
  SARON      : Swiss National Bank data portal, cube zirepo item H0 (close of trading)
  NIBOR 3M   : OECD Main Economic Indicators via FRED, IR3TIB01NOM156N (monthly averages)
  CD 91-day  : Bank of Korea ECOS, table 721Y001 item 2010000 (already a quarterly average)
  CIBOR 3M   : OECD Main Economic Indicators via FRED, IR3TIB01DKM156N (monthly averages)
  SHIBOR 3M  : National Interbank Funding Center (shibor.org / chinamoney.com.cn), PBoC-authorised
  BKBM 3M    : OECD Main Economic Indicators via FRED, IR3TIB01NZM156N (monthly averages)
  SORA 3M    : MAS via SingStat Table Builder, M700071 series 23 (compounded 3-month SORA,
               end of month); the quarter-end value compounds the quarter's overnight SORA

The benchmarks follow what the BDC filings tag (STIBOR, NIBOR, CIBOR, SARON, TONA and the Korean
short-term rate). STIBOR / NIBOR / CIBOR are licensed by their administrators (SFBF, NoRe, DFBF) and
no central bank republishes them, so the OECD monthly averages of the daily fixings are used; the
quarterly value is the mean of the three monthly averages (BKBM likewise). No filing tags a CNY benchmark; SHIBOR
3M is used as the interbank term rate analogous to EURIBOR 3M. ECOS is queried with its public
"sample" key (10 rows per call) unless ECOS_API_KEY is set.

Usage:
    pip install pandas requests
    python augment_rates.py SOFR.csv SOFR_augmented.csv
"""
import csv
import io
import os
import sys
import time

import pandas as pd
import requests

START = "2023-01-01"                    # panel starts 2023Q1 (preprocessor quarter_from)
UA = {"User-Agent": "Mozilla/5.0 (research data download)"}


def _get(url, tries=4, **kw):
    for i in range(tries):                     # FRED occasionally drops back-to-back requests
        try:
            # FRED refuses browser-like User-Agents, so its URLs go out with the requests default
            hdr = {} if "fred.stlouisfed.org" in url else UA
            r = requests.get(url, headers=hdr, timeout=60, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


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


def parse_boj(text: str) -> pd.Series:
    lines = text.splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("SERIES_CODE"))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    return pd.Series(df["VALUES"].values,
                     index=pd.to_datetime(df["SURVEY_DATES"].astype(str), format="%Y%m%d"))


def parse_snb(text: str, item: str = "H0") -> pd.Series:
    lines = text.lstrip("﻿").splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith('"Date"'))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), sep=";")
    df = df[df["D0"].eq(item)]
    return pd.Series(df["Value"].values, index=pd.to_datetime(df["Date"]))


def parse_ecos(rows: list) -> pd.Series:
    idx = pd.PeriodIndex([r["TIME"] for r in rows], freq="Q")
    return pd.Series([float(r["DATA_VALUE"]) for r in rows], index=idx)


def parse_shibor(js: dict, tenor: str = "3M") -> pd.Series:
    recs = js["records"]
    return pd.Series([r[tenor] for r in recs], index=pd.to_datetime([r["showDateCN"] for r in recs]))


def parse_singstat(js: dict, series_no: str = "23") -> pd.Series:
    """SingStat Table Builder JSON -> the series' monthly values indexed by month."""
    row = next(r for r in js["Data"]["row"] if str(r["seriesNo"]) == series_no)
    cols = row["columns"]
    return pd.Series([c["value"] for c in cols],
                     index=pd.to_datetime([c["key"] for c in cols], format="%Y %b"))


def quarter_end_value(s: pd.Series) -> pd.Series:
    """Value in each quarter's last month (a compounded-in-arrears rate at quarter end)."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[(s.index >= START) & s.index.month.isin([3, 6, 9, 12])]
    return pd.Series(s.values, index=s.index.to_period("Q"))


def quarterly_mean_of_monthly(s: pd.Series) -> pd.Series:
    """Mean of the monthly averages in each quarter; quarters missing a month are dropped."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s.index >= START]
    g = s.groupby(s.index.to_period("Q"))
    return g.mean()[g.count() == 3]


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


def _fred(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={START}"
    return parse_fred(_get(url).text)


def stibor3m():
    return quarterly_mean_of_monthly(_fred("IR3TIB01SEM156N"))


def nibor3m():
    return quarterly_mean_of_monthly(_fred("IR3TIB01NOM156N"))


def cibor3m():
    return quarterly_mean_of_monthly(_fred("IR3TIB01DKM156N"))


def bkbm3m():
    return quarterly_mean_of_monthly(_fred("IR3TIB01NZM156N"))


def sora3m():
    url = "https://tablebuilder.singstat.gov.sg/api/table/tabledata/M700071"
    return quarter_end_value(parse_singstat(_get(url).json(), "23"))


def tona():
    url = ("https://www.stat-search.boj.or.jp/api/v1/getDataCode?format=csv&lang=en"
           f"&db=FM01&code=STRDCLUCON&startDate={START[:4]}{START[5:7]}")
    return quarterly_mean(parse_boj(_get(url).text))


def saron():
    url = f"https://data.snb.ch/api/cube/zirepo/data/csv/en?fromDate={START}"
    return quarterly_mean(parse_snb(_get(url).text))


def cd91():
    key = os.environ.get("ECOS_API_KEY", "sample")
    end = pd.Timestamp.today().to_period("Q")
    rows, start = [], 1
    while True:                                  # the sample key returns at most 10 rows per call
        url = (f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/en/{start}/{start + 9}/"
               f"721Y001/Q/{pd.Period(START, 'Q')}/{end}/2010000")
        js = _get(url).json()["StatisticSearch"]
        rows += js["row"]
        start += 10
        if start > js["list_total_count"]:
            break
    return parse_ecos(rows)


def shibor3m():
    # The endpoint caps the date range, so request one quarter at a time.
    parts = []
    for q in pd.period_range(START, pd.Timestamp.today(), freq="Q"):
        url = ("https://www.chinamoney.com.cn/ags/ms/cm-u-bk-shibor/ShiborHis?lang=en"
               f"&startDate={q.start_time:%Y-%m-%d}&endDate={q.end_time:%Y-%m-%d}")
        parts.append(parse_shibor(_get(url).json()))
    return quarterly_mean(pd.concat(parts))


COLUMNS = {
    "EUR EURIBOR 3-month - ECB - Historical close, average of observations through period "
    "(FM.Q.U2.EUR.RT.MM.EURIBOR3MD_.HSTA)": euribor3m,
    "GBP Sterling Overnight Index Average SONIA - Bank of England via FRED - "
    "average of daily observations through period (IUDSOIA)": sonia,
    "CAD Canadian Overnight Repo Rate Average CORRA - Bank of Canada - "
    "average of daily observations through period (AVG.INTWO)": corra,
    "AUD Bank Bill Swap Rate BBSW 3-month - RBA - "
    "average of daily observations through period (F1 FIRMMBAB90D)": bbsw3m,
    "SEK Stockholm Interbank Offered Rate STIBOR 3-month - SFBF via OECD MEI (FRED) - "
    "average of monthly averages through period (IR3TIB01SEM156N)": stibor3m,
    "JPY Tokyo Overnight Average Rate TONA - Bank of Japan - "
    "average of daily observations through period (FM01 STRDCLUCON)": tona,
    "CHF Swiss Average Rate Overnight SARON - Swiss National Bank - "
    "average of daily observations through period (zirepo H0)": saron,
    "NOK Norwegian Interbank Offered Rate NIBOR 3-month - NoRe via OECD MEI (FRED) - "
    "average of monthly averages through period (IR3TIB01NOM156N)": nibor3m,
    "KRW Certificate of Deposit 91-day yield - Bank of Korea ECOS - "
    "quarterly average (721Y001 2010000)": cd91,
    "DKK Copenhagen Interbank Offered Rate CIBOR 3-month - DFBF via OECD MEI (FRED) - "
    "average of monthly averages through period (IR3TIB01DKM156N)": cibor3m,
    "CNY Shanghai Interbank Offered Rate SHIBOR 3-month - National Interbank Funding Center - "
    "average of daily observations through period (ShiborHis 3M)": shibor3m,
    "NZD Bank Bill Benchmark Rate BKBM 3-month - NZFMA via OECD MEI (FRED) - "
    "average of monthly averages through period (IR3TIB01NZM156N)": bkbm3m,
    "SGD Compounded Singapore Overnight Rate Average SORA 3-month - MAS via SingStat - "
    "value at quarter end, compounding the quarter (M700071 series 23)": sora3m,
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
