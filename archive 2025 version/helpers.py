import pandas as pd

def standardize_periods_to_calendar_quarter(periods, return_type="date"):
    s = pd.Series(pd.to_datetime(periods, errors="coerce"))
    qe = s.apply(lambda x: pd.NaT if pd.isna(x) else pd.offsets.QuarterEnd().rollforward(x).normalize())

    if return_type == "date":
        return qe
    q = qe.dt.to_period("Q")          # PeriodIndex like 2025Q2
    if return_type == "period":
        return q
    if return_type == "label":
        return q.astype(str)          # '2025Q2'
    raise ValueError("return_type must be one of {'date','label','period'}")


def add_calendar_quarter_columns(df, period_col="period",
                                 out_date_col="cal_qe",
                                 out_label_col="cal_q",
                                 drop_original=False):
    """
    Add standardized calendar quarter columns to a DataFrame.
    """
    df = df.copy()
    df[out_date_col]  = standardize_periods_to_calendar_quarter(df[period_col], "date")
    df[out_label_col] = standardize_periods_to_calendar_quarter(df[period_col], "label")
    if drop_original:
        df = df.drop(columns=[period_col])
    return df