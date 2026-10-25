import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from ixbrl_utils import (
    # --- core cleaning ---
    normalize_interest_columns,
    convert_currencies,
    initialize_clean_dataframe,
    classify_rate_types,
    perform_initial_swap,
    clean_fixed_rate_data,
    perform_additional_corrections,
    clean_additional_rate_issues,
    fix_pic_from_ir_minus_pik,
    fix_ir_lt_pic,
    apply_spread_rate_rules,
    drop_fully_missing_rate_rows,

    # --- unresolved logic ---
    unresolved_rate_pic,
    unresolved_rate_pik_pic,
    unresolved_pik,
    unresolved_spread_rate_pic_pik,
    unresolved_pic,
    unresolved_spread_pik_pic,
    unresolved2_spread_rate_pik,
    unresolved_spread_rate_pic,
    unresolved_spread,
    unresolved_spread_rate,
    unresolved_spread_pik,
    unresolved_spread_pic,
    unresolved_spread_rate_pik,


    # --- SOFR & helpers ---
    determine_currency,
    add_estimate_with_sofr,
    percentile_range,
    RATE,
    SPREAD,
    add_check_2
)


def run_pipeline(
    data_path: str,
    fx_path: str,
    sofr_path: str = "SOFR.csv",
) -> pd.DataFrame:

    # -------------------------
    # Load base data
    # -------------------------
    df = pd.read_csv(data_path)

    df["cal_qe"] = pd.to_datetime(df["cal_qe"], errors="coerce")
    df["cal_q"] = df["cal_qe"].dt.to_period("Q").astype(str)
    
    # Filter rows without shares but with meaningful term info
    df = df.loc[df["InvestmentOwnedBalanceShares"].isna()]
    df = df.loc[~df["context_type"].isin(["amounts_only", "empty"])]

    # -------------------------
    # Normalize & classify
    # -------------------------
    df = normalize_interest_columns(df)

    df["RateType"] = df[
        "InvestmentVariableInterestRateTypeExtensibleEnumeration"
    ].apply(lambda x: x.split("#")[-1] if isinstance(x, str) and "#" in x else x)

    df = convert_currencies(df, fx_path)

    df = initialize_clean_dataframe(df)
    df = classify_rate_types(df)

    # -------------------------
    # Core cleaning passes
    # -------------------------
    df = perform_initial_swap(df)
    df = clean_fixed_rate_data(df)
    df = perform_additional_corrections(df)
    df = clean_additional_rate_issues(df)
    df = fix_pic_from_ir_minus_pik(df)
    df = fix_ir_lt_pic(df)

    # -------------------------
    # Unresolved rate logic
    # -------------------------
    df = apply_spread_rate_rules(df)
    df = drop_fully_missing_rate_rows(df)

    df = add_check_2(df)

    df = unresolved_rate_pic(df)
    df = unresolved_rate_pik_pic(df)
    df = unresolved_pik(df)
    df = unresolved_spread_rate_pic_pik(df)
    df = unresolved_pic(df)
    df = unresolved_spread_pik_pic(df)
    df = unresolved_spread_rate_pik(df)
    df = unresolved_spread_rate_pic(df)

    # -------------------------
    # SOFR + estimation
    # -------------------------
    sofr = pd.read_csv(sofr_path)
    sofr.rename(columns={sofr.columns[2]: "sofr"}, inplace=True)
    sofr["TIME PERIOD"] = sofr["TIME PERIOD"].astype(str).str.strip()
    sofr["sofr"] = sofr["sofr"] / 100

    df["currency"] = df.apply(determine_currency, axis=1)

    rate_range = percentile_range(df[RATE], 0.05)
    spread_range = percentile_range(df[SPREAD], 0.05)

    df = add_estimate_with_sofr(
        df,
        sofr_df=sofr,
        rate_range=rate_range,
        spread_range=spread_range,
    )

    # -------------------------
    # Final unresolved handling
    # -------------------------
    df = unresolved_spread(df)
    df = unresolved_spread_rate(df)
    df = unresolved_spread_pik(df)
    df = unresolved_spread_pic(df)
    df = unresolved2_spread_rate_pik(df)

    #print(df["check_1"].value_counts(dropna=False))
    return df


# if __name__ == "__main__":
#     df_clean = run_pipeline(
#         data_path="ixbrl_clean.csv",
#         fx_path="FX.csv",
#         sofr_path="SOFR.csv",
#     )

#     df_clean.to_csv("ixbrl_cleaned_out.csv", index=False)