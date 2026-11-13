"""Loan-quarter panel on tranche chains (tranches.build_tranche_panel): PIK transitions,
dispersion (2+ families) and borrower crowding. The input to the warning signs, the
borrower-quarter panel and the case studies."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_data_processing"))
from paths import add_code_dirs_to_path  # noqa: E402
add_code_dirs_to_path()

import tranches as T  # noqa: E402
import val_dispersion as VD  # noqa: E402
from manager_family import add_family  # noqa: E402
from index_construction import load_and_prepare_investment_data, compute_final_interest_rates  # noqa: E402
from pik_metrics import classify_loan_pik_transitions  # noqa: E402
from concentration_crowding import compute_borrower_crowding_metrics  # noqa: E402

# catch-all sector labels merged into one, in `sector` and `sector_report`
SECTOR_LABEL_MERGE = {"Other / Unknown / Unresolved": "Other / Unclassified", "Other / Unknown": "Other / Unclassified"}
RATE_CEILING = 1.0   # rates above 100% are nulled, never the row (none today: step 6 caps them at 50%)


def load_rows(final_csv) -> pl.DataFrame:
    """FINAL_DATASET rows with final rates, merged sector labels and manager family. Implausible
    rates are nulled before compute_final_interest_rates(), which can then derive them from the
    other two rate fields."""
    df = load_and_prepare_investment_data(str(final_csv))
    df = df.with_columns([pl.when(pl.col(c).abs() > RATE_CEILING).then(None).otherwise(pl.col(c)).alias(c)
                          for c in ("rate_cash", "interest_rate", "rate_pik") if c in df.columns])
    for col in ("sector", "sector_report"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).replace(SECTOR_LABEL_MERGE))
    return add_family(compute_final_interest_rates(df))


def _add_pik(tr: pd.DataFrame) -> pd.DataFrame:
    p = classify_loan_pik_transitions(pl.from_pandas(
        tr[["cik", "chain_id", "cal_q", "PIK_Final", "IR_Final", "PAR", "pik_at_origination"]]
        .assign(tranche_id=tr["chain_id"].astype(str), cik=tr["cik"].astype(str))
    )).to_pandas()
    p = p[["chain_id", "cal_q", "pik_share", "is_distress_amendment", "pik_acquired"]]
    return tr.merge(p, on=["chain_id", "cal_q"], how="left")


def _add_dispersion(tr: pd.DataFrame) -> pd.DataFrame:
    disp = VD.tranche_dispersion(tr, min_families=2, max_tier=2)
    fam = VD.family_marks_vs_consensus(tr, disp=disp)
    tr = tr.merge(disp[["cal_q", "xb_id", "V1_std", "V1_range", "V2_low_mark"]], on=["cal_q", "xb_id"], how="left")
    tr = tr.merge(fam[["cal_q", "xb_id", "family", "mark_vs_consensus"]], on=["cal_q", "xb_id", "family"], how="left")
    tr["mark_distance"] = tr["mark_vs_consensus"].abs()
    return tr


def _add_crowding(tr: pd.DataFrame, rows: pl.DataFrame) -> pd.DataFrame:
    c = compute_borrower_crowding_metrics(rows).to_pandas()
    c = c[["cal_q", "borrower", "n_families", "lender_hhi", "top_share", "d_families_4q", "log_par"]].rename(columns={
        "n_families": "C1_families", "lender_hhi": "C2_lender_hhi", "top_share": "C3_top_share",
        "d_families_4q": "C4_d_families", "log_par": "log_borrower_par"})
    tr["_b"] = tr["borrower_key"].str.strip().str.upper()
    return tr.merge(c, left_on=["cal_q", "_b"], right_on=["cal_q", "borrower"], how="left").drop(columns=["_b", "borrower"])


def build_monitor_panel(final_csv) -> tuple[pd.DataFrame, pl.DataFrame]:
    """Returns (tranche-quarter panel with signals, the row-level frame)."""
    rows = load_rows(final_csv)
    tr = T.build_tranche_panel(rows).sort_values(["chain_id", "qsort"]).reset_index(drop=True)
    tr = _add_pik(tr)
    tr = _add_dispersion(tr)
    tr = _add_crowding(tr, rows)
    return tr, rows
