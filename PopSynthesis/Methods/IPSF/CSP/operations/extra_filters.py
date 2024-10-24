"""Extra filter to help process pool"""


import pandas as pd
from typing import List


def filter_mismatch_hhsz(
    df: pd.DataFrame, hhsz_col: str, sub_cols: List[str], max_hhsz: int = 8
) -> pd.DataFrame:
    """Filter the df to rm mismatch hhsize"""
    # NOTE: assume max is 8+
    temp_col = "syn_hhsz"
    df[temp_col] = df[sub_cols].sum(axis=1)
    df[temp_col] = df[temp_col].apply(
        lambda x: str(x) if x < max_hhsz else f"{max_hhsz}+"
    )
    df = df[df[hhsz_col] == df[temp_col]].drop(columns=[temp_col])
    return df


def filter_paired_pool_agegr(
    pool: pd.DataFrame, agegr_col_younger: str, agegr_col_older: str, min_gap: int
) -> pd.DataFrame:
    """Filter the pool to have the agegr gap"""
    # convert the agegr to int
    pool["younger"] = pool[agegr_col_younger].apply(
        lambda x: int(x.split("-")[0].replace("+", ""))
    )
    pool["older"] = pool[agegr_col_older].apply(
        lambda x: int(x.split("-")[0].replace("+", ""))
    )
    pool = pool[pool["older"] - pool["younger"] >= min_gap].drop(
        columns=["younger", "older"]
    )
    return pool
