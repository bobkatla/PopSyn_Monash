import pandas as pd
from typing import List, Literal


def convert_full_to_marg_count(
    full_pop: pd.DataFrame, filter_ls: list[str] = [], zone_field: str = "zone_id"
) -> pd.DataFrame:
    assert zone_field in full_pop.columns
    cols = [x for x in full_pop.columns if x not in filter_ls + [zone_field]]
    ls_temp_hold = []
    for att in cols:
        full_pop[att] = full_pop[att].astype(str)
        temp_hold = full_pop.groupby(zone_field)[att].value_counts().unstack().fillna(0)
        temp_hold.columns = [(temp_hold.columns.name, x) for x in temp_hold.columns]
        temp_hold = temp_hold.astype(int)
        ls_temp_hold.append(temp_hold)
    marg_new_raw = pd.concat(ls_temp_hold, axis=1)
    convert_marg_dict = {
        com_col: marg_new_raw[com_col] for com_col in marg_new_raw.columns
    }
    convert_marg_dict[(zone_field, None)] = marg_new_raw.index
    new_marg_hh = pd.DataFrame(convert_marg_dict)
    ls_drop_m = list(
        new_marg_hh.columns[new_marg_hh.columns.get_level_values(0).isin([zone_field])]
    )
    new_marg_hh = new_marg_hh.drop(columns=ls_drop_m)
    return new_marg_hh


def add_0_to_missing(
    df: pd.DataFrame, ls_missing: List[str], axis: Literal[0, 1]
) -> pd.DataFrame:
    for missing in ls_missing:
        if axis == 1:  # by row
            df.loc[missing] = 0
        elif axis == 0:  # by col
            df[missing] = 0
    return df


def impute_new_marg(
    converted_census_marg: pd.DataFrame, converted_new_hh_marg: pd.DataFrame
) -> pd.DataFrame:
    converted_census_marg.index = converted_census_marg.index.astype(str)
    converted_new_hh_marg.index = converted_new_hh_marg.index.astype(str)
    # make sure they both have the same rows and cols, if not it means 0
    missing_cols_ori = set(converted_new_hh_marg.columns) - set(
        converted_census_marg.columns
    )
    missing_cols_kept = set(converted_census_marg.columns) - set(
        converted_new_hh_marg.columns
    )
    missing_rows_ori = set(converted_new_hh_marg.index) - set(
        converted_census_marg.index
    )
    missing_rows_kept = set(converted_census_marg.index) - set(
        converted_new_hh_marg.index
    )

    converted_new_hh_marg = add_0_to_missing(
        converted_new_hh_marg, missing_cols_kept, 0
    )
    converted_new_hh_marg = add_0_to_missing(
        converted_new_hh_marg, missing_rows_kept, 1
    )
    converted_census_marg = add_0_to_missing(converted_census_marg, missing_cols_ori, 0)
    converted_census_marg = add_0_to_missing(converted_census_marg, missing_rows_ori, 1)
    return converted_new_hh_marg


def inflate_based_on_total(df: pd.DataFrame, weight_cols: str = "total") -> pd.DataFrame:
    assert weight_cols in df.columns, f"The dataframe must contain a '{weight_cols}' column to inflate based on."
    # Repeat rows based on 'total'
    df_repeated = df.loc[df.index.repeat(df[weight_cols])].reset_index(drop=True)

    # Drop the column
    df_repeated = df_repeated.drop(columns=weight_cols)
    return df_repeated