"""
Check synthetic population with census

Input will be synthetic pop and its related census
This remove the least part we can to make it correct (smaller than census)
This require func to convert synthetic to census form (agg func)
Then func to detect which part we can remove
Output will be part of synthetic we keep and part we want to remove
"""

import pandas as pd
import pandas as pd
import numpy as np
from itertools import combinations, product
from PopSynthesis.Methods.IPSF.const import count_field, zone_field
from typing import List, Literal, Tuple, Dict
import sys


def segment_df(df: pd.DataFrame, chunk_sz: int) -> List[pd.DataFrame]:
    start = 0
    ls_df = []
    while start < len(df):
        sub_df = df.iloc[start : start + chunk_sz]
        ls_df.append(sub_df)
        start += chunk_sz
    return ls_df


def convert_count_to_full(
    count_df: pd.DataFrame, count_col: str = count_field
) -> pd.DataFrame:
    assert count_col in count_df.columns
    repeated_idx = list(count_df.index.repeat(count_df[count_col]))
    fin = count_df.loc[repeated_idx]
    fin = fin.drop(columns=[count_col])
    fin = fin.reset_index(drop=True)
    return fin


def convert_full_to_marg_count(
    full_pop: pd.DataFrame, filter_ls: list[str] = []
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


def get_diff_marg(
    converted_census_marg: pd.DataFrame, converted_new_hh_marg: pd.DataFrame
) -> pd.DataFrame:
    print("getting the diff marg df")
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
    return converted_census_marg - converted_new_hh_marg


def convert_to_dict_ls(tup: Tuple[Tuple[str, str]]) -> Dict[str, str]:
    di = {}
    for a, b in tup:
        di.setdefault(a, []).append((a, b))
    return di


def adjust_kept_rec_match_census(
    syn_records: pd.DataFrame, diff_census: pd.DataFrame
) -> pd.DataFrame:
    # The point is to remove the chosen in
    syn_records = syn_records.astype(str)
    count_kept = syn_records.value_counts()
    # diff_census = diff_census.head(10) # sample to check smaller
    # diff_census = diff_census.set_index(diff_census.columns[diff_census.columns.get_level_values(0)==zone_field])
    for zone, r in diff_census.iterrows():
        sys.stdout.write(f"\rDOING deleting to match cencus diff for {zone}")
        sys.stdout.flush()
        sub_count_kept = count_kept.loc[
            count_kept.index.get_level_values(zone_field) == zone
        ]
        # before_sum = sub_count_kept.sum()
        prev_indexs = sub_count_kept.index
        neg_cols = r[r < 0]
        # re check with neg val
        dict_neg_v = convert_to_dict_ls(neg_cols.index)
        for i in range(len(dict_neg_v)):
            raws_before_comb = combinations(dict_neg_v.values(), len(dict_neg_v) - i)
            for raw in raws_before_comb:
                if neg_cols.sum() == 0:
                    break
                ls_pos_neg_comb = list(product(*raw))
                for comb in ls_pos_neg_comb:
                    # loop through each neg combs all
                    condi_check = True
                    to_del_n = np.inf
                    # search for sub df with combs and also dfind the
                    for att, state in comb:
                        condi_check &= (
                            sub_count_kept.index.get_level_values(att) == state
                        )
                        if att != zone_field:
                            check_v = neg_cols.loc[(att, state)] * -1
                            if check_v < to_del_n:
                                to_del_n = check_v
                    filtered_combs_from_kept = sub_count_kept.loc[condi_check]

                    if len(filtered_combs_from_kept) == 0 or to_del_n == 0:
                        continue

                    sum_val = filtered_combs_from_kept.sum()
                    if sum_val < to_del_n:
                        to_del_n = sum_val

                    # we need to spread the del_n by the dist
                    temp_hold_combs = filtered_combs_from_kept / sum_val
                    temp_hold_combs = temp_hold_combs * to_del_n

                    # First del by just normal rounding
                    to_del_first = np.floor(temp_hold_combs)
                    filtered_combs_from_kept = filtered_combs_from_kept - to_del_first
                    remaining_to_del = to_del_n - to_del_first.sum()
                    # we will spread the remaing to del for the top
                    filtered_combs_from_kept.sort_values(ascending=False, inplace=True)
                    filtered_combs_from_kept.iloc[: int(remaining_to_del)] -= 1
                    # Make sure there are no neg
                    assert not any(filtered_combs_from_kept < 0)

                    # Update the count kept
                    sub_count_kept.loc[
                        filtered_combs_from_kept.index
                    ] = filtered_combs_from_kept
                    neg_cols.loc[list(comb)] += to_del_n
                    sub_count_kept = sub_count_kept[sub_count_kept > 0]
        zero_indexes = set(prev_indexs) - set(sub_count_kept.index)
        count_kept.loc[sub_count_kept.index] = sub_count_kept
        count_kept.loc[list(zero_indexes)] = 0
        assert neg_cols.sum() == 0
        # after_sum = count_kept.loc[
        #     count_kept.index.get_level_values(zone_field) == zone
        # ].sum()
        # sys.stdout.write(f"\rFINISHED deleting {before_sum - after_sum} records to match cencus diff for {zone}")
        # sys.stdout.flush()
    print()
    return convert_count_to_full(count_kept.reset_index())
