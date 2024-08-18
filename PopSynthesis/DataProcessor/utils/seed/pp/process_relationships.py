from collections import defaultdict
from PopSynthesis.Methods.connect_HH_PP.scripts.const import LS_GR_RELA, HANDLE_THE_REST_RELA
import polars as pl
import numpy as np
import pandas as pd
from typing import List

MIN_PARENT_CHILD_GAP = 16
MIN_GRANDPARENT_GRANDCHILD_GAP = 33
MIN_COUPLE_GAP = 20 # This only apply when we do the conversion


def check_rela_gb(gb_df):
    for hhid, rela_gr in zip(gb_df.index, gb_df):
        check_dict = defaultdict(lambda: 0)
        for i in rela_gr:
            check_dict[i] += 1
        if check_dict["Self"] == 0:
            # print(hhid)
            print([f"{x} - {y}" for x, y in check_dict.items() if x != "Self"])
        elif check_dict["Self"] > 1:
            print("NOOOOOOOOOO", hhid, rela_gr)

    
def process_not_accept_values(pp_df):
    # Remove not accept value
    # At the moment we remove Null and Missing for income
    pp_df = pp_df.drop_nulls()
    pp_df_missing = pp_df.filter(pl.col("persinc")=="Missing/Refused")
    to_rm_hhid = list(pp_df_missing["hhid"].unique())
    pp_df = pp_df.filter(~pl.col("hhid").is_in(to_rm_hhid))
    return pp_df


def convert_simple_income(income_str):
    if "Negative" in income_str:
        return -1
    elif "Missing" in income_str:
        # This should not happen as we will filter no income
        return -2
    elif "Zero" in income_str:
        return 0
    elif "-" in income_str:
        return int(income_str.split("-")[0].replace("$", ""))
    elif "+" in income_str:
        return 2000
    else:
        raise ValueError("Weird")
    

def idx_max_val_return(ls):
    max_idx = None
    max_val = None
    for idx, val in enumerate(ls):
        if max_val is None or val > max_val:
            max_idx = idx
            max_val = val
    return max_idx


def find_idx_value(ls, find_val):
    for idx, val in enumerate(ls):
        if val == find_val:
            return idx
    return None


def process_info_each_house(r):
    val_combine = r["id_combine"]
    ls_id, ls_age, ls_sex, ls_income, ls_rela = np.array(val_combine).T
    ls_rela[0] = "Main" # replace all
    
    # Check Child and GrandChild
    self_age = ls_age[0]
    implausible_case = False
    for rela, age in zip(ls_rela, ls_age):
        age_gap = int(self_age) - int(age)
        implausible_child = rela == "Child" and age_gap < MIN_PARENT_CHILD_GAP
        implausible_grandchild =  rela == "Grandchild" and age_gap < MIN_GRANDPARENT_GRANDCHILD_GAP
        if implausible_child or implausible_grandchild:
            implausible_case = True
            break
            
    # the_oldest = ls_id[idx_max_val_return(ls_age)]
    # the_highest_income = ls_id[idx_max_val_return(ls_income)]
    
    # the_oldest_rela = ls_rela[idx_max_val_return(ls_age.astype(int))]
    the_highest_income_rela = ls_rela[idx_max_val_return(ls_income.astype(int))]

    # idx_self = find_idx_value(ls_rela, "Self")
    # the_self = None if idx_self is None else ls_id[idx_self]
    num_main = list(ls_rela).count("Main")
    num_self = list(ls_rela).count("Self")
    # Check the Main replacement
    assert num_self == 0
    assert num_main == 1
    # num_pp = len(ls_rela)
    return the_highest_income_rela, implausible_case


def _extract_highest_n_main_idx(ordered_incomes, ordered_rela, check_rela):
    assert list(ordered_rela).count("Self") == 0 or list(ordered_rela).count("Self") == 1
    ordered_rela[0] = "Main" # replace all, this is maybe redundant
    main_idx = find_idx_value(ordered_rela, "Main")
    highest_idx = idx_max_val_return(ordered_incomes.astype(int))
    # confirm again, never bad to do this
    assert ordered_rela[highest_idx] == check_rela
    return main_idx, highest_idx


def process_func_diff_rela(id_combine, check_rela: str) -> dict[str, str]:
    ls_id, ls_age, ls_sex, ls_income, ls_rela = np.array(id_combine).T
    main_idx, highest_idx = _extract_highest_n_main_idx(ls_income, ls_rela, check_rela)
    if check_rela == "Spouse":
        # Simple swap
        main_id = ls_id[main_idx]
        highest_income_id = ls_id[highest_idx]
        return {main_id: check_rela, highest_income_id: "Main"}
    elif check_rela == "Child":
        # We all assume that they don't consider their in-law their Child
        # If simple no other and no Grandchild we can simply swap, other Child will become Sibling
        NotImplemented
        # If Granchild but no Other, we check the gap, and if fit, it's the child, not then Others
        # If Others, we check the age gap and sex, if fit then turn them to Spouse (assert that the Child is above 18)
        # Check the Grandchild, if 
    

def process_highest_inc(rela_only_df: pd.DataFrame, check_rela: str) -> pd.Series:
    """Return dict of new changes, we will basically map them"""
    assert list(rela_only_df["highest_inc"].unique()) == [check_rela]        
    result_mapping = rela_only_df["id_combine"].apply(lambda id_combine: process_func_diff_rela(id_combine, check_rela))
    return result_mapping


def process_chosen_to_others(pp_data: pd.DataFrame, to_others_rela: List[str] = ["Other relative", "Unrelated", "Other"], other_name:str = "Others") -> pd.DataFrame:
    # Maybe let's keep sibling, as sibling defo cannot be spouse, and they maybe special
    filter_data_to_process = pp_data[pp_data["relationship"].isin(to_others_rela)]
    to_process_pp_data = pp_data.copy(deep=True) # ensure we don't modify the original data
    to_process_pp_data.loc[filter_data_to_process.index, "relationship"] = other_name
    return to_process_pp_data


def process_rela(pp_df: pl.DataFrame):
    # To handle relationship, generally we based on income, age and gender
    # Due to complexity, I will keep pandas here for now
    pp_df: pd.DataFrame = pp_df.to_pandas()
    pp_df = process_chosen_to_others(pp_df) # Note, it's Others not Other, minor to help tell diff

    pp_df["converted_income"] = pp_df["persinc"].apply(convert_simple_income)
    pp_df["id_combine"] = pp_df.apply(lambda r: [r["persid"], r["age"], r["sex"], r["converted_income"], r["relationship"]], axis=1)
    gb_pid = pp_df.groupby("hhid")["id_combine"].apply(lambda x: list(x))
    pid_df = pd.DataFrame(gb_pid)
    corr_cols = ["highest_inc", "implausible_case"]
    pid_df[corr_cols] = pid_df.apply(process_info_each_house, result_type="expand", axis=1)
    # Now we need to remove hh that has implauble case
    implausible_hh = pid_df[pid_df["implausible_case"]]
    print(f"Remove {len(implausible_hh)} households with implausible combinations")
    filtered_cases = pid_df[~pid_df["implausible_case"]]

    # Next filter case of not Main for highest income
    no_change_cases = filtered_cases[filtered_cases["highest_inc"]=="Main"]
    # to_check_cases = filtered_cases[filtered_cases["highest_inc"]!="Main"]
    # print(to_check_cases["highest_inc"].value_counts())
    check_spouse_cases = filtered_cases[filtered_cases["highest_inc"]=="Spouse"]
    check_child_cases = filtered_cases[filtered_cases["highest_inc"]=="Child"]
    check_grandchild_cases = filtered_cases[filtered_cases["highest_inc"]=="Grandchild"]
    # Still need to think about how to handle case of Other
    processed_spouse = process_highest_inc(check_spouse_cases, "Spouse")
    processed_child = process_highest_inc(check_child_cases, "Child")
    # Last step would be combined all of the df into 1
    # And check for all of them the highest income is Main, again
    # return pp_df
