"""
Converting original seeds to paired case

Inputs: hh_seed, pp_seed
output: paired for HH (with rela count) and Main, Main and each rela
Maybe put the pool creation here as well (as we need to check for the synthesis of hhsize == total rela count)
"""
import pandas as pd
from typing import Dict

def convert_seeds_to_pairs(hh_seed: pd.DataFrame, pp_seed: pd.DataFrame, id_col:str, pp_segment_col: str, main_state: str) -> Dict[str, pd.DataFrame]:
    pp_states = pp_seed[pp_segment_col].unique()
    assert main_state in pp_states
    assert id_col in hh_seed.columns
    assert id_col in pp_seed.columns
    
    hh_seed[id_col] = hh_seed[id_col].astype(str)
    pp_seed[id_col] = pp_seed[id_col].astype(str)
    hh_name = "HH" # simply for naming convention

    segmented_pp = segment_pp_seed(pp_seed, pp_segment_col)
    assert len(segmented_pp[main_state]) == len(hh_seed)
    
    # pair up HH - Main first
    result_pairs = {f"{hh_name}-{main_state}": pair_by_id(hh_seed, segmented_pp[main_state], id_col, hh_name, main_state)}
    assert len(result_pairs[f"{hh_name}-{main_state}"]) == len(hh_seed)
    for pp_state in pp_states:
        if pp_state != main_state:
            result_pairs[f"{main_state}-{pp_state}"] = pair_by_id(segmented_pp[main_state], segmented_pp[pp_state], id_col, main_state, pp_state)
    return result_pairs


def pair_by_id(df1: pd.DataFrame, df2: pd.DataFrame, id: str, name1:str="x", name2:str="y") -> pd.DataFrame:
    # join by the id col with inner (so only matched one got accepted)
    # Likely id will not be unique for df2 as the normal rela may have multiple in 1 hh
    join_result = df1.merge(df2, on=id, how="inner", suffixes=[f"_{name1}", f"_{name2}"])
    assert len(join_result) == min(len(df1), len(df2))
    return join_result


def segment_pp_seed(pp_seed: pd.DataFrame, segment_col: str) -> Dict[str, pd.DataFrame]:
    result_seg_pp = {}
    for state in pp_seed[segment_col].unique():
        result_seg_pp[state] = pp_seed[pp_seed[segment_col]==state]
    return result_seg_pp
