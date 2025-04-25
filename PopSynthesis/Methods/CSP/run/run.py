"""Main placeholder to run_csp"""
from PopSynthesis.Methods.CSP.run.csp_sample_simple_back import csp_sample_by_hh
from PopSynthesis.Methods.CSP.run.csp_sample_one_way import sample_one_way
from PopSynthesis.Methods.CSP.run.csp_BN_sample import sample_rela_BN
from PopSynthesis.Methods.CSP.run.create_pool_pairs import create_pool_pairs
from PopSynthesis.Methods.CSP.run.process_pools_by_needs import process_original_pools
from PopSynthesis.Methods.CSP.const import ZONE_ID
from PopSynthesis.Methods.CSP.run.rela_const import EXPECTED_RELATIONSHIPS, HH_TAG
import pandas as pd
from typing import Dict, Union, List


def get_possible_states_each_att(hh_df: pd.DataFrame, pp_df: pd.DataFrame, exclude_cols: List[str], relationship:str) -> pd.DataFrame:
    results = {}
    # process hh
    for col in hh_df.columns:
        if col in exclude_cols:
            continue
        if col in results:
            raise ValueError(f"Likely, duplicates column names in the seed data")
        # Get the unique values for each column in hh_df
        unique_values = hh_df[col].unique()
        results[f"{HH_TAG}_{col}"] = list(unique_values)
    # process pp
    for rela in EXPECTED_RELATIONSHIPS:
        sub_pp_df = pp_df[pp_df[relationship] == rela].copy()
        for col in sub_pp_df.columns:
            if col in exclude_cols:
                continue
            if col in results:
                raise ValueError(f"Likely, duplicates column names in the seed data")
            # Get the unique values for each column in pp_df
            unique_values = sub_pp_df[col].unique()
            results[f"{rela}_{col}"] = list(unique_values)
    return results


def run_csp(hh_df: pd.DataFrame, configs: Dict[str, Union[str, pd.DataFrame]], handle_by_zone: bool=False, handle_1_way: bool=True, use_BN:bool = False, hh_has_n_already: bool = False) -> pd.DataFrame:
    """Run CSP with the given hh df and configs"""
    # From config we can have the seed hh, seed pp, we constraint by hh_size
    hh_seed = configs["hh_seed"]
    pp_seed = configs["pp_seed"]
    hhid = configs["hhid"]
    relationship = configs["relationship"]
    hhsz = configs["hh_size"]

    possible_states = get_possible_states_each_att(hh_df, pp_seed, exclude_cols=[hhid, relationship], relationship=relationship)

    sample_method = csp_sample_by_hh
    include_n_count_all = False
    if use_BN:
        sample_method = sample_rela_BN
        include_n_count_all = True
    elif handle_1_way:
        sample_method = sample_one_way
        include_n_count_all = True

    ori_pools = create_pool_pairs(hh_seed, pp_seed, hhid, relationship, include_n_count_all)
    # If we use IPF we can just use the original pool pairs (as all samples exist)
    final_conditonals = process_original_pools(ori_pools, method="original")
    if handle_by_zone:
        final_syn_pp = []
        for zid in hh_df[ZONE_ID].unique():
            print(f"Processing zone {zid}")
            syn_pp = sample_method(hh_df[hh_df[ZONE_ID]==zid].drop(columns=[ZONE_ID]), final_conditonals, hhsz, relationship, possible_states, hh_has_n_already)
            syn_pp[ZONE_ID] = zid
            final_syn_pp.append(syn_pp)
        return pd.concat(final_syn_pp, ignore_index=True)
    return sample_method(hh_df.drop(columns=[ZONE_ID], errors="ignore"), final_conditonals, hhsz, relationship, possible_states, hh_has_n_already)

