"""Main place to run SAA for households synthesis"""


import pandas as pd
from PopSynthesis.Methods.IPSF.const import (
    data_dir,
    small_test_dir,
    processed_dir,
    zone_field,
)
from PopSynthesis.Methods.IPSF.utils.synthetic_checked_census import (
    adjust_kept_rec_match_census,
    get_diff_marg,
    convert_full_to_marg_count,
)
from PopSynthesis.Methods.IPSF.SAA.SAA import SAA
from typing import Tuple, List, Union
import random


def get_test_hh() -> Tuple[pd.DataFrame, pd.DataFrame]:
    hh_marg = pd.read_csv(small_test_dir / "hh_marginals_small.csv", header=[0, 1])
    hh_marg = hh_marg.set_index(hh_marg.columns[hh_marg.columns.get_level_values(0) == zone_field][0])
    pool = pd.read_csv(small_test_dir / "HH_pool_small_test.csv")
    return hh_marg, pool


def get_hh_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    hh_marg = pd.read_csv(data_dir / "hh_marginals_ipu.csv", header=[0, 1])
    hh_marg = hh_marg.drop(
        columns=hh_marg.columns[hh_marg.columns.get_level_values(0) == "sample_geog"][0]
    ).set_index(hh_marg.columns[hh_marg.columns.get_level_values(0) == zone_field][0])
    pool = pd.read_csv(processed_dir / "HH_pool.csv")
    return hh_marg, pool


def err_check_against_marg(syn_pop: pd.DataFrame, marg: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # error check
    marg_from_created = convert_full_to_marg_count(syn_pop, [zone_field])
    converted_marg = marg
    diff_marg = get_diff_marg(converted_marg, marg_from_created)

    kept_syn = adjust_kept_rec_match_census(syn_pop, diff_marg)

    # checking
    kept_marg = convert_full_to_marg_count(kept_syn, [zone_field])
    new_diff_marg = get_diff_marg(converted_marg, kept_marg)
    # check it is no neg indeed
    checking_not_neg = new_diff_marg < 0
    assert checking_not_neg.any(axis=None) == False
    # now get the new marg
    new_diff_marg.index = new_diff_marg.index.astype(int)
    new_diff_marg.index.name = zone_field
    return kept_syn, new_diff_marg


def process_shuffle_order(orginal_order: List[str], shuffle_order: Union[bool, List[str]]) -> List[str]:
    if isinstance(shuffle_order, list):
        assert set(shuffle_order) <= set(orginal_order)
        not_in_shuffle = [x for x in orginal_order if x not in shuffle_order]
        random.shuffle(not_in_shuffle)
        return shuffle_order + not_in_shuffle
        
    elif isinstance(shuffle_order, bool):
        if shuffle_order:
            random.shuffle(orginal_order)
        return orginal_order
    else:
        raise ValueError("Shuffle order should be a list or a boolean")


def saa_run(targeted_marg: pd.DataFrame, pool: pd.DataFrame, considered_atts=List[str], ordered_to_adjust_atts=List[str], shuffle_order:Union[bool, List[str]]=False, max_run_time:int=30) -> Tuple[pd.DataFrame, List[int]]:
    assert set(ordered_to_adjust_atts) <= set(considered_atts)
    atts_in_marg = set(targeted_marg.columns.get_level_values(0)) - {zone_field}
    assert set(ordered_to_adjust_atts) <= atts_in_marg
    assert zone_field in targeted_marg.index.name

    n_run_time = 0
    # init with the total HH we want
    n_removed_err = targeted_marg.sum().sum() / len(atts_in_marg)
    chosen_syn = []
    err_rm = []
    while n_run_time < max_run_time and n_removed_err > 0:
        # to randomly shuffle for each adjustment or not
        ordered_to_adjust_atts = process_shuffle_order(ordered_to_adjust_atts, shuffle_order)
        err_rm.append(n_removed_err)
        print(
            f"For run {n_run_time}, order is: {ordered_to_adjust_atts}, aim for {n_removed_err} HHs"
        )
        saa = SAA(targeted_marg, considered_atts, ordered_to_adjust_atts, pool)
        ###
        final_syn_pop = saa.run(extra_name=f"_{n_run_time}")
        ###
        kept_syn, new_marg = err_check_against_marg(final_syn_pop, targeted_marg)

        # append to the chosen
        if n_run_time == max_run_time:
            # not adjusting anymore
            chosen_syn.append(final_syn_pop)
        else:
            # continue with adjusting for missing
            chosen_syn.append(kept_syn)

        # Update for next run
        n_run_time += 1
        n_removed_err = len(final_syn_pop) - len(kept_syn)
        targeted_marg = new_marg

    final_syn_hh = pd.concat(chosen_syn)

    return final_syn_hh, err_rm

    

    
