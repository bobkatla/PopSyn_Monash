"""Main place to run SAA for households synthesis"""


import pandas as pd
from PopSynthesis.Methods.IPSF.const import (
    data_dir,
    small_test_dir,
    processed_dir,
    zone_field,
    SAA_ODERED_ATTS_HH,
    CONSIDERED_ATTS_HH,
)
from PopSynthesis.Methods.IPSF.utils.synthetic_checked_census import (
    adjust_kept_rec_match_census,
    get_diff_marg,
    convert_full_to_marg_count,
)
from PopSynthesis.Methods.IPSF.SAA.SAA import SAA
from typing import Tuple, List
import random


def get_test_hh() -> Tuple[pd.DataFrame, pd.DataFrame]:
    hh_marg = pd.read_csv(small_test_dir / "hh_marginals_small.csv", header=[0, 1])
    pool = pd.read_csv(small_test_dir / "HH_pool_small_test.csv")
    return hh_marg, pool


def get_hh_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    hh_marg = pd.read_csv(data_dir / "hh_marginals_ipu.csv", header=[0, 1])
    hh_marg = hh_marg.drop(
        columns=hh_marg.columns[hh_marg.columns.get_level_values(0) == "sample_geog"][0]
    )
    pool = pd.read_csv(processed_dir / "HH_pool.csv")
    return hh_marg, pool


def err_check_against_marg(syn_pop: pd.DataFrame, marg: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # error check
    marg_from_created = convert_full_to_marg_count(syn_pop, [zone_field])
    converted_marg = marg.set_index(
        marg.columns[marg.columns.get_level_values(0) == zone_field][0]
    )
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
    return kept_syn, new_diff_marg.reset_index()


def saa_run(targeted_marg: pd.DataFrame, pool: pd.DataFrame, max_run_time:int=30) -> Tuple[pd.DataFrame, List[int]]:
    n_run_time = 0
    # init with the total HH we want
    n_removed_err = targeted_marg.sum().sum() / len(SAA_ODERED_ATTS_HH)
    chosen_syn = []
    err_rm = []
    while n_run_time < max_run_time and n_removed_err > 0:
        # randomly shuffle for each adjustment
        random.shuffle(SAA_ODERED_ATTS_HH)
        err_rm.append(n_removed_err)
        print(
            f"For run {n_run_time}, order is: {SAA_ODERED_ATTS_HH}, aim for {n_removed_err} HHs"
        )
        saa = SAA(targeted_marg, CONSIDERED_ATTS_HH, SAA_ODERED_ATTS_HH, pool)
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

    

    
