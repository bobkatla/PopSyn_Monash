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
import polars as pl
from typing import Tuple, List, Union, Literal
import random


def get_test_hh() -> Tuple[pd.DataFrame, pd.DataFrame]:
    hh_marg = pd.read_csv(small_test_dir / "hh_marginals_small.csv", header=[0, 1])
    hh_marg = hh_marg.set_index(
        hh_marg.columns[hh_marg.columns.get_level_values(0) == zone_field][0]
    )
    pool = pd.read_csv(small_test_dir / "HH_pool_small_test.csv")
    return hh_marg, pool


def get_hh_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    hh_marg = pd.read_csv(data_dir / "hh_marginals_ipu.csv", header=[0, 1])
    hh_marg = hh_marg.drop(
        columns=hh_marg.columns[hh_marg.columns.get_level_values(0) == "sample_geog"][0]
    ).set_index(hh_marg.columns[hh_marg.columns.get_level_values(0) == zone_field][0])
    pool = pd.read_csv(processed_dir / "HH_pool.csv")
    return hh_marg, pool


def err_check_against_marg(
    syn_pop: pd.DataFrame, marg: pd.DataFrame, extra_rm_frac: float = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # rm extra first
    assert extra_rm_frac <= 1 and extra_rm_frac >= 0
    remain_syn = syn_pop
    remain_syn = remain_syn.sample(frac=1-extra_rm_frac)
    print(f"removed first {len(syn_pop) - len(remain_syn)} hh")

    marg_from_created = convert_full_to_marg_count(remain_syn, [zone_field])
    converted_marg = marg
    diff_marg = get_diff_marg(converted_marg, marg_from_created)

    kept_syn = adjust_kept_rec_match_census(remain_syn, diff_marg)

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


def process_shuffle_order(
    orginal_order: List[str], shuffle_order: List[str], idx_atts_states: pd.MultiIndex, check_run_time: Literal["first", "mid", "last"]
) -> List[str]:
    if check_run_time == "first":
        # First run, change the order from most states to least states
        flatten_idx = idx_atts_states.to_frame(index=False, name=["att", "state"])
        count_state = flatten_idx.groupby("att").count().sort_values("state", ascending=False)
        return count_state.index.tolist()
    elif check_run_time == "mid":
        random.shuffle(orginal_order)
        return orginal_order
    elif check_run_time == "last":
        assert set(shuffle_order) <= set(orginal_order)
        not_in_shuffle = [x for x in orginal_order if x not in shuffle_order]
        random.shuffle(not_in_shuffle)
        return shuffle_order + not_in_shuffle
    else:
        raise ValueError("Check run time should be first, mid or last")


def saa_run(
    targeted_marg: pd.DataFrame,
    count_pool: pl.DataFrame,
    considered_atts=List[str],
    ordered_to_adjust_atts=List[str],
    shuffle_order: List[str] = [],
    max_run_time: int = 30,
    extra_rm_frac: float = 0,
    output_each_step: bool = False,
) -> Tuple[pd.DataFrame, List[int]]:
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
        check_run_time = "mid"
        if n_run_time == 0:
            check_run_time = "first"
        elif n_run_time == max_run_time - 1:
            check_run_time = "last"
        # to randomly shuffle for each adjustment or not
        ordered_to_adjust_atts = process_shuffle_order(
            ordered_to_adjust_atts, shuffle_order, targeted_marg.columns, check_run_time=check_run_time
        )
        err_rm.append(n_removed_err)
        print(
            f"For run {n_run_time}, order is: {ordered_to_adjust_atts}, aim for {n_removed_err} HHs"
        )
        saa = SAA(targeted_marg, considered_atts, ordered_to_adjust_atts, count_pool)
        ### Actual running to get the synthetic pop
        final_syn_pop = saa.run(extra_name=f"_{n_run_time}", output_each_step=output_each_step)
        assert len(final_syn_pop) == n_removed_err
        ###
        to_check_syn = final_syn_pop.to_pandas()
        kept_syn, new_marg = err_check_against_marg(to_check_syn, targeted_marg, extra_rm_frac)

        n_run_time += 1
        # append to the chosen
        if n_run_time == max_run_time:
            # not adjusting anymore
            final_syn_pop = final_syn_pop.with_columns(pl.col(zone_field).cast(pl.String))
            chosen_syn.append(final_syn_pop)
        elif len(kept_syn) > 0:
            # continue with adjusting for missing
            chosen = pl.from_pandas(kept_syn)
            chosen = chosen.with_columns(pl.col(zone_field).cast(pl.String))
            chosen_syn.append(chosen)

        # Update for next run
        n_removed_err = len(final_syn_pop) - len(kept_syn)
        targeted_marg = new_marg

    final_syn_hh = pl.concat([df.select(considered_atts+[zone_field]) for df in chosen_syn])
    return final_syn_hh, err_rm
