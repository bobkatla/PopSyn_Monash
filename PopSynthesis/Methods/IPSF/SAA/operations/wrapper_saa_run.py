"""Main place to run SAA for households synthesis"""


import pandas as pd
from PopSynthesis.Methods.IPSF.const import (
    data_dir,
    small_test_dir,
    processed_dir,
    zone_field,
    output_dir,
)
from PopSynthesis.Methods.IPSF.utils.synthetic_checked_census import (
    adjust_kept_rec_match_census,
    get_diff_marg,
    convert_full_to_marg_count,
)
from PopSynthesis.Methods.IPSF.SAA.SAA import SAA
import polars as pl
from typing import Tuple, List
import random
from pathlib import Path


def get_test_pp() -> Tuple[pd.DataFrame, pd.DataFrame]:
    pp_marg = pd.read_csv(small_test_dir / "pp_small_marginals.csv", header=[0, 1])
    pp_marg = pp_marg.set_index(
        pp_marg.columns[pp_marg.columns.get_level_values(0) == zone_field][0]
    )
    pool = pd.read_csv(data_dir / "pp_sample_ipu.csv")
    pool = pool.drop(columns=["serialno", "sample_geog"])
    return pp_marg, pool


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


def get_pp_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    pp_marg = pd.read_csv(data_dir / "person_marginals_ipu.csv", header=[0, 1])
    pp_marg = pp_marg.set_index(pp_marg.columns[pp_marg.columns.get_level_values(0) == zone_field][0])
    pool = pd.read_csv(processed_dir / "PP_pool.csv")
    return pp_marg, pool



def err_check_against_marg(
    syn_pop: pd.DataFrame, marg: pd.DataFrame, extra_rm_frac: float = 0, exclude_atts: List[str] = []
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # rm extra first
    assert extra_rm_frac <= 1 and extra_rm_frac >= 0
    remain_syn = syn_pop.copy()
    remain_syn = remain_syn.sample(frac=1-extra_rm_frac)
    print(f"removed first {len(syn_pop) - len(remain_syn)} agents")

    marg_from_created = convert_full_to_marg_count(remain_syn, [zone_field])
    converted_marg = marg
    converted_marg = converted_marg.drop(converted_marg.columns[converted_marg.columns.get_level_values(0).isin(exclude_atts)], axis=1)
    marg_from_created = marg_from_created.drop(marg_from_created.columns[marg_from_created.columns.get_level_values(0).isin(exclude_atts)], axis=1)

    diff_marg = get_diff_marg(converted_marg, marg_from_created)

    kept_syn = adjust_kept_rec_match_census(remain_syn, diff_marg)

    # checking
    kept_marg = convert_full_to_marg_count(kept_syn, [zone_field])
    kept_marg = kept_marg.drop(kept_marg.columns[kept_marg.columns.get_level_values(0).isin(exclude_atts)], axis=1)
    new_diff_marg = get_diff_marg(converted_marg, kept_marg)
    # check it is no neg indeed
    checking_not_neg = new_diff_marg < 0
    assert checking_not_neg.any(axis=None) == False
    # now get the new marg
    new_diff_marg.index = new_diff_marg.index.astype(int)
    new_diff_marg.index.name = zone_field
    return kept_syn, new_diff_marg


def process_shuffle_order(
    orginal_order: List[str], shuffle_order: List[str], idx_atts_states: pd.MultiIndex, check_run_time: str, randomly_add_last: List[str] = []
) -> List[str]:
    if check_run_time == "first":
        # First run, change the order from most states to least states
        flatten_idx = idx_atts_states.to_frame(index=False, name=["att", "state"])
        count_state = flatten_idx.groupby("att").count().sort_values("state", ascending=False)
        return count_state.index.tolist()
    elif check_run_time == "last":
        assert set(shuffle_order) <= set(orginal_order)
        not_in_shuffle = [x for x in orginal_order if x not in shuffle_order]
        random.shuffle(not_in_shuffle)
        return shuffle_order + not_in_shuffle
    else:
        # mid case, this will be a number
        n_run_time = int(check_run_time)
        # want each value got to be the first once at least
        first_att = orginal_order[n_run_time % len(orginal_order)]
        random_boolean = random.choices([True, False], weights=[1, 3])[0]
        
        random.shuffle(orginal_order)
        orginal_order.remove(first_att)
        orginal_order.insert(0, first_att)
        if random_boolean and len(randomly_add_last) > 0:
            for att in randomly_add_last:
                orginal_order.remove(att)
                orginal_order.append(att)

        return orginal_order


def saa_run(
    targeted_marg: pd.DataFrame,
    count_pool: pl.DataFrame,
    considered_atts=List[str],
    ordered_to_adjust_atts=List[str],
    last_adjustment_order: List[str] = [],
    max_run_time: int = 30,
    extra_rm_frac: float = 0,
    output_each_step: bool = False,
    add_name_for_step_output: str = "",
    include_zero_cell_values: bool = False,
    randomly_add_last: List[str] = [],
    meta_output_dir: Path = output_dir,
) -> Tuple[pd.DataFrame, List[int]]:
    assert set(ordered_to_adjust_atts) <= set(considered_atts)
    atts_in_marg = set(targeted_marg.columns.get_level_values(0)) - {zone_field}
    assert set(ordered_to_adjust_atts) <= atts_in_marg
    assert zone_field in targeted_marg.index.name

    excluded_atts = list(set(considered_atts) - set(ordered_to_adjust_atts))
    print(f"These atts are synthesized but not adjusted: {excluded_atts}")

    n_run_time = 0
    # init with the total HH we want
    n_removed_err = targeted_marg.sum().sum() / len(atts_in_marg)
    chosen_syn = []
    err_rm = []
    while n_run_time < max_run_time and n_removed_err > 0:
        check_run_time = str(n_run_time)
        if n_run_time == 0:
            check_run_time = "first"
        if n_run_time == max_run_time - 1:
            check_run_time = "last" # priortise last so the wanted order would be performed
        # to randomly shuffle for each adjustment or not
        ordered_to_adjust_atts = process_shuffle_order(
            ordered_to_adjust_atts, last_adjustment_order, targeted_marg.columns, check_run_time=check_run_time, randomly_add_last=randomly_add_last
        )
        err_rm.append(n_removed_err)
        print(
            f"For run {n_run_time}, order is: {ordered_to_adjust_atts}, aim for {n_removed_err} agents"
        )
        saa = SAA(targeted_marg, considered_atts, ordered_to_adjust_atts, count_pool)
        ### Actual running to get the synthetic pop
        final_syn_pop = saa.run(
            extra_name=f"_{add_name_for_step_output}_{n_run_time}", 
            output_each_step=output_each_step, 
            include_zero_cell_values=include_zero_cell_values,
            output_dir=meta_output_dir
        )
        assert len(final_syn_pop) == n_removed_err
        ###

        n_run_time += 1
        # append to the chosen
        if n_run_time == max_run_time:
            # not adjusting anymore, last run
            final_syn_pop = final_syn_pop.with_columns(pl.col(zone_field).cast(pl.String))
            chosen_syn.append(final_syn_pop)
        else:
            to_check_syn = final_syn_pop.to_pandas()
            kept_syn, new_marg = err_check_against_marg(to_check_syn, targeted_marg, extra_rm_frac, excluded_atts)
            if len(kept_syn) > 0:
                # continue with adjusting for missing
                chosen = pl.from_pandas(kept_syn)
                chosen = chosen.with_columns(pl.col(zone_field).cast(pl.String))
                chosen_syn.append(chosen)
            else:
                chosen_syn.append(None)
            # Update for next run
            n_removed_err = len(final_syn_pop) - len(kept_syn)
            targeted_marg = new_marg
    
    if output_each_step:
        for i, df in enumerate(chosen_syn):
            if df is not None:
                df.write_csv(
                    meta_output_dir / f"kept_syn_run_{i}.csv"
                )

    final_syn_hh = pl.concat([df.select(considered_atts+[zone_field]) for df in chosen_syn if df is not None])
    return final_syn_hh, err_rm
