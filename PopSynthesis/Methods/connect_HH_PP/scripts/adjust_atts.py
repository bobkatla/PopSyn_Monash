"""
SAA method, eventually we may want to have a log to record and maybe turn each method into a class

This scripts will loop use the pools from BNs and update each atts to match with census
NOTE: we may consider removing the geo_lev var and put a general name for it such as "geog", the geo_lev applied at data processing only
"""

import pandas as pd
import numpy as np

from typing import Dict, List, Tuple, Any, Union

# Very bad, should list out the imports you want
from PopSynthesis.Methods.connect_HH_PP.scripts.sample_hh_main import *
from PopSynthesis.Methods.connect_HH_PP.scripts.process_all_hh_pp import *


def cal_states_diff(att: str, pop_df: pd.DataFrame, census_data: pd.DataFrame, geo_lev: str) -> Dict[str, Dict[str, int]]:
    """ This calculate the differences between current syn_pop and the census at a specific geo_lev """
    pop_counts = pop_df[[geo_lev, att]].value_counts()

    sub_census_data = census_data[
        census_data.columns[census_data.columns.get_level_values(0) == att]
    ]
    ls_states = list(sub_census_data.columns.get_level_values(1))
    re_dict = {}
    for state in ls_states:
        for zone in census_data.index:
            val_pop = (
                pop_counts[(zone, state)] if (zone, state,) in pop_counts.index else 0
            )
            val_census = sub_census_data.loc[zone, (att, state)]
            diff = val_census - val_pop
            if zone in re_dict:
                re_dict[zone][state] = diff
            else:
                re_dict[zone] = {state: diff}
    return re_dict


def get_neg_pos_ls(count_vals: Dict[str, int]) -> Tuple[List[str], List[str]]:
    """ Get list of negative or positive states (aka, to del and to add) """
    ls_neg_states = []
    ls_pos_states = []
    for state in count_vals:
        if count_vals[state] < 0:
            ls_neg_states.append(state)
        else:
            ls_pos_states.append(state)

    # rank the list neg to have the priority order
    # at the moment it is just from the largest number
    # but maybe will create to rank from least likely to make combinations
    ls_neg_states = sorted(ls_neg_states, key=lambda x: count_vals[x])

    return ls_neg_states, ls_pos_states


def get_comb_count(main_att: str, to_change_state: str, to_maintain_atts: List[str], pool: pd.DataFrame, normalize: bool =True) -> pd.Series:
    """ Get the count for different combination """
    sub_pool = pool[pool[main_att] == to_change_state]
    sub_pool = sub_pool[to_maintain_atts] # These are the atts got adjusted already, we used them to find matching combinations
    comb_counts = sub_pool.value_counts(
        normalize=normalize, ascending=True
    )  # if it is from the syn_pop that this is already perfected
    return comb_counts


def process_pos_states_counts(main_att: str, ls_pos_states: List[str], to_maintain_atts: List[str], pool: pd.DataFrame) -> Dict[str, pd.Series]:
    """ Get combinations from pos states list """
    re_dict = {}
    for state in ls_pos_states:
        re_dict[state] = get_comb_count(main_att, state, to_maintain_atts, pool)
        # Will think maybe will create the sample dict here as well
    return re_dict


def get_ls_ranked_comb(comb: Any, dict_comb: Dict[str, pd.Series]) -> List[str]:
    """ Return a ordered from largest to smallest in counts states """
    # At the moment, comb is a tuple of states for the maintain atts
    hold_dict = {}
    ordered_states_by_counts = []
    for state, counts in dict_comb.items():
        if comb in counts.index:
            hold_dict[state] = counts[comb]
            re_ls.append(state)
    return sorted(re_ls, key=lambda x: hold_dict[x], reverse=True)


def update_syn_pop(
    syn_pop: pd.DataFrame,
    pool: pd.DataFrame,
    n_adjust: int,
    prev_atts: List[str],
    main_att: str,
    comb: Any,
    del_state: str,
    plus_state: str,
    geo_lev: str,
    zone: str,
) -> pd.DataFrame:
    """ We update the current syn pop with a specific state replacement at a specific zone """
    # So the runtime we have is sum(tot_states for all atts) * num_zones
    syn_pop = syn_pop.reset_index(drop=True)
    # Filter to have sub_df of syn_pop about del
    q_based = ""
    for i, att in enumerate(prev_atts):
        q_based += f"{att}=='{comb[i]}' & "
    del_q = q_based + f"{main_att}=='{del_state}' & {geo_lev}=='{zone}'"
    plus_q = q_based + f"{main_att}=='{plus_state}'"

    sub_df_pop_syn = syn_pop.query(del_q)
    drop_indices = np.random.choice(sub_df_pop_syn.index, int(n_adjust), replace=False)
    kept_pop_syn = syn_pop.drop(drop_indices)
    assert len(syn_pop) - len(drop_indices) == len(kept_pop_syn)

    sub_pool = pool.query(plus_q)
    plus_df = sub_pool.sample(n=int(n_adjust), replace=True)
    plus_df[geo_lev] = zone
    assert len(kept_pop_syn) + len(plus_df) == len(syn_pop)

    new_syn_pop = pd.concat([kept_pop_syn, plus_df])
    return new_syn_pop


def wrapper_adjust_state(syn_pop: pd.DataFrame, dict_diff: Dict[str, Dict[str, int]], processed_atts: List[str], main_att:str, pool: pd.DataFrame, geo_lev:str) -> pd.DataFrame:
    """ A wrapper """
    # Doing zone by zone
    for zone in dict_diff:
        count_vals = dict_diff[zone]
        sub_syn_pop = syn_pop[syn_pop[geo_lev] == zone]
        # Process the pos_states counts from the pool
        ls_neg_states, ls_pos_states = get_neg_pos_ls(count_vals)
        dict_pos_comb_counts: Dict[str, pd.Series] = process_pos_states_counts(
            main_att, ls_pos_states, processed_atts, pool
        )

        # Now, will start the adjustment
        for neg_state in ls_neg_states:
            print(
                f"DOING adjustment for {main_att}_{neg_state} for zone {zone} with val: {count_vals[neg_state]}"
            )
            comb_count_neg = get_comb_count(
                main_att, neg_state, processed_atts, sub_syn_pop, normalize=False
            )
            # We will start delete from the top down (least likely to exist)
            for comb in comb_count_neg.index:
                # Get all possible pos can make with this comb, from most likely to not
                ls_ranked_comb_pos = get_ls_ranked_comb(comb, dict_pos_comb_counts)
                to_del_val = comb_count_neg[comb]
                for pos_state in ls_ranked_comb_pos:
                    # Now compare, the num to del is bigger or lower
                    if to_del_val <= 0:
                        # Finish now, no need to go further
                        break
                    to_plus_val = count_vals[pos_state]
                    n_adjust = min(to_del_val, to_plus_val)
                    if n_adjust > 0:
                        syn_pop = update_syn_pop(
                            syn_pop,
                            pool,
                            n_adjust,
                            processed_atts,
                            main_att,
                            comb,
                            neg_state,
                            pos_state,
                            geo_lev,
                            zone,
                        )
                        # Update
                        count_vals[neg_state] += n_adjust
                        count_vals[pos_state] -= n_adjust
                    to_del_val = to_del_val - to_plus_val
            if count_vals[neg_state] < 0:
                # Looping through all possible combinations but could not fill it
                print(
                    f"WARNING: could not do total adjustment for {main_att}_{neg_state}, still have {count_vals[neg_state]}"
                )
        # Maybe do errror for this zone, is there a way to optimise it maybe like vector calculation?
    return syn_pop


def process_data_general(census_data: pd.DataFrame, pool: pd.DataFrame, geo_lev: str, adjust_atts_order: List[str]) -> None:
    # Census data will be in format of count with zone_id, and columns in 2 levels
    # Loop through each att
    census_data = census_data.set_index(
        census_data.columns[census_data.columns.get_level_values(0) == "zone_id"][0]
    )
    cols_drop = census_data.columns[
        census_data.columns.get_level_values(0).isin(["zone_id", "sample_geog"])
    ]
    census_data = census_data.drop(columns=cols_drop)
    # ls_atts_order = list(census_data.columns.get_level_values(0).unique()) #at the moment it is just by order from marginals file, will fix later
    census_data.index = census_data.index.astype(str)

    syn_pop = None
    processed_atts = []
    for att in adjust_atts_order:
        if syn_pop is None:  # first time run, this should be perfect
            syn_pop = samp_from_pool_1layer(pool, census_data, att, geo_lev)
            syn_pop = syn_pop.astype(str)
            syn_pop.index = syn_pop.index.astype(str)
        else:
            # Now we need to process from the syn pop
            dict_diff = cal_states_diff(att, syn_pop, census_data, geo_lev)
            syn_pop = wrapper_adjust_state(
                syn_pop, dict_diff, processed_atts, att, pool, geo_lev
            )
            # I belive lol
        processed_atts.append(att)
        syn_pop.to_csv(
            os.path.join(output_dir, "testland", f"saving_hh_test2_{att}.csv"),
            index=False,
        )
    return syn_pop


def main() -> None:
    census_data = pd.read_csv(
        os.path.join(data_dir, "hh_marginals_ipu.csv"), header=[0, 1]
    )
    df_seed = pd.read_csv(os.path.join(processed_data, "ori_sample_hh.csv"))
    # drop all the ids as they are not needed for in BN learning
    id_cols = [x for x in df_seed.columns if "hhid" in x or "persid" in x]
    df_seed = df_seed.drop(columns=id_cols)

    with open(os.path.join(processed_data, "dict_hh_states.pickle"), "rb") as handle:
        hh_state_names = pickle.load(handle)

    pool = get_pool(df_seed, hh_state_names, pool_sz=POOL_SZ)
    geo_lev = "POA"
    adjust_atts_order = ["hhsize", "hhinc"]
    syn_pop = process_data_general(census_data, pool, geo_lev, adjust_atts_order)
    print(syn_pop)
    # syn_pop.to_csv(os.path.join(output_dir, "test_new_just_hhinc.csv"), index=False)


def sample_without_adjustments():
    import random
    from itertools import chain

    census_data = pd.read_csv(
        os.path.join(data_dir, "hh_marginals_ipu.csv"), header=[0, 1]
    )
    df_seed = pd.read_csv(os.path.join(processed_data, "ori_sample_hh.csv"))
    # drop all the ids as they are not needed for in BN learning
    id_cols = [x for x in df_seed.columns if "hhid" in x or "persid" in x]
    df_seed = df_seed.drop(columns=id_cols)

    with open(os.path.join(processed_data, "dict_hh_states.pickle"), "rb") as handle:
        hh_state_names = pickle.load(handle)

    # Sample without adjustment
    tot = census_data[
        census_data.columns[census_data.columns.get_level_values(0) == "hhsize"]
    ].sum(axis=1)
    ls_zones = census_data[
        census_data.columns[census_data.columns.get_level_values(0) == "zone_id"]
    ]
    combine = pd.concat([tot, ls_zones], axis=1)
    combine.columns = ["tot", "zones"]
    check = tot.sum()
    ls_df = []
    while check > 0:
        print(check)
        sample = get_pool(df_seed, hh_state_names, pool_sz=check, special=False)
        ls_df.append(sample)
        check -= len(sample)
    fin_no_ad = pd.concat(ls_df)
    combine["ls_zones"] = combine.apply(lambda r: [r["zones"]] * r["tot"], axis=1)
    ls_vals_zone = list(chain.from_iterable(combine["ls_zones"]))
    random.shuffle(ls_vals_zone)
    fin_no_ad["POA"] = ls_vals_zone
    print(fin_no_ad)
    fin_no_ad.to_csv(os.path.join(output_dir, "hh_no_adjustments.csv"), index=False)


if __name__ == "__main__":
    # main()
    sample_without_adjustments()
