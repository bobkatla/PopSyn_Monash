"""
This scripts will loop use the pools from BNs and update each atts to match with census
"""
import pandas as pd
import numpy as np
import random

from PopSynthesis.Methods.connect_HH_PP.scripts.sample_hh_main import *
from PopSynthesis.Methods.connect_HH_PP.scripts.process_all_hh_pp import *


def cal_states_diff(att, pop_df, census_data, geo_lev):
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


def get_neg_pos_ls(count_vals: dict[str, int]):
    ls_neg_states = []
    ls_pos_states = []
    for state, val in count_vals.items():
        if val < 0:
            ls_neg_states.append(state)
        else:
            ls_pos_states.append(state)

    # rank the list neg to have the priority order
    # at the moment it is just from the largest number
    # but maybe will create to rank from least likely to make combinations
    # ls_neg_states = sorted(ls_neg_states, key=lambda x: count_vals[x])
    # Make it random
    random.shuffle(ls_neg_states)

    return ls_neg_states, ls_pos_states


def get_comb_count(
    main_att, to_change_state, to_maintain_atts, pool: pd.DataFrame, normalize=True
):
    sub_pool = pool[pool[main_att] == to_change_state]
    sub_pool = sub_pool[to_maintain_atts + ["count"]]
    sub_pool["count"] = sub_pool["count"].astype(int)
    comb_counts = sub_pool.groupby(to_maintain_atts)["count"].sum()
    if normalize:
        comb_counts = comb_counts / comb_counts.sum()
    comb_counts = comb_counts.sort_values()
    return comb_counts


def process_pos_states_counts(main_att, ls_pos_states, to_maintain_atts, pool_count):
    re_dict = {}
    for state in ls_pos_states:
        comb_counts = get_comb_count(main_att, state, to_maintain_atts, pool_count)
        re_dict[state] = comb_counts
        # Will think maybe will create the sample dict here as well
    return re_dict


def get_ls_ranked_comb(comb, dict_comb):
    hold_dict = {}
    re_ls = []
    for state, counts in dict_comb.items():
        if comb in counts.index:
            hold_dict[state] = counts[comb]
            re_ls.append(state)
    return sorted(re_ls, key=lambda x: hold_dict[x], reverse=True)


def update_syn_pop(
    syn_pop,
    pool_count,
    n_adjust,
    prev_atts,
    main_att,
    comb,
    del_state,
    plus_state,
    geo_lev,
    zone,
):
    syn_pop = syn_pop.reset_index(drop=True)
    # Filter to have sub_df of syn_pop about del
    q_based = ""
    for i, att in enumerate(prev_atts):
        comb_state = comb if len(prev_atts) == 1 else comb[i]
        q_based += f"{att}=='{comb_state}' & "
    del_q = q_based + f"{main_att}=='{del_state}' & {geo_lev}=='{zone}'"
    plus_q = q_based + f"{main_att}=='{plus_state}'"

    sub_df_pop_syn = syn_pop.query(del_q)
    drop_indices = np.random.choice(sub_df_pop_syn.index, int(n_adjust), replace=False)
    kept_pop_syn = syn_pop.drop(drop_indices)
    assert len(syn_pop) - len(drop_indices) == len(kept_pop_syn)

    sub_pool = pool_count.query(plus_q)
    plus_df = sub_pool.sample(n=int(n_adjust), replace=True, weights="count")
    plus_df[geo_lev] = zone
    plus_df = plus_df.drop(columns=["count"])
    assert len(kept_pop_syn) + len(plus_df) == len(syn_pop)

    new_syn_pop = pd.concat([kept_pop_syn, plus_df])
    return new_syn_pop


def wrapper_adjust_state(
    syn_pop, dict_diff, processed_atts, main_att, pool_count, geo_lev
):
    # Doing zone by zone
    for zone in dict_diff:
        count_vals = dict_diff[zone]
        sub_syn_pop = syn_pop[syn_pop[geo_lev] == zone].value_counts().reset_index()
        # Process the pos_states counts from the pool
        ls_neg_states, ls_pos_states = get_neg_pos_ls(count_vals)
        dict_pos_comb_counts = process_pos_states_counts(
            main_att, ls_pos_states, processed_atts, pool_count
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
                            pool_count,
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


def process_data_general(
    census_data, pool_count, geo_lev, adjust_atts_order, is_ipu_data=True
):
    # Census data will be in format of count with zone_id, and columns in 2 levels
    # Loop through each att
    if is_ipu_data:
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
            syn_pop = samp_from_pool_1layer(pool_count, census_data, att, geo_lev)
            syn_pop = syn_pop.astype(str)
            syn_pop.index = syn_pop.index.astype(str)
        else:
            # Now we need to process from the syn pop
            dict_diff = cal_states_diff(att, syn_pop, census_data, geo_lev)
            syn_pop = wrapper_adjust_state(
                syn_pop, dict_diff, processed_atts, att, pool_count, geo_lev
            )
            # I belive lol
        processed_atts.append(att)
        syn_pop.to_csv(
            os.path.join(output_dir, "testland", f"saving_hh_test2_{att}.csv"),
            index=False,
        )
    return syn_pop


def main():
    census_data = pd.read_csv(
        os.path.join(data_dir, "hh_marginals_ipu.csv"), header=[0, 1]
    )
    df_seed = pd.read_csv(os.path.join(processed_data, "ori_sample_hh.csv"))
    # drop all the ids as they are not needed for in BN learning
    id_cols = [x for x in df_seed.columns if "hhid" in x or "persid" in x]
    df_seed = df_seed.drop(columns=id_cols)

    with open(os.path.join(processed_data, "dict_hh_states.pickle"), "rb") as handle:
        hh_state_names = pickle.load(handle)

    pool_count = pd.read_csv(
        os.path.join(processed_data, "save_pools", "final_pool_count.csv")
    )
    geo_lev = "POA"
    adjust_atts_order = ["hhsize", "hhinc"]
    syn_pop = process_data_general(census_data, pool_count, geo_lev, adjust_atts_order)
    print(syn_pop)


if __name__ == "__main__":
    main()
