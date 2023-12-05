"""
This scripts will loop use the pools from BNs and update each atts to match with census
"""
import pandas as pd
import numpy as np

from PopSynthesis.Methods.connect_HH_PP.scripts.sample_hh_main import *
from PopSynthesis.Methods.connect_HH_PP.scripts.process_all_hh_pp import *


def cal_states_diff(att, pop_df, census_data, geo_lev):
    pop_counts = pop_df[[geo_lev, att]].value_counts()

    sub_census_data = census_data[census_data.columns[census_data.columns.get_level_values(0)==att]]
    ls_states = list(sub_census_data.columns.get_level_values(1))

    re_dict = {}
    for state in ls_states:
        for zone in census_data.index:
            val_pop = pop_counts[(zone, state)] if (zone, state) in pop_counts.index else 0
            val_census = sub_census_data.loc[zone, (att, state)]
            diff = val_census - val_pop
            if zone in re_dict:
                re_dict[zone][state] = diff
            else:
                re_dict[zone] = {state: diff}
    return re_dict


def process_data_general(census_data, pool, geo_lev):
    # Census data will be in format of count with zone_id, and columns in 2 levels
    # Loop through each att
    census_data = census_data.set_index(census_data.columns[census_data.columns.get_level_values(0)=="zone_id"][0])
    cols_drop = census_data.columns[census_data.columns.get_level_values(0).isin(["zone_id", "sample_geog"])]
    census_data = census_data.drop(columns= cols_drop)
    ls_atts_order = list(census_data.columns.get_level_values(0).unique()) #at the moment it is just by order from marginals file, will fix later

    syn_pop = None
    for att in ls_atts_order:
        if syn_pop is None: # first time run, this should be perfect
            syn_pop = samp_from_pool_1layer(pool, census_data, att, geo_lev)
        else:
            # Now we need to process from the syn pop
            dict_diff = cal_states_diff(att, syn_pop, census_data, geo_lev)
            # Doing zone by zone
            for zone in dict_diff:
                count_vals = dict_diff[zone]

            break


def main():
    census_data = pd.read_csv(os.path.join(data_dir, "hh_marginals_ipu.csv"), header=[0,1])
    df_seed = pd.read_csv(os.path.join(processed_data, "ori_sample_hh.csv"))
    # drop all the ids as they are not needed for in BN learning
    id_cols = [x for x in df_seed.columns if "hhid" in x or "persid" in x]
    df_seed = df_seed.drop(columns=id_cols)

    with open(os.path.join(processed_data, 'dict_hh_states.pickle'), 'rb') as handle:
        hh_state_names = pickle.load(handle)
    
    pool = get_pool(df_seed, hh_state_names)
    geo_lev = "POA"
    syn_pop = process_data_general(census_data, pool, geo_lev)


if __name__ == "__main__":
    main()