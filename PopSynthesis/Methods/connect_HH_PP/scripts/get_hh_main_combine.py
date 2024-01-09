"""
We have the HH already (with the all the adjusted atts), now we need to run this script
to have the main pp df
then we can others rela

maybe I need to group the generating results into 1 place (census, samples, IPU and this, maybe we should get the normal BN as well)

"""

import pandas as pd
import numpy as np
from PopSynthesis.Methods.connect_HH_PP.paras_dir import processed_data, geo_lev, output_dir
from PopSynthesis.Methods.connect_HH_PP.scripts.const import *
from PopSynthesis.Methods.connect_HH_PP.scripts.sample_hh_main import *
from PopSynthesis.Methods.connect_HH_PP.scripts.sample_pp import *
from PopSynthesis.Methods.connect_HH_PP.scripts.process_all_hh_pp import *


def get_combine_df(hh_df):
    pp_state_names = None
    with open(os.path.join(processed_data, 'dict_pp_states.pickle'), 'rb') as handle:
        pp_state_names = pickle.load(handle)
    hh_state_names = None
    with open(os.path.join(processed_data, 'dict_hh_states.pickle'), 'rb') as handle:
        hh_state_names = pickle.load(handle)
    state_names = hh_state_names | pp_state_names

    #learning to get the HH only with main person
    df_seed = pd.read_csv(os.path.join(processed_data, "connect_hh_main.csv"))
    # drop all the ids as they are not needed for in BN learning
    id_cols = [x for x in df_seed.columns if "hhid" in x or "persid" in x]
    df_seed = df_seed.drop(columns=id_cols)
    pool_hh_main = get_pool(df_seed, state_names)
    pool_hh_main = pool_hh_main.astype(str)

    ls_match = list(hh_df.columns)
    ls_match.remove("POA")
    hh_df["hhid"] = hh_df.index
    hh_df = hh_df.astype(str)

    ls_df = []
    for poa in hh_df["POA"].unique():
        print(f"DOING {poa}")
        sub_df = hh_df[hh_df["POA"]==poa]
        combine = sub_df.merge(pool_hh_main, on=ls_match, how="left")
        combine = combine.sample(frac=1).drop_duplicates(subset='hhid')
        ls_df.append(combine)
    
    fin_df_all = pd.concat(ls_df)
    fin_df_all.to_csv("./my_fisrt_comb3.csv", index=False)


def main():
    hh_df = pd.read_csv(r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\PopSynthesis\Methods\connect_HH_PP\output\adjust\saving_hh_totalvehs.csv")
    combine_df = get_combine_df(hh_df)


if __name__ == "__main__":
    main()