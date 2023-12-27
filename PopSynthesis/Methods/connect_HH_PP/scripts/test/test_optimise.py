import pandas as pd
import numpy as np
import os
from PopSynthesis.Methods.connect_HH_PP.scripts.sample_hh_main import *
from PopSynthesis.Methods.connect_HH_PP.scripts.process_all_hh_pp import *


def wrapper_get_pool(have_the_file=False):
    if have_the_file:
        pool = pd.read_csv("./pool_all_samples.csv")
    else:
        df_seed = pd.read_csv(os.path.join(processed_data, "ori_sample_hh.csv"))
        # drop all the ids as they are not needed for in BN learning
        id_cols = [x for x in df_seed.columns if "hhid" in x or "persid" in x]
        df_seed = df_seed.drop(columns=id_cols)

        with open(os.path.join(processed_data, 'dict_hh_states.pickle'), 'rb') as handle:
            hh_state_names = pickle.load(handle)
        
        pool = get_pool(df_seed, hh_state_names)
        pool.to_csv("./pool_all_samples.csv", index=False)
    return pool


def main():
    # Maybe create a tree structure to store the possible combination and then pass them down
    # We need to create a way to store those combinations
    # The idea of tree struct would be each level is a state
    # Maybe create based on dict
    pool = wrapper_get_pool(have_the_file=False)
    print(pool)


if __name__ == "__main__":
    main()
    