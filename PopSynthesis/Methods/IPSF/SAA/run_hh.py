"""Main place to run SAA for households synthesis"""


import pandas as pd
import numpy as np
import pickle
from PopSynthesis.Methods.IPSF.const import (
    data_dir,
    processed_dir,
    output_dir,
    zone_field,
    POOL_SIZE,
)
from PopSynthesis.Methods.IPSF.utils.pool_utils import create_pool
from PopSynthesis.Methods.IPSF.utils.synthetic_checked_census import adjust_kept_rec_match_census, get_diff_marg, convert_full_to_marg_count
from PopSynthesis.Methods.IPSF.SAA.SAA import SAA
import random
import time


def run_main() -> None:
    hh_marg = pd.read_csv(data_dir / "hh_marginals_ipu.csv", header=[0, 1])
    hh_marg = hh_marg.drop(columns=hh_marg.columns[hh_marg.columns.get_level_values(0)=="sample_geog"][0])
    hh_seed = pd.read_csv(data_dir / "hh_sample_ipu.csv")
    with open(processed_dir / "dict_hh_states.pickle", "rb") as handle:
        hh_att_state = pickle.load(handle)
    order_adjustment = [
        "hhsize",
        "hhinc",
        "totalvehs",
        "dwelltype",
        "owndwell",
    ]
    considered_atts = [
        "hhsize",
        "hhinc",
        "totalvehs",
        "dwelltype",
        "owndwell",
    ]
    hh_seed = hh_seed[order_adjustment]
    pool = create_pool(seed=hh_seed, state_names=hh_att_state, pool_sz=POOL_SIZE)

    start_time = time.time()

    n_run_time = 0
    n_removed_err_hh = np.inf
    MAX_RUN_TIME = 30
    chosen_hhs = []
    while n_run_time < MAX_RUN_TIME and n_removed_err_hh > 0:
        # randomly shuffle for each adjustment
        random.shuffle(order_adjustment)
        print(f"For run {n_run_time}, order is: {order_adjustment}, aim for {n_removed_err_hh} HHs")
        saa = SAA(hh_marg, considered_atts, order_adjustment, hh_att_state, pool)
        ###
        final_syn_pop = saa.run(extra_name=f"_{n_run_time}")
        ###
        # error check
        marg_from_kept_hh = convert_full_to_marg_count(
            final_syn_pop, [zone_field]
        )
        converted_hh_marg = hh_marg.set_index(hh_marg.columns[hh_marg.columns.get_level_values(0)==zone_field][0])
        diff_marg = get_diff_marg(converted_hh_marg, marg_from_kept_hh)
        
        kept_hh = adjust_kept_rec_match_census(final_syn_pop, diff_marg)

        # checking
        kept_marg = convert_full_to_marg_count(
            kept_hh, [zone_field]
        )
        new_diff_marg = get_diff_marg(converted_hh_marg, kept_marg)
        # check it is no neg indeed
        checking_not_neg = new_diff_marg < 0
        assert checking_not_neg.any(axis=None) == False
        # now get the new marg
        new_diff_marg.index = new_diff_marg.index.astype(int)
        new_diff_marg.index.name = zone_field
        hh_marg = new_diff_marg.reset_index()

        n_run_time += 1
        n_removed_err_hh = len(final_syn_pop) - len(kept_hh)
        if n_run_time == MAX_RUN_TIME:
            # not adjusting anymore
            chosen_hhs.append(final_syn_pop)
        else:
            # continue with adjusting for missing
            chosen_hhs.append(kept_hh)

    final_syn_hh = pd.concat(chosen_hhs)

    # record time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(rem, 60)  # 60 seconds in a minute
    print(f"Processing took {int(hours)}h-{int(minutes)}m-{seconds:.2f}s")

    # output
    final_syn_hh.to_csv(output_dir / "SAA_HH_only_looped.csv")


if __name__ == "__main__":
    run_main()
