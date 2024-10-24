"""Run the CSP from a given syn HH (also SAA to adjust again)"""


import pandas as pd
from PopSynthesis.Methods.IPSF.const import (
    output_dir,
    PP_ATTS,
    NOT_INCLUDED_IN_BN_LEARN,
    zone_field,
)
from PopSynthesis.Methods.IPSF.CSP.common_run_funcs import get_all_data, update_CSP_combined_syn_hhmarg_pools
from PopSynthesis.Methods.IPSF.SAA.SAA import SAA
from PopSynthesis.Methods.IPSF.CSP.CSP import HHID
import time


# For SAA
order_adjustment = [
        "hhsize",
        "hhinc",
        "totalvehs",
        "dwelltype",
        "owndwell",
    ]  # these must exist in both marg and syn


def main():
    # TODO: is there anyway to use pp marg
    syn_hh, hh_pool, hh_marg, pools_ref = get_all_data()
    
    # get attributes
    pp_atts = list(set(PP_ATTS) - set(NOT_INCLUDED_IN_BN_LEARN))
    hh_atts = [x for x in syn_hh.columns if x not in [zone_field, HHID]]
    all_rela = list(set([x.split("-")[-1] for x in pools_ref.keys()]))

    # Run the CSP - first run
    updated_syn_hh, syn_pp, hh_marg, pools_ref = update_CSP_combined_syn_hhmarg_pools(syn_hh, hh_marg, pools_ref, pp_atts, hh_atts, all_rela)

    

    start_time = time.time()
    n_run_time = 0
    # init with the total HH we want
    n_removed_err_hh = hh_marg.sum().sum() / len(order_adjustment)
    MAX_RUN_TIME = 30
    chosen_hhs = [updated_syn_hh]
    chosen_pp = [syn_pp]
    err_rm_hh = []
    highest_id = updated_syn_hh[HHID].astype(int).max()
    while n_run_time < MAX_RUN_TIME and n_removed_err_hh > 0:
        # randomly shuffle for each adjustment
        err_rm_hh.append(n_removed_err_hh)
        print(
            f"For run {n_run_time}, order is: {order_adjustment}, aim for {n_removed_err_hh} HHs"
        )
        saa = SAA(hh_marg, order_adjustment, order_adjustment, hh_pool)
        ###
        added_syn_hh = saa.run(extra_name=f"_IPSF_{n_run_time}")
        added_syn_hh[HHID] = range(highest_id+1, highest_id+len(added_syn_hh)+1)
        ###
        # error check
        new_syn_hh, new_syn_pp, hh_marg, pools_ref = update_CSP_combined_syn_hhmarg_pools(added_syn_hh, hh_marg, pools_ref, pp_atts, hh_atts, all_rela)

        n_run_time += 1
        n_removed_err_hh = len(added_syn_hh) - len(new_syn_hh)
        if n_run_time == MAX_RUN_TIME:
            # not adjusting anymore
            chosen_hhs.append(added_syn_hh)
        else:
            # continue with adjusting for missing
            chosen_hhs.append(new_syn_hh)
        chosen_pp.append(new_syn_pp)

    final_syn_hh = pd.concat(chosen_hhs)
    final_syn_pp = pd.concat(chosen_pp)


    # record time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(rem, 60)  # 60 seconds in a minute
    print(f"IPSF took {int(hours)}h-{int(minutes)}m-{seconds:.2f}s")
    print(f"Error hh rm are: {err_rm_hh}")

    # output
    final_syn_hh.to_csv(output_dir / "IPSF_HH.csv")
    final_syn_pp.to_csv(output_dir / "IPSF_PP.csv")
   

if __name__ == "__main__":
    main()
