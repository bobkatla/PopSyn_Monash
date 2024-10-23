"""Run the CSP from a given syn HH (also SAA to adjust again)"""


import pickle
import pandas as pd
from PopSynthesis.Methods.IPSF.const import (
    data_dir,
    processed_dir,
    output_dir,
    PP_ATTS,
    NOT_INCLUDED_IN_BN_LEARN,
    zone_field,
)
from PopSynthesis.Methods.IPSF.CSP.operations.sample_from_pairs import (
    update_by_rm_for_pool,
)
from PopSynthesis.Methods.IPSF.utils.synthetic_checked_census import (
    adjust_kept_rec_match_census,
    get_diff_marg,
    convert_full_to_marg_count,
)
from PopSynthesis.Methods.IPSF.SAA.SAA import SAA
from PopSynthesis.Methods.IPSF.CSP.CSP import CSP_run, HHID
from typing import Tuple, Dict, List
import time


def get_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    # Get stored data
    print("Loading data")
    syn_hh = pd.read_csv(
        output_dir / "SAA_HH_fixed_ad.csv", index_col=0
    ).reset_index(drop=True)
    syn_hh[HHID] = syn_hh.index

    hh_pool = pd.read_csv(processed_dir / "HH_pool.csv")

    hh_marg = pd.read_csv(data_dir / "hh_marginals_ipu.csv", header=[0, 1])
    hh_marg = hh_marg.drop(
        columns=hh_marg.columns[hh_marg.columns.get_level_values(0) == "sample_geog"][0]
    )

    with open(processed_dir / "dict_pool_pairs_by_layers.pickle", "rb") as handle:
        pools_ref = pickle.load(handle)

    return syn_hh, hh_pool, hh_marg, pools_ref


def get_remaining_hh_n_new_marg(hh_marg: pd.DataFrame, final_syn_pop: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # get the marg from the kept hh
    marg_from_kept_hh = convert_full_to_marg_count(final_syn_pop, [zone_field])
    # get the marg from the original marg
    converted_hh_marg = hh_marg.set_index(
        hh_marg.columns[hh_marg.columns.get_level_values(0) == zone_field][0]
    )
    # get the diff
    diff_marg = get_diff_marg(converted_hh_marg, marg_from_kept_hh)
    # adjust the kept hh
    kept_hh = adjust_kept_rec_match_census(final_syn_pop, diff_marg)
    # checking
    kept_marg = convert_full_to_marg_count(kept_hh, [zone_field])
    new_diff_marg = get_diff_marg(converted_hh_marg, kept_marg)
    # check it is no neg indeed
    checking_not_neg = new_diff_marg < 0
    assert checking_not_neg.any(axis=None) == False
    # now get the new marg
    new_diff_marg.index = new_diff_marg.index.astype(int)
    new_diff_marg.index.name = zone_field
    return kept_hh, new_diff_marg.reset_index()


def update_CSP_combined_syn_hhmarg_pools(syn_hh: pd.DataFrame, hh_marg: pd.DataFrame, pools_ref: Dict[str, pd.DataFrame], pp_atts: List[str], hh_atts: List[str], all_rela: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    syn_hh, syn_pp, error_recs = CSP_run(syn_hh, pools_ref, pp_atts, hh_atts, all_rela)
    updated_hh, updated_hh_marg = get_remaining_hh_n_new_marg(hh_marg, syn_hh)
    updated_pp = syn_pp[syn_pp[HHID].isin(updated_hh[HHID])]
    updated_pool_ref = {}
    # update pools by error recs
    print("Updating pools")
    for rela, error_rec in error_recs.items():
        check_cols = hh_atts if rela == "HH" else [f"{x}_{rela}" for x in pp_atts]
        updated_pool_ref[rela] = update_by_rm_for_pool(error_rec, pools_ref[rela], check_cols)

    return updated_hh, updated_pp, updated_hh_marg, updated_pool_ref


def main():
    # TODO: is there anyway to use pp marg
    syn_hh, hh_pool, hh_marg, pools_ref = get_all_data()
    
    # get attributes
    pp_atts = list(set(PP_ATTS) - set(NOT_INCLUDED_IN_BN_LEARN))
    hh_atts = [x for x in syn_hh.columns if x not in [zone_field, HHID]]
    all_rela = list(set([x.split("-")[-1] for x in pools_ref.keys()]))

    # Run the CSP - first run
    updated_syn_hh, syn_pp, hh_marg, pools_ref = update_CSP_combined_syn_hhmarg_pools(syn_hh, hh_marg, pools_ref, pp_atts, hh_atts, all_rela)

    order_adjustment = [
        "hhsize",
        "hhinc",
        "totalvehs",
        "dwelltype",
        "owndwell",
    ]  # these must exist in both marg and syn

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
