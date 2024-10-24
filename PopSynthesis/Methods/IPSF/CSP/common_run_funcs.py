# Contain all common functions used in the CSP run
import pickle
import pandas as pd
from PopSynthesis.Methods.IPSF.const import (
    data_dir,
    small_test_dir,
    processed_dir,
    output_dir,
    zone_field,
    PP_ATTS,
    HH_TAG,
    NOT_INCLUDED_IN_BN_LEARN,
)
from PopSynthesis.Methods.IPSF.CSP.operations.sample_from_pairs import (
    update_by_rm_for_pool,
)
from PopSynthesis.Methods.IPSF.utils.synthetic_checked_census import (
    adjust_kept_rec_match_census,
    get_diff_marg,
    convert_full_to_marg_count,
)
from PopSynthesis.Methods.IPSF.CSP.CSP import CSP_run, HHID
from PopSynthesis.Methods.IPSF.SAA.SAA import SAA
from typing import Tuple, Dict, List


def get_cross_checked_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    # Get stored data
    print("Loading cross-checked data")
    with open(processed_dir / "dict_pool_pairs_check_HH_main.pickle", "rb") as handle:
        pools_ref = pickle.load(handle)

    syn_hh = pd.read_csv(
        output_dir / "SAA_HH_fixed_ad.csv", index_col=0
    ).reset_index(drop=True)
    syn_hh[HHID] = syn_hh.index

    hh_marg = pd.read_csv(data_dir / "hh_marginals_ipu.csv", header=[0, 1])
    hh_marg = hh_marg.drop(
        columns=hh_marg.columns[hh_marg.columns.get_level_values(0) == "sample_geog"][0]
    )

    hh_pool = pools_ref[HH_TAG]

    del pools_ref[HH_TAG]
    # removed HHs not existing in pools (normally if we used the same pool we would not need this step)
    # TODO

    return syn_hh, hh_pool, hh_marg, pools_ref


def get_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    # Get stored data
    print("Loading test data")
    syn_hh = pd.read_csv(
        small_test_dir / "SAA_HH_small.csv", index_col=0
    ).reset_index(drop=True)
    syn_hh[HHID] = syn_hh.index

    hh_pool = pd.read_csv(small_test_dir / "HH_pool_small_test.csv")

    hh_marg = pd.read_csv(small_test_dir / "hh_marginals_small.csv", header=[0, 1])

    with open(small_test_dir / "dict_pool_pairs_by_layers_small.pickle", "rb") as handle:
        pools_ref = pickle.load(handle)

    return syn_hh, hh_pool, hh_marg, pools_ref


def get_full_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    # Get stored data
    print("Loading full data")
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
    marg_from_kept_hh = convert_full_to_marg_count(final_syn_pop.drop(columns=[HHID], errors="ignore"), [zone_field])
    # get the marg from the original marg
    converted_hh_marg = hh_marg.set_index(
        hh_marg.columns[hh_marg.columns.get_level_values(0) == zone_field][0]
    )
    # get the diff
    diff_marg = get_diff_marg(converted_hh_marg, marg_from_kept_hh)
    # adjust the kept hh
    kept_hh = adjust_kept_rec_match_census(final_syn_pop, diff_marg)
    # checking
    kept_marg = convert_full_to_marg_count(kept_hh.drop(columns=[HHID], errors="ignore"), [zone_field])
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
        related_pools = [x for x in pools_ref.keys() if rela in x]
        check_cols = hh_atts if rela == "HH" else [f"{x}_{rela}" for x in pp_atts]
        for pool_name in related_pools:
            print(f"Updating {pool_name}")
            updated_pool_ref[pool_name] = update_by_rm_for_pool(error_rec, pools_ref[pool_name], check_cols)

    return updated_hh, updated_pp, updated_hh_marg, updated_pool_ref


def ipsf_full_loop(order_adjustment, syn_hh, hh_pool, hh_marg, pools_ref, max_run_time=30, output_each_step=False):
    # get attributes
    pp_atts = list(set(PP_ATTS) - set(NOT_INCLUDED_IN_BN_LEARN))
    hh_atts = [x for x in syn_hh.columns if x not in [zone_field, HHID]]
    all_rela = list(set([x.split("-")[-1] for x in pools_ref.keys()]))

    # Run the CSP - first run
    updated_syn_hh, syn_pp, hh_marg, pools_ref = update_CSP_combined_syn_hhmarg_pools(syn_hh, hh_marg, pools_ref, pp_atts, hh_atts, all_rela)

    # init with the total HH we want
    chosen_hhs = [updated_syn_hh]
    chosen_pp = [syn_pp]
    highest_id = updated_syn_hh[HHID].astype(int).max()
    left_over_hh = None

    n_removed_err = hh_marg.sum().sum() / len(order_adjustment)
    n_run_time = 0
    err_rm_hh = [n_removed_err]
    while n_run_time < max_run_time and n_removed_err > 0:
        # randomly shuffle for each adjustment
        print(
            f"For run {n_run_time}, order is: {order_adjustment}, aim for {n_removed_err} HHs"
        )
        saa = SAA(hh_marg, order_adjustment, order_adjustment, hh_pool)
        ###
        added_syn_hh = saa.run(output_each_step=output_each_step, extra_name=f"_IPSF_{n_run_time}")
        added_syn_hh[HHID] = range(highest_id+1, highest_id+len(added_syn_hh)+1)
        ###
        # error check
        new_syn_hh, new_syn_pp, hh_marg, pools_ref = update_CSP_combined_syn_hhmarg_pools(added_syn_hh, hh_marg, pools_ref, pp_atts, hh_atts, all_rela)

        # added the adjusted ones
        chosen_hhs.append(new_syn_hh)
        chosen_pp.append(new_syn_pp)

        highest_id = new_syn_hh[HHID].astype(int).max()
        n_run_time += 1
        n_removed_err = len(added_syn_hh) - len(new_syn_hh)
        err_rm_hh.append(n_removed_err)

        if n_run_time == max_run_time and n_removed_err > 0:
            # not adjusting anymore
            left_over_hh = added_syn_hh[~added_syn_hh[HHID].astype(str).isin(new_syn_hh[HHID])]
            assert len(left_over_hh) == n_removed_err

    if left_over_hh is not None: # meaning the there are some hh withour pp assigned
        # run the last CSP
        syn_hh, syn_pp, _ = CSP_run(left_over_hh, pools_ref, pp_atts, hh_atts, all_rela)
        chosen_hhs.append(syn_hh)
        chosen_hhs.append(syn_pp)
        remaining_unadjustable_hh = len(left_over_hh) - len(syn_hh)
        if remaining_unadjustable_hh > 0:
            print(f"WARNING: There are {remaining_unadjustable_hh} HHs that cannot assign people to")
        else:
            print("All HHs are adjusted and have pp assigned")

    final_syn_hh = pd.concat(chosen_hhs)
    final_syn_pp = pd.concat(chosen_pp)
    return final_syn_hh, final_syn_pp, err_rm_hh
