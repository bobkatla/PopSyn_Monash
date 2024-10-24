# Contain all common functions used in the CSP run
import pickle
import pandas as pd
from PopSynthesis.Methods.IPSF.const import (
    data_dir,
    processed_dir,
    output_dir,
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
from PopSynthesis.Methods.IPSF.CSP.CSP import CSP_run, HHID
from typing import Tuple, Dict, List


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