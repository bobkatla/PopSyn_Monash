"""Run the CSP from a given syn HH (also SAA to adjust again)"""


import pickle
import pandas as pd
from PopSynthesis.Methods.IPSF.const import (
    processed_dir,
    output_dir,
    PP_ATTS,
    HH_TAG,
    NOT_INCLUDED_IN_BN_LEARN,
    zone_field,
)
from PopSynthesis.Methods.IPSF.CSP.operations.sample_from_pairs import (
    sample_matching_from_pairs,
    create_count_col,
    update_by_rm_for_pool,
    update_by_rm_for_all_pools,
)
import numpy as np
from typing import Literal


ordered_pairs =[
    [
        ("HH", "Main")
    ],
    [
        ("Main", "Spouse"), 
        ("Main", "Child"), 
        ("Main", "Parent"),
        ("Main", "Sibling"),
        ("Main", "Others")
    ], 
    [
        ("Child", "Grandchild"), 
        ("Parent", "Grandparent")
    ]
]


def selecting_key_pp_to_sample(pp: pd.DataFrame, hhid:str, age_col:str, strategy:Literal["oldest", "youngest", "random"]) -> pd.DataFrame:
    """Filter the pp so each hh only have 1 pp so it can be used for sampling later"""
    pp = pp.copy(deep=True).reset_index(drop=True)
    pp["ppid"] = pp.index
    pp["converted_age"] = pp[age_col].apply(lambda x: int(x.split("-")[0].replace("+", "")))
    gb_hhid = pp.groupby(hhid)[["ppid", "converted_age"]].apply(lambda x: list(x.to_numpy()))
    gb_hhid = gb_hhid.apply(lambda x: sorted(x, key=lambda x: x[1]))
    if strategy == "oldest":
        gb_hhid = gb_hhid.apply(lambda x: x[-1][0])
    elif strategy == "youngest":
        gb_hhid = gb_hhid.apply(lambda x: x[0][0])
    elif strategy == "random":
        gb_hhid = gb_hhid.apply(lambda x: x[np.random.randint(0, len(x))][0])
    else:
        raise ValueError("Strategy not found")
    return pp[pp["ppid"].isin(gb_hhid)].drop(columns=["ppid", "converted_age"])


def main():
    # TODO: is there anyway to use pp marg
    HHID = "hhid"
    main_rela = "Main"

    syn_hh = pd.read_csv(
        output_dir / "SAA_output_HH_again.csv", index_col=0
    ).reset_index(drop=True)
    syn_hh["hhid"] = syn_hh.index

    with open(processed_dir / "dict_pool_pairs_by_layers.pickle", "rb") as handle:
        pools_ref = pickle.load(handle)
    hh_pool = pd.read_csv(processed_dir / "HH_pool.csv")
    
    # get attributes
    pp_atts = list(set(PP_ATTS) - set(NOT_INCLUDED_IN_BN_LEARN))
    hh_atts = [x for x in syn_hh.columns if x not in [zone_field, HHID]]
    all_rela = [x.split("-")[-1] for x in pools_ref.keys()]

    # NOTE: the syn main people will be updated with the new values, it is the first val in the array
    syn_results = {HH_TAG: syn_hh}
    # Will do again, will do synthesis by levels to ensure the age order
    for layer in ordered_pairs:
        for root_rela, sample_rela in layer:
            pool_name = f"{root_rela}-{sample_rela}"
            sample_cols = [f"{x}_{sample_rela}" for x in pp_atts]

            if root_rela == HH_TAG:
                to_process_syn = syn_hh
                sample_cols = sample_cols + all_rela
                evidence_cols = hh_atts
            else:
                to_process_syn = syn_results[root_rela]
                to_process_syn.loc[:, sample_rela] = list(to_process_syn[sample_rela].astype(int))
                to_process_syn = to_process_syn[to_process_syn[sample_rela] > 0]
                to_process_syn = create_count_col(to_process_syn, sample_rela)
                evidence_cols = [f"{x}_{root_rela}" for x in pp_atts]
            
            # process to_process_syn if duplicated hhid
            if to_process_syn[HHID].duplicated().any():
                to_process_syn = selecting_key_pp_to_sample(to_process_syn, HHID, "age", "random")

            rela_pp, removed_syn, _ = sample_matching_from_pairs(
                given_syn=to_process_syn,
                syn_id=HHID,
                paired_pool=pools_ref[pool_name],
                evidence_cols=evidence_cols,
                sample_cols=sample_cols,
            )
            
            # speical process for second layer
            if sample_rela == "Parent":
                map_dict = dict(zip(to_process_syn[HHID], to_process_syn["Grandparent"]))
                rela_pp["Grandparent"] = rela_pp[HHID].map(map_dict)
            elif sample_rela == "Child":
                map_dict = dict(zip(to_process_syn[HHID], to_process_syn["Grandchild"]))
                rela_pp["Grandchild"] = rela_pp[HHID].map(map_dict)
                
            rela_pp["relationship"] = sample_rela
            syn_results[sample_rela] = rela_pp
            rm_hhid = list(removed_syn[HHID])
            syn_results[root_rela] = syn_results[root_rela][~syn_results[root_rela][HHID].isin(rm_hhid)]
    
    # We need to concat them
    temp_pp = []
    for rela, df in syn_results.items():
        if rela != HH_TAG:
            rename_rela = {f"{x}_{rela}": x for x in pp_atts}
            temp_pp.append(df.rename(columns=rename_rela).drop(columns=all_rela, errors="ignore"))
    final_pp = pd.concat(temp_pp, ignore_index=True)

if __name__ == "__main__":
    main()
