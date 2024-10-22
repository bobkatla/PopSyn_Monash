"""
Here defind the object of CSP that will take in synthetic households to generate people

Inputs: 
This will create the pools for households and people (by each pairs)
Go through pair hh-main to get main (ensure the main people match all exist in pools)
Then main with other rela and update the pools of main and other rela as well
Maybe need pools for people (in general) and pools for households, we then can cross check
we then will update the households again (cross check with census and re-run it)
We will add the loop here as well (the longest is the fisrt SAA which is seperately)
"""
import pandas as pd
from PopSynthesis.Methods.IPSF.const import HH_TAG
from PopSynthesis.Methods.IPSF.CSP.operations.sample_from_pairs import (
    sample_matching_from_pairs,
    create_count_col,
)
import numpy as np
from typing import Literal, Dict, Tuple

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

HHID = "hhid"


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


def CSP_run(syn_hh: pd.DataFrame, pools_pp: dict, pp_atts: list, hh_atts: list, all_rela: list) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Run the CSP for the given synthetic households and pools"""

    syn_results = {HH_TAG: syn_hh}
    removed_recs = {}
    
    for layer in ordered_pairs:
        for root_rela, sample_rela in layer:
            print(f"Processing {root_rela} - {sample_rela}")
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
                to_process_syn = selecting_key_pp_to_sample(to_process_syn, HHID, f"age_{root_rela}", "random")

            rela_pp, removed_syn, _ = sample_matching_from_pairs(
                given_syn=to_process_syn,
                syn_id=HHID,
                paired_pool=pools_pp[pool_name],
                evidence_cols=evidence_cols,
                sample_cols=sample_cols,
            )

            # Update removed syn
            if root_rela in removed_recs:
                removed_recs[root_rela].append(removed_syn)
            else:
                removed_recs[root_rela] = [removed_syn]
            
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
    
    print("Concatenating results and output")
    # We need to concat them
    temp_pp = []
    for rela, df in syn_results.items():
        if rela != HH_TAG:
            rename_rela = {f"{x}_{rela}": x for x in pp_atts}
            temp_pp.append(df.rename(columns=rename_rela).drop(columns=all_rela, errors="ignore"))
    final_pp = pd.concat(temp_pp, ignore_index=True)
    
    # return impossible combinations by values
    for rela, removed in removed_recs.items():
        removed_recs[rela] = pd.concat(removed, ignore_index=True)
    final_hh = syn_results[HH_TAG]
    final_hh.loc[:, HHID] = list(final_hh[HHID].astype(str))
    final_hh = final_hh[final_hh[HHID].isin(list(final_pp[HHID]))]

    return final_hh, final_pp, removed_recs


