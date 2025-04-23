"""This should be easier
with each connection we build a model
each model will have the ability to sample from the knowing (we can setup to say all possible values for each att)
we can assume all possible states from seed
then we build the models
then we sample given evidence for each att
So to do that we need to do groupby to have the syn_count
BN will be just forward sampling not backward to update (not needed)
turn out the key is simply adding the count into sampling
"""
"""To sample one way, i.e. no loop back, we use the n_rela count as well"""
import pandas as pd
from PopSynthesis.Methods.CSP.const import HHID
from PopSynthesis.Methods.CSP.run.rela_const import HH_TAG, RELA_BY_LEVELS, BACK_CONNECTIONS
from PopSynthesis.Methods.CSP.run.sample_utils import (
    TARGET_ID,
    SYN_COUNT_COL,
    determine_n_rela_for_each_hh,
    direct_sample_from_conditional,
    handle_resample_and_update_possible_df,
    merge_chosen_target_ids_with_known_cond
)

def sample_one_way(hh_df, final_conditonals, hhsz, relationship):
    """Sample one way from the hh df"""
    # the n rela will be determined as normal
    # process each conditionals to have target id
    hh_df = hh_df.rename(columns={col: f"{HH_TAG}_{col}" for col in hh_df.columns if col != HHID})
    for key in final_conditonals.keys():
        final_conditonals[key] = final_conditonals[key].reset_index(drop=True) # ensure we have a clean index
        final_conditonals[key][TARGET_ID] = final_conditonals[key].index + 1
    # print("Processing hh_df... to get the n_rela for each hh")
    processed_hh_df = determine_n_rela_for_each_hh(hh_df.copy(), hhsz, final_conditonals[f"{HH_TAG}-counts"])
    assert processed_hh_df[HHID].nunique() == len(hh_df), "Processed hh df must have same hhid as original hh df"
    processed_hh_df[f"n_{HH_TAG}"] = 1 # just for compleness
    print(processed_hh_df)
    raise ValueError("DEBUG")