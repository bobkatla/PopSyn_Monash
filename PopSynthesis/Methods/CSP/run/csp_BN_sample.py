"""Using BN to sample should quick for each step, but the conditional need to think abit"""

# need the possible states for each atts, also including the min and max for each count (n_rela)
# process the given dict of conditionals (seed matching with count) to do the fitting for BN
# dertermine the n_rela for each hh using BN (merge later), need conditional sampling and then merge
# 
import pandas as pd
import random
from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score, learn_para_BN
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
from pgmpy.base import DAG
from PopSynthesis.Methods.CSP.run.rela_const import HH_TAG, RELA_BY_LEVELS, BACK_CONNECTIONS, EXPECTED_RELATIONSHIPS, COUNT_COL
from PopSynthesis.Methods.CSP.const import HHID, PP_ATTS
from typing import Dict, List, Callable


def model_sample(model: DAG, evidences: pd.DataFrame, target_cols: List[str], func_check: Callable ) -> pd.DataFrame:
    # groupby evidences into different combinations and count
    assert HHID in evidences.columns, "The evidences must contain the HHID column"
    assert COUNT_COL not in evidences.columns, "The given evidences must be full population"
    evidence_cols = [x for x in evidences.columns if x != HHID]
    processed_evidences = evidences.groupby(evidence_cols)[HHID].apply(list).reset_index()
    processed_evidences[COUNT_COL] = processed_evidences[HHID].apply(len)

    # loop through each to sample
    inference = BayesianModelSampling(model)
    sampled_dfs = []
    for _, row in processed_evidences.iterrows():
        # get the evidence for this row
        sample_evidences = [State(att, state) for att, state in row[evidence_cols].items()]
        # sample from the model
        sampled_results = inference.likelihood_weighted_sample(evidence=sample_evidences, size=row[COUNT_COL])
        random.shuffle(row[HHID])
        sampled_results[HHID] = row[HHID]
        # do the resample impossibles here
        sampled_dfs.append(sampled_results)
    # concat and return
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    # check the sampled df
    assert len(sampled_df) == len(evidences), "The sampled df must have the same count as the evidences"
    assert set(sampled_df[HHID]) == set(evidences[HHID]), "The sampled df must have the same hhid as the evidences"
    return sampled_df[target_cols].copy()


def build_models_for_each_connection(conditional: Dict[str, pd.DataFrame], possible_states: Dict[str, List[str]]) -> Dict[str, DAG]:
    # create the models for each connection using possible states
    n_rela_cols = [f"n_{rela}" for rela in EXPECTED_RELATIONSHIPS]
    results = {}
    for conn, df in conditional.items():
        # get the states for each att
        states = {}
        for att in df.columns:
            if att == COUNT_COL or att == HHID:
                continue
            actual_att = att if att in n_rela_cols else att.split("_")[1]
            states[att] = possible_states[actual_att]
        # create the model
        processed_df = df.drop(columns=[HHID], errors='ignore').rename(columns={COUNT_COL: "_weight"})
        model = learn_struct_BN_score(processed_df, state_names=states, show_struct=False)
        model = learn_para_BN(model, processed_df, state_names=states)
        results[conn] = model
    return results


def sample_rela_BN(hh_df: pd.DataFrame, final_conditonals: pd.DataFrame, hhsz: str, relationship: str, possible_states:Dict[str, List[str]]=None) -> pd.DataFrame:
    # go through each and sample, sample directly from the model
    # for some cases we need to resample the impossible cases
    n_rela_cols = [f"n_{rela}" for rela in EXPECTED_RELATIONSHIPS]
    hh_df = hh_df.rename(columns={col: f"{HH_TAG}_{col}" for col in hh_df.columns if col != HHID})
    hh_counts_cond = final_conditonals[f"{HH_TAG}-counts"]
    # Special handle for n relas to update the possible states
    n_counts_states = {}
    for rela in EXPECTED_RELATIONSHIPS:
        rela_count_states = list(range(hh_counts_cond[f"n_{rela}"].min(), hh_counts_cond[f"n_{rela}"].max() + 1))
        n_counts_states[f"n_{rela}"] = rela_count_states
    update_possible_states = {**possible_states, **n_counts_states}
    
    # create the models
    conn_models = build_models_for_each_connection(final_conditonals, update_possible_states)

    processed_hh_df = direct_model_sample(conn_models[f"{HH_TAG}-counts"], hh_df, n_rela_cols+hh_df.columns.tolist())
    print(processed_hh_df)
    evidences_store = {HH_TAG: processed_hh_df}
    
    raise NotImplementedError("This function is not implemented yet")

###### here some funcs to check the impossible cases
def check_hhsz_mismatch(df: pd.DataFrame, hhsz: str) -> bool:
    """Check if the hh size is mismatch with the df"""
    NotImplemented