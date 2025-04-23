"""Using BN to sample should quick for each step, but the conditional need to think abit"""

import pandas as pd
import random
from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score, learn_para_BN
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
from pgmpy.base import DAG
from PopSynthesis.Methods.CSP.run.rela_const import HH_TAG, RELA_BY_LEVELS, BACK_CONNECTIONS, EXPECTED_RELATIONSHIPS, COUNT_COL
from PopSynthesis.Methods.CSP.run.sample_utils import SYN_COUNT_COL
from PopSynthesis.Methods.CSP.const import HHID, PP_ATTS
from typing import Dict, List, Callable
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


N_RELA_COLS = [f"n_{rela}" for rela in EXPECTED_RELATIONSHIPS]


def model_sample(model: DAG, evidences: pd.DataFrame, target_cols: List[str], func_check: Callable) -> pd.DataFrame:
    # groupby evidences into different combinations and count
    assert HHID in evidences.columns, "The evidences must contain the HHID column"
    assert SYN_COUNT_COL in evidences.columns, "The evidences must contain the SYN_COUNT_COL column"
    # Multiple hhid, choose the first one only, TODO: may need to rethink this later
    evidences = evidences[~evidences[HHID].duplicated(keep='first')]
    # Multiple outputs, we can just increase the hhid and create list and explode
    evidences[HHID] = evidences.apply(lambda x: [x[HHID]] * x[SYN_COUNT_COL], axis=1)
    evidences = evidences.explode(HHID).drop(columns=[SYN_COUNT_COL])
  
    evidence_cols = [x for x in evidences.columns if x != HHID]
    processed_evidences = evidences.groupby(evidence_cols)[HHID].apply(list).reset_index()

    # loop through each to sample
    inference = BayesianModelSampling(model)
    sampled_dfs = []
    for _, row in processed_evidences.iterrows():
        # get the evidence for this row
        sample_evidences = [State(att, state) for att, state in row[evidence_cols].items()]
        comb_correct_results = []
        # sample from the model
        hhid_ls = row[HHID]
        random.shuffle(hhid_ls)
        while True:
            sampled_results = inference.likelihood_weighted_sample(evidence=sample_evidences, size=len(hhid_ls))
            sampled_results[HHID] = hhid_ls
            if func_check is None:
                comb_correct_results.append(sampled_results)
                break
            check_possible = func_check(sampled_results.copy())
            correct_samples = sampled_results[check_possible]
            incorrect_samples = sampled_results[~check_possible]
            if len(correct_samples) > 0:
                comb_correct_results.append(correct_samples)
            if len(incorrect_samples) == 0:
                break
            hhid_ls = incorrect_samples[HHID].tolist()
        final_sampled_df = pd.concat(comb_correct_results, ignore_index=True)
        sampled_dfs.append(final_sampled_df)
    # concat and return
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    # check the sampled df
    assert len(sampled_df) == len(evidences), "The sampled df must have the same count as the processed evidences"
    assert set(sampled_df[HHID]) == set(evidences[HHID]), "The sampled df must have the same hhid as the evidences"
    return sampled_df[target_cols].copy()


def build_models_for_each_connection(conditional: Dict[str, pd.DataFrame], possible_states: Dict[str, List[str]]) -> Dict[str, DAG]:
    # create the models for each connection using possible states
    results = {}
    for conn, df in conditional.items():
        # get the states for each att
        states = {}
        for att in df.columns:
            if att == COUNT_COL or att == HHID:
                continue
            actual_att = att if att in N_RELA_COLS else att.split("_")[1]
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
    hh_df = hh_df.rename(columns={col: f"{HH_TAG}_{col}" for col in hh_df.columns if col != HHID})
    hh_att_cols = [col for col in hh_df.columns]
    hh_counts_cond = final_conditonals[f"{HH_TAG}-counts"]
    # Special handle for n relas to update the possible states
    n_counts_states = {}
    for rela in EXPECTED_RELATIONSHIPS:
        rela_count_states = list(range(hh_counts_cond[f"n_{rela}"].min(), hh_counts_cond[f"n_{rela}"].max() + 1))
        n_counts_states[f"n_{rela}"] = rela_count_states
    update_possible_states = {**possible_states, **n_counts_states}
    
    # create the models
    conn_models = build_models_for_each_connection(final_conditonals, update_possible_states)

    hh_df[SYN_COUNT_COL] = 1 # init all is one
    processed_hh_df = model_sample(conn_models[f"{HH_TAG}-counts"], hh_df, N_RELA_COLS + hh_att_cols, check_hhsz_mismatch)
    expected_n_pp = processed_hh_df[N_RELA_COLS].sum().sum()

    evidences_store = {HH_TAG: processed_hh_df}
    for relationships in RELA_BY_LEVELS:
        # Doing the relationship in this level
        for rela in relationships:
            for prev_rela in BACK_CONNECTIONS[rela]:
                print(f"Sampling {rela} from {prev_rela}...")
                rela_results = []
                if prev_rela not in evidences_store:
                    # Somehow this area does not have the prev rela
                    continue
                evidences = evidences_store[prev_rela]
                check = (evidences[f"n_{rela}"] > 0)
                if f"n_{prev_rela}" in evidences.columns:
                    check = check & (evidences[f"n_{prev_rela}"] > 0)
                evidences = evidences[check].copy() # only care where we need to sample
                if len(evidences) == 0:
                    continue
                evidences[SYN_COUNT_COL] = evidences[f"n_{rela}"]
                sampled_df = model_sample(conn_models[f"{prev_rela}-{rela}"], evidences, N_RELA_COLS + [f"{rela}_{att}" for att in PP_ATTS] + [HHID], None)
                rela_results.append(sampled_df)
            if len(rela_results) == 0:
                continue
            evidences_store[rela] = pd.concat(rela_results, ignore_index=True)
            assert len(evidences_store[rela]) == processed_hh_df[f"n_{rela}"].sum(), f"Must be able to sample all for this {rela}"

    concat_pp_ls = []
    for rela, df in evidences_store.items():
        if rela == HH_TAG:
            continue
        df = df.drop(columns=N_RELA_COLS, errors='ignore')
        df[relationship] = rela
        df = df.rename(columns={f"{rela}_{att}": att for att in PP_ATTS})
        concat_pp_ls.append(df)
    final_pp = pd.concat(concat_pp_ls, ignore_index=True)
    assert len(final_pp) == expected_n_pp, f"Final syn pp must match from HH, but got {len(final_pp)} vs {expected_n_pp}"
    return final_pp


###### here some funcs to check the impossible cases
def check_hhsz_mismatch(syn_hh_df: pd.DataFrame) -> pd.Series:
    """Check if the hh size is mismatch with the df"""
    hhsz = "HH_hhsize" # TODO: hardcoded here, find another way later
    assert hhsz in syn_hh_df.columns, "The df must contain the hh size column"
    syn_hh_df["syn_hhsize"] = syn_hh_df[N_RELA_COLS].sum(axis=1)
    def check_hhsz(row):
        if row[hhsz] == "8+":
            return row["syn_hhsize"] >= 8
        else:
            return int(row[hhsz]) == row["syn_hhsize"]
    return syn_hh_df.apply(check_hhsz, axis=1)