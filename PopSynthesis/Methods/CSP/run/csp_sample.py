"""CSP core: sample by each relationship and ensure the household size matching"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from PopSynthesis.Methods.CSP.run.rela_const import (
    HH_TAG,
    MAIN_PERSON,
    COUNT_COL,
    EXPECTED_RELATIONSHIPS,
    EPXECTED_CONNECTIONS,
    BACK_CONNECTIONS,
    RELA_BY_LEVELS
)
from PopSynthesis.Methods.CSP.const import HHID

TEMP_ID = "temp_id"
SYN_COUNT_COL = "syn_count"
MAP_IDS_COL = "potential_temp_ids"
MAP_COUNTS_COL = "asscociated_counts"


def split_and_process_by_temp_id(df: pd.DataFrame, evidence_cols: List[str], temp_id: str) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Split the df by temp_id and return the evidence and sample df"""
    assert temp_id in df.columns, f"{temp_id} must be in df columns"
    evidences_cond = df[evidence_cols + [temp_id, COUNT_COL]].copy()
    sample_cond = df[[x for x in df.columns if x not in (evidence_cols + [COUNT_COL])]].copy()
    evidences_to_ids = evidences_cond.groupby(evidence_cols)[temp_id].apply(list)
    evidences_to_counts = evidences_cond.groupby(evidence_cols)[COUNT_COL].apply(list)
    return evidences_to_ids, evidences_to_counts, sample_cond


def get_potentials_to_sample(conditionals: pd.DataFrame, evidences: pd.DataFrame, can_sample_all: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process conditionals by having it evidence as index and process the remainings"""
    assert COUNT_COL in conditionals.columns, "Count column must be in conditionals"

    evidence_cols = [x for x in evidences.columns if x != HHID]

    assert set(evidence_cols).issubset(set(conditionals.columns)), "Evidences must be subset of conditionals"

    evidences_to_ids, evidences_to_counts, sample_cond = split_and_process_by_temp_id(conditionals, evidence_cols, TEMP_ID)
    assert len(evidences_to_ids) == len(evidences_to_counts), "Evidences to ids and counts must be same length"
    assert set(evidences_to_ids.index) == set(evidences_to_counts.index), "Evidences to ids and counts must be same things"

    # count the evidences occurences
    evidences_hhid = evidences.groupby(evidence_cols)[HHID].apply(list)
    evidences_counts = evidences[evidence_cols].value_counts()
    if can_sample_all:
        assert set(evidences_counts.index) <= set(evidences_to_ids.index), "Given evidences must be subset of condtionals ids" 
        assert set(evidences_counts.index) <= set(evidences_to_counts.index), "Counts must be subset of condtionals counts"

    to_sample_df = evidences_hhid.reset_index(name=HHID)
    to_sample_df[SYN_COUNT_COL] = evidences_hhid.index.map(evidences_counts)
    to_sample_df[MAP_IDS_COL] = evidences_hhid.index.map(evidences_to_ids)
    to_sample_df[MAP_COUNTS_COL] = evidences_hhid.index.map(evidences_to_counts)

    # split to_sample_df by NA vals
    possible_to_sample_df = to_sample_df[to_sample_df[MAP_IDS_COL].notna()]
    impossible_to_sample_df = to_sample_df[to_sample_df[MAP_IDS_COL].isna()]

    if can_sample_all:
        assert len(impossible_to_sample_df) == 0, "Impossible to sample df must be empty"
    assert len(possible_to_sample_df) + len(impossible_to_sample_df) == len(evidences_counts), "Results must be same length as original evidences"

    return possible_to_sample_df, impossible_to_sample_df, sample_cond


def sample_and_choose_temp_ids(possible_to_sample: pd.DataFrame) -> pd.DataFrame:
    def sample_by_row(row):
        ids = row[MAP_IDS_COL]
        counts = row[MAP_COUNTS_COL]
        probs = [count / sum(counts) for count in counts]
        n = row[SYN_COUNT_COL]
        return np.random.choice(ids, size=n, p=probs, replace=True)
    
    # Apply to get the sampled ids
    to_sample_df = possible_to_sample.copy()
    to_sample_df[TEMP_ID] = to_sample_df.apply(sample_by_row, axis=1)

    return to_sample_df


def merge_chosen_temp_ids_with_known_cond(to_sample_df: pd.DataFrame, sample_cond: pd.DataFrame, agg_ids: bool) -> pd.DataFrame:
    # merge with known conds to get the final sampled df
    if agg_ids:
        # the case of HHID are list
        to_sample_df['zip_cols'] = to_sample_df.apply(lambda row: list(zip(row[HHID], row[TEMP_ID])), axis=1)
        to_sample_df = to_sample_df.explode('zip_cols')
        to_sample_df[[HHID, TEMP_ID]] = pd.DataFrame(to_sample_df['zip_cols'].tolist(), index=to_sample_df.index)

    else:
        # the case of HHID are just 1 value
        n_before_explode = len(to_sample_df)
        to_sample_df = to_sample_df.explode(TEMP_ID)
        assert len(to_sample_df) == n_before_explode, "The explode is expected to be same length"

    to_sample_df = to_sample_df.drop(columns=[MAP_IDS_COL, MAP_COUNTS_COL, SYN_COUNT_COL, 'zip_cols'], errors='ignore')
    # merge with known conds to get the final sampled df
    final_sampled_df = to_sample_df.merge(sample_cond, on=TEMP_ID, how="inner")
    assert len(final_sampled_df) == len(to_sample_df), "Sampled df must be same length as original df"
    return final_sampled_df


def direct_sample_from_conditional(conditionals: pd.DataFrame, evidences: pd.DataFrame, can_sample_all: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process conditionals by having it evidence as index and process the remainings"""
    possible_to_sample, impossible_to_sample, sample_cond = get_potentials_to_sample(conditionals, evidences, can_sample_all)

    to_sample_df = sample_and_choose_temp_ids(possible_to_sample)
    final_sampled = merge_chosen_temp_ids_with_known_cond(to_sample_df, sample_cond, agg_ids=True)
    
    assert len(final_sampled) == len(evidences), "Sampled df must be same length as original df"
    return final_sampled, possible_to_sample, impossible_to_sample, sample_cond


def handle_wrong_results_and_update_possible_df(wrong_results: pd.DataFrame, possible_to_sample_df: pd.DataFrame, sample_cond: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Resample based on the wrong results and remove the just sampled one as well"""
    evidence_cols = [x for x in possible_to_sample_df.columns if x not in (HHID, TEMP_ID, SYN_COUNT_COL, MAP_IDS_COL, MAP_COUNTS_COL)]
    # by the evidence cols we can get new hhid, counts
    wrong_combs_w_ids = wrong_results.groupby(evidence_cols)[HHID].apply(list)
    sub_possibles = possible_to_sample_df.set_index(evidence_cols).loc[wrong_combs_w_ids.index]
    sub_possibles[HHID] = sub_possibles.index.map(wrong_combs_w_ids) # update to smaller hh to hanldle
    sub_possibles = sub_possibles.drop(columns=[SYN_COUNT_COL]) # just not important yet, drop to avoid confusion
    map_hhid_wrong_temp_id = wrong_results.set_index(HHID)[TEMP_ID]
    
    # map to get rm ids
    sub_possibles = sub_possibles.explode(HHID) # explode to get the temp id for each hhid
    sub_possibles["to_rm_ids"] = sub_possibles[HHID].map(map_hhid_wrong_temp_id) # map the temp id to remove
    def process_get_new_sampling(row):
        # get the temp id to remove
        to_rm_id = row["to_rm_ids"] # 1 value
        # get the possible ids to sample from
        possible_ids = row[MAP_IDS_COL]
        associated_counts = row[MAP_COUNTS_COL]
        rm_idx = possible_ids.index(to_rm_id) # get the index to remove
        # remove the id from the possible ids
        new_possible_ids = possible_ids[:rm_idx] + possible_ids[rm_idx+1:]
        # do the same for counts
        new_associated_counts = associated_counts[:rm_idx] + associated_counts[rm_idx+1:]
        if len(new_possible_ids) == 0:
            return None, None
        return new_possible_ids, new_associated_counts
    
    sub_possibles[[MAP_IDS_COL, MAP_COUNTS_COL]] = sub_possibles.apply(process_get_new_sampling, axis=1, result_type="expand")
    sub_possibles = sub_possibles.drop(columns=["to_rm_ids"])

    cannot_sample_anymore = sub_possibles[sub_possibles[MAP_IDS_COL].isna()]

    sub_possibles = sub_possibles[sub_possibles[MAP_IDS_COL].notna()]
    sub_possibles[SYN_COUNT_COL] = 1
    # now we can sample from the new possible df
    updated_chosen_temp_ids = sample_and_choose_temp_ids(sub_possibles)

    return updated_chosen_temp_ids.reset_index(), cannot_sample_anymore


def determine_n_rela_for_each_hh(hh_df: pd.DataFrame, hhsz: str, n_rela_conditional: pd.DataFrame) -> Dict[str, int]:
    # we can just build a BN or a direct sample from the hh_df
    hh_df_rename = hh_df.rename(columns={col: f"{HH_TAG}_{col}" for col in hh_df.columns if col != HHID})
    final_sampled_df, possible_to_sample, impossible_to_sample, sample_cond = direct_sample_from_conditional(n_rela_conditional, hh_df_rename, can_sample_all=True)
    assert len(impossible_to_sample) == 0, "Impossible to sample df must be empty"

    # check with hhsz
    n_rela_cols = [f"n_{rela}" for rela in EXPECTED_RELATIONSHIPS]
    final_results = []
    while True:
        final_sampled_df["synthesized_hhsz"] = final_sampled_df[n_rela_cols].sum(axis=1)
        def check_match_hhsz(row):
            expected_hhsz = row[f"{HH_TAG}_{hhsz}"]
            synthesized_hhsz = row["synthesized_hhsz"]
            if expected_hhsz == "8+":
                return synthesized_hhsz >= 8
            else:
                return synthesized_hhsz == int(expected_hhsz)
        final_sampled_df["correct_hhsz"] = final_sampled_df.apply(check_match_hhsz, axis=1)
        correct_results = final_sampled_df[final_sampled_df["correct_hhsz"]]
        final_results.append(correct_results)
        wrong_results = final_sampled_df[~final_sampled_df["correct_hhsz"]]
        n_wrong = len(wrong_results)
        if n_wrong > 0:
            updated_possible_samples, cannot_sample_cases = handle_wrong_results_and_update_possible_df(wrong_results, possible_to_sample, sample_cond)
            assert len(cannot_sample_cases) == 0, "Cannot sample cases must be empty" # this is the current case for hhsz confirmation
            new_sampled = merge_chosen_temp_ids_with_known_cond(updated_possible_samples, sample_cond, agg_ids=False)
            final_sampled_df = new_sampled
            possible_to_sample = updated_possible_samples
        else:
            break

    return pd.concat(final_results, ignore_index=True).drop(columns=["synthesized_hhsz", "correct_hhsz"])


def csp_sample_by_hh(hh_df: pd.DataFrame, final_conditonals: Dict[str, pd.DataFrame], hhsz:str, relationship:str) -> pd.DataFrame:
    # process each conditionals to have temp id
    for key in final_conditonals.keys():
        final_conditonals[key] = final_conditonals[key].reset_index(drop=True) # ensure we have a clean index
        final_conditonals[key][TEMP_ID] = final_conditonals[key].index + 1
    # add hhid
    if HHID not in hh_df.columns:
        hh_df[HHID] = hh_df.reset_index(drop=True).index + 1
    processed_hh_df = determine_n_rela_for_each_hh(hh_df, hhsz, final_conditonals[f"{HH_TAG}-counts"])
    assert processed_hh_df[HHID].nunique() == len(hh_df), "Processed hh df must have same hhid as original hh df"

    # Start the sampling process
    # for relationships in RELA_BY_LEVELS:
    #     # Doing the relationship in this level
    #     for rela in relationships:
    #         # handle all or handle at each?  maybe at each, then we should return at each step
    #         direct_sample_from_conditional


    return None