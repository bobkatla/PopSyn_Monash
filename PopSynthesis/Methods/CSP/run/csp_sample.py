"""CSP core: sample by each relationship and ensure the household size matching"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from PopSynthesis.Methods.CSP.run.create_pool_pairs import (
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
MAP_IDS_COL = "potential_ids"
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

    conditionals = conditionals.reset_index(drop=True) # ensure we have a clean index
    conditionals[TEMP_ID] = conditionals.index # add a temp id to sample
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



def direct_sample_from_conditional(conditionals: pd.DataFrame, evidences: pd.DataFrame, can_sample_all: bool) -> pd.DataFrame:
    """Process conditionals by having it evidence as index and process the remainings"""
    possible_to_sample_df, impossible_to_sample_df, sample_cond = get_potentials_to_sample(conditionals, evidences, can_sample_all)
    
    def sample_by_row(row):
        ids = row[MAP_IDS_COL]
        counts = row[MAP_COUNTS_COL]
        probs = [count / sum(counts) for count in counts]
        n = row[SYN_COUNT_COL]
        return np.random.choice(ids, size=n, p=probs, replace=True)
    
    # Apply to get the sampled ids
    to_sample_df = possible_to_sample_df.copy()
    to_sample_df[TEMP_ID] = to_sample_df.apply(sample_by_row, axis=1)

    # explode both hhid and chose at same time to get the results
    to_sample_df['zip_cols'] = to_sample_df.apply(lambda row: list(zip(row[HHID], row[TEMP_ID])), axis=1)
    to_sample_df = to_sample_df.explode('zip_cols')
    to_sample_df[[HHID, TEMP_ID]] = pd.DataFrame(to_sample_df['zip_cols'].tolist(), index=to_sample_df.index)
    to_sample_df = to_sample_df.drop(columns=[MAP_IDS_COL, MAP_COUNTS_COL, SYN_COUNT_COL, 'zip_cols'])

    # merge with known conds to get the final sampled df
    final_sampled_df = to_sample_df.merge(sample_cond, on=TEMP_ID, how="inner").drop(columns=[TEMP_ID])
    assert len(final_sampled_df) == len(evidences), "Sampled df must be same length as original df"

    return final_sampled_df


def determine_n_rela_for_each_hh(hh_df: pd.DataFrame, hhsz: str, n_rela_conditional: pd.DataFrame) -> Dict[str, int]:
    # we can just build a BN or a direct sample from the hh_df
    hh_df_rename = hh_df.rename(columns={col: f"{HH_TAG}_{col}" for col in hh_df.columns if col != HHID})
    return direct_sample_from_conditional(n_rela_conditional, hh_df_rename, can_sample_all=True)


def csp_sample_by_hh(hh_df: pd.DataFrame, final_conditonals: Dict[str, pd.DataFrame], hhsz:str, relationship:str) -> pd.DataFrame:
    # add hhid
    if HHID not in hh_df.columns:
        hh_df[HHID] = hh_df.reset_index(drop=True).index + 1
    processed_hh_df = determine_n_rela_for_each_hh(hh_df, hhsz, final_conditonals[f"{HH_TAG}-counts"])
    assert processed_hh_df[HHID].nunique() == len(hh_df), "Processed hh df must have same hhid as original hh df"

    return None