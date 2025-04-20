"""CSP core: sample by each relationship and ensure the household size matching"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from PopSynthesis.Methods.CSP.run.create_pool_pairs import HH_TAG, MAIN_PERSON, COUNT_COL, EXPECTED_RELATIONSHIPS, EPXECTED_CONNECTIONS


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


def direct_sample_from_conditional(conditionals: pd.DataFrame, evidences: pd.DataFrame) -> pd.DataFrame:
    """Process conditionals by having it evidence as index and process the remainings"""
    assert COUNT_COL in conditionals.columns, "Count column must be in conditionals"
    assert set(evidences.columns).issubset(set(conditionals.columns)), "Evidences columns must be subset of conditionals columns"

    evidence_cols = evidences.columns.tolist()
    conditionals = conditionals.reset_index(drop=True) # ensure we have a clean index
    conditionals[TEMP_ID] = conditionals.index # add a temp id to sample
    evidences_to_ids, evidences_to_counts, sample_cond = split_and_process_by_temp_id(conditionals, evidence_cols, TEMP_ID)

    # count the evidences occurences
    evidences_counts = evidences.value_counts()
    assert set(evidences_counts.index) <= set(evidences_to_ids.index), "Given evidences must be subset of condtionals ids" 
    assert set(evidences_counts.index) <= set(evidences_to_counts.index), "Counts must be subset of condtionals counts"
    to_sample_df = evidences_counts.reset_index(name=SYN_COUNT_COL)

    to_sample_df[MAP_IDS_COL] = evidences_counts.index.map(evidences_to_ids)
    to_sample_df[MAP_COUNTS_COL] = evidences_counts.index.map(evidences_to_counts)
    
    def sample_by_row(row):
        ids = row[MAP_IDS_COL]
        counts = row[MAP_COUNTS_COL]
        probs = [count / sum(counts) for count in counts]
        n = row[SYN_COUNT_COL]
        return np.random.choice(ids, size=n, p=probs, replace=True)
    
    to_sample_df[TEMP_ID] = to_sample_df.apply(sample_by_row, axis=1)
    print(to_sample_df)


def determine_n_rela_for_each_hh(hh_df: pd.DataFrame, hhsz: str, n_rela_conditional: pd.DataFrame) -> Dict[str, int]:
    # we can just build a BN or a direct sample from the hh_df
    hh_df_rename = hh_df.rename(columns={col: f"{HH_TAG}_{col}" for col in hh_df.columns})
    return direct_sample_from_conditional(n_rela_conditional, hh_df_rename)


def csp_sample_by_hh(hh_df: pd.DataFrame, final_conditonals: Dict[str, pd.DataFrame], hhsz:str, relationship:str) -> pd.DataFrame:
    processed_hh_df = determine_n_rela_for_each_hh(hh_df, hhsz, final_conditonals[f"{HH_TAG}-counts"])
    return None