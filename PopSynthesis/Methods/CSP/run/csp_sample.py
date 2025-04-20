"""CSP core: sample by each relationship and ensure the household size matching"""

import pandas as pd
from typing import Dict, List, Tuple
from PopSynthesis.Methods.CSP.run.create_pool_pairs import HH_TAG, MAIN_PERSON, COUNT_COL, EXPECTED_RELATIONSHIPS, EPXECTED_CONNECTIONS


TEMP_ID = "temp_id"


def split_and_process_by_temp_id(df: pd.DataFrame, evidence_cols: List[str], temp_id: str) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Split the df by temp_id and return the evidence and sample df"""
    assert temp_id in df.columns, f"{temp_id} must be in df columns"
    evidences_cond = df[evidence_cols + [temp_id, COUNT_COL]].copy()
    sample_cond = df[[x for x in df.columns if x not in (evidence_cols + [COUNT_COL])]].copy()
    evidences_to_ids = evidences_cond.groupby(evidence_cols)[temp_id].apply(list)
    evidences_to_counts = evidences_cond.groupby(evidence_cols)[COUNT_COL].apply(list)
    return evidences_to_ids, evidences_to_counts, sample_cond


def process_conditionals_to_sample(conditionals: pd.DataFrame, evidences: pd.DataFrame) -> pd.DataFrame:
    """Process conditionals by having it evidence as index and process the remainings"""
    evidence_cols = evidences.columns.tolist()
    conditionals = conditionals.reset_index(drop=True) # ensure we have a clean index
    conditionals[TEMP_ID] = conditionals.index # add a temp id to sample
    evidences_to_ids, evidences_to_counts, sample_cond = split_and_process_by_temp_id(conditionals, evidence_cols, TEMP_ID)

    # count the evidences occurences
    evidences_counts = evidences.value_counts()
    # We do not count for the case of mismatch between evidences and conditionals
    # print(evidences_counts.loc[('2', 'Missing', '8000+', '4+', 'Being Purchased')])
    # print(evidences_to_ids.loc[('2', 'Missing', '8000+', '4+', 'Being Purchased')])
    # print(set(evidences_counts.index) - set(evidences_to_ids.index))
    assert set(evidences_counts.index) <= set(evidences_to_ids.index), "Given evidences must be subset of condtionals ids" 
    assert set(evidences_counts.index) <= set(evidences_to_counts.index), "Counts must be subset of condtionals counts"

    # evidences["potential_ids"] = evidences.set_index(evidence_cols).index.map(evidences_to_ids)
    # evidences["asscociated_counts"] = evidences.set_index(evidence_cols).index.map(evidences_to_counts)
    print(evidences_counts)


def direct_sample_from_conditional(conditionals: pd.DataFrame, evidences: pd.DataFrame) -> pd.DataFrame:
    """Return the direct sample from conditional based on the given evidences"""
    assert COUNT_COL in conditionals.columns, "Count column must be in conditionals"
    assert set(evidences.columns).issubset(set(conditionals.columns)), "Evidences columns must be subset of conditionals columns"
    processed_conditionals = process_conditionals_to_sample(conditionals, evidences)
    # loop through evidence
    # each evidence we will directly sameple the remaining from the conditionals


def determine_n_rela_for_each_hh(hh_df: pd.DataFrame, hhsz: str, n_rela_conditional: pd.DataFrame) -> Dict[str, int]:
    # we can just build a BN or a direct sample from the hh_df
    hh_df_rename = hh_df.rename(columns={col: f"{HH_TAG}_{col}" for col in hh_df.columns})
    return direct_sample_from_conditional(n_rela_conditional, hh_df_rename)


def csp_sample_by_hh(hh_df: pd.DataFrame, final_conditonals: Dict[str, pd.DataFrame], hhsz:str, relationship:str) -> pd.DataFrame:
    processed_hh_df = determine_n_rela_for_each_hh(hh_df, hhsz, final_conditonals[f"{HH_TAG}-counts"])
    return None