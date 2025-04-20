"""Create the pool pairs from given hh_df and pp_df"""


import pandas as pd
from typing import List, Tuple, Dict


MAIN_PERSON = "Main"
HH_TAG = "HH"
EXPECTED_RELATIONSHIPS = [
    "Spouse",
    "Child",
    "Parent",
    "Grandparent",
    "Grandchild",
    "Sibling",
    "Others",
] + [MAIN_PERSON]
EPXECTED_CONNECTIONS = [
    (HH_TAG, MAIN_PERSON),
    (MAIN_PERSON, "Spouse"),
    (MAIN_PERSON, "Child"),
    (MAIN_PERSON, "Parent"),
    (MAIN_PERSON, "Grandparent"),
    (MAIN_PERSON, "Grandchild"),
    (MAIN_PERSON, "Sibling"),
    (MAIN_PERSON, "Others"),
    ("Child", "Grandchild"),
    ("Parent", "Grandparent"),
]


def count_n_states_for_each_hh(pp_df: pd.DataFrame, rel_col: str, id_col: str) -> pd.DataFrame:
    """Count the number of states for each hh in the pp_df"""
    # Count the number of states for each hh in the pp_df
    # pp_count = pp_df.groupby(id_col)[col].apply(lambda x: [x.value_counts().get(rela, 0) for rela in EXPECTED_RELATIONSHIPS])
    pp_gb = pp_df.groupby(id_col)[rel_col].apply(lambda x: list(x))
    pp_gb_df = pp_gb.apply(lambda x: [x.count(rela) for rela in EXPECTED_RELATIONSHIPS]).reset_index()
    expanded_relationships = pd.DataFrame(pp_gb_df[rel_col].tolist(), columns=EXPECTED_RELATIONSHIPS)
    expanded_relationships = expanded_relationships.rename(
        columns={rela: f"n_{rela}" for rela in EXPECTED_RELATIONSHIPS})
    return pp_gb_df.drop(columns=rel_col).join(expanded_relationships)


def pair_by_id(
    df1: pd.DataFrame, df2: pd.DataFrame, id1: str, id2: str
) -> pd.DataFrame:
    # join by the id col with inner (so only matched one got accepted)
    # Likely id will not be unique for df2 as the normal rela may have multiple in 1 hh (e.g. 2 children)
    join_result = df1.merge(
        df2, left_on=id1, right_on=id2, how="inner"
    )
    return join_result


def segment_by_col(df: pd.DataFrame, segment_col: str, rename: bool=True) -> Dict[str, pd.DataFrame]:
    result_seg_pp = {}
    for state in df[segment_col].unique():
        sub_df = df[df[segment_col] == state].copy()
        if rename:
            sub_df.rename(columns={col: f"{state}_{col}" for col in sub_df.columns}, inplace=True)
        result_seg_pp[state] = sub_df.drop(columns=[f"{state}_{segment_col}"])
    return result_seg_pp


def create_pool_pairs(hh: pd.DataFrame, pp: pd.DataFrame, hhid: str, relationship: str) -> Dict[str, pd.DataFrame]:
    assert set(pp[relationship]) == set(EXPECTED_RELATIONSHIPS), "Invalid relationship in pp"
    segmented_pp_by_rela = segment_by_col(pp, relationship)
    assert len(segmented_pp_by_rela[MAIN_PERSON]) == len(hh) # ensure each hh has 1 main person
    hh = hh.rename(columns={col: f"{HH_TAG}_{col}" for col in hh.columns})

    paired_connections = {}
    for src_rela, dst_rela in EPXECTED_CONNECTIONS:
        src_df = segmented_pp_by_rela.get(src_rela, hh)
        dst_df = segmented_pp_by_rela.get(dst_rela, hh)
        paired_connections[f"{src_rela}-{dst_rela}"] = pair_by_id(
            src_df, dst_df, f"{src_rela}_{hhid}", f"{dst_rela}_{hhid}"
        ).drop(columns=[f"{src_rela}_{hhid}", f"{dst_rela}_{hhid}"])

    # special case for counting rela
    rel_counts_each_hhid = count_n_states_for_each_hh(pp, relationship, hhid)
    paired_connections[f"{HH_TAG}-counts"] = hh.merge(
        rel_counts_each_hhid, left_on=f"{HH_TAG}_{hhid}", right_on=hhid, how="inner"
    ).drop(columns=[f"{HH_TAG}_{hhid}", hhid])

    return paired_connections
    