"""Sampling from pairs with a given df"""

import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
# NOTE: make sure it can synthesize something for each rela


COUNT_COL = "rela_count"  # created col for sampling
SUM_COUNT_COL = "sum_count"  # for condensed syn
ID_COUNT_COL = "id_n_count"  # for condensed syn
TO_SAMPLE_COL = "to_sample"  # for condensed pool


def create_count_col(syn_with_count: pd.DataFrame, cou_col: str) -> pd.DataFrame:
    """Update the syn with the rela_count based on the seed"""
    syn_with_count = syn_with_count.copy(deep=True)
    syn_with_count.loc[:, COUNT_COL] = list(syn_with_count[cou_col])
    return syn_with_count


def update_by_rm_for_all_pools(
    rm_df: pd.DataFrame, pools: Dict[str, pd.DataFrame], check_cols: List[str]
) -> Dict[str, pd.DataFrame]:
    results = {}
    for names, pool in pools.items():
        results[names] = update_by_rm_for_pool(rm_df, pool, check_cols)
    return results


def update_by_rm_for_pool(
    rm_df: pd.DataFrame, target_pool: pd.DataFrame, check_cols: List[str]
) -> pd.DataFrame:
    assert set(check_cols) <= set(rm_df.columns)
    assert set(check_cols) <= set(target_pool.columns)
    converted_rm_df = rm_df.set_index(check_cols)
    converted_target_pool = target_pool.set_index(check_cols)
    rm_comb = set(converted_rm_df.index)
    targeted_comb = set(converted_target_pool.index)
    updated_target_comb = targeted_comb - rm_comb
    filered_pool = converted_target_pool.loc[
        converted_target_pool.index.isin(updated_target_comb)
    ]
    return filered_pool.reset_index()


def convert_condensed_syn_back(
    condensed_syn: pd.DataFrame, syn_id: str
) -> pd.DataFrame:
    condensed_syn = condensed_syn.drop(columns=[SUM_COUNT_COL])
    # 1 col only now: ID_COUNT_COLc
    condensed_syn = condensed_syn.explode(ID_COUNT_COL)
    condensed_syn[syn_id] = condensed_syn[ID_COUNT_COL].apply(lambda x: x[0])
    return condensed_syn.reset_index().drop(columns=[ID_COUNT_COL])


def condense_evidence_syn(
    given_syn: pd.DataFrame, syn_id: str, evidence_cols: List[str]
) -> pd.DataFrame:
    """condense the given df to have comb as """
    assert COUNT_COL in given_syn.columns
    given_syn[COUNT_COL] = given_syn[COUNT_COL].astype(int)
    check_syn = given_syn[evidence_cols + [COUNT_COL, syn_id]]
    syn_gb_ids = (
        check_syn.groupby(evidence_cols)[[syn_id, COUNT_COL]]
        .apply(lambda x: list(x.to_numpy()))
        .rename(ID_COUNT_COL)
    )
    syn_gb_sum = check_syn.groupby(evidence_cols)[COUNT_COL].sum().rename(SUM_COUNT_COL)
    return pd.concat([syn_gb_ids, syn_gb_sum], axis=1)


def decoupling_paired_pool(
    paired_pool: pd.DataFrame, evidence_cols: List[str], sample_cols: List[str]
) -> pd.DataFrame:
    """Output special df that seperate the evidence and to sample as index and values"""
    target_pool = paired_pool.copy(deep=True).reset_index(drop=True)
    # Note the order is important here
    target_pool.loc[:, TO_SAMPLE_COL] = pd.Series(
        list(target_pool[sample_cols].to_numpy())
    )
    assert not target_pool[TO_SAMPLE_COL].isna().any()
    pool_gb_evidences = target_pool.groupby(evidence_cols)[TO_SAMPLE_COL].apply(
        lambda x: np.array(x)
    )
    return pd.DataFrame(pool_gb_evidences)  # 1 col only


def sample_matching_from_pairs(
    given_syn: pd.DataFrame,
    syn_id: str,
    paired_pool: pd.DataFrame,
    evidence_cols: List[str],
    sample_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Given pools and evidences to get the matching and to remove HH/Main and to kept"""
    # NOTE: make sure the id is correct, including the removed
    assert set(evidence_cols) <= set(paired_pool.columns)
    assert set(sample_cols) <= set(paired_pool.columns)
    assert set(evidence_cols) <= set(given_syn.columns)
    assert not given_syn[syn_id].duplicated().any()
    check_syn = given_syn.copy(deep=True)
    check_pool = paired_pool.copy(deep=True)
    if COUNT_COL not in given_syn.columns:
        # the case use directly to have 1 for each only
        check_syn[COUNT_COL] = 1

    # Condense both syn and pool, they will now have similar indexes
    condensed_syn = condense_evidence_syn(check_syn, syn_id, evidence_cols)
    condensed_pool = decoupling_paired_pool(check_pool, evidence_cols, sample_cols)

    # NOTE: the sample_cols order is for pool
    # NOTE: the order for syn is, [id, count]
    comb_in_syn = set(condensed_syn.index)
    comb_in_pool = set(condensed_pool.index)
    to_rm_comb_syn = comb_in_syn - comb_in_pool
    common_comb = comb_in_syn & comb_in_pool

    rm_condensed = condensed_syn.loc[condensed_syn.index.isin(to_rm_comb_syn)]
    rm_syn_rec = convert_condensed_syn_back(rm_condensed, syn_id)

    if len(common_comb) == 0:
        # Means no matching comb for syn and pool
        assert len(rm_syn_rec) == len(check_syn)
        return pd.DataFrame(), rm_syn_rec, pd.DataFrame()

    remained_condensed_syn = condensed_syn.loc[condensed_syn.index.isin(common_comb)]
    kept_syn_rec = convert_condensed_syn_back(remained_condensed_syn, syn_id)

    filtered_condensed_pool = condensed_pool.loc[condensed_pool.index.isin(common_comb)]
    assert len(remained_condensed_syn) == len(filtered_condensed_pool)
    combined_condense = pd.merge(
        remained_condensed_syn,
        filtered_condensed_pool,
        left_index=True,
        right_index=True,
    )

    # func to sample for each case
    def sample_rec(r):
        results = []
        possible_recs = r[TO_SAMPLE_COL]
        segment_by_id = r[ID_COUNT_COL]
        tot_samples = r[SUM_COUNT_COL]
        chosen_recs = np.random.choice(possible_recs, tot_samples)
        start = 0
        for sid, val in segment_by_id:
            rec_details = np.array(
                [list(x) + [sid] for x in chosen_recs[start : start + val]]
            )
            results.append(rec_details)
            start += val
        assert start == tot_samples
        return results

    sample_result_col = "sample_results"
    combined_condense[sample_result_col] = combined_condense.apply(sample_rec, axis=1)
    combined_condense = combined_condense.drop(
        columns=[TO_SAMPLE_COL, ID_COUNT_COL, SUM_COUNT_COL]
    )
    combined_condense_exploded = combined_condense.explode(sample_result_col)

    combined_sampled_rec = np.vstack(
        tuple(combined_condense_exploded[sample_result_col])
    )
    fin_samples = pd.DataFrame(combined_sampled_rec, columns=sample_cols + [syn_id])

    return fin_samples, rm_syn_rec, kept_syn_rec
