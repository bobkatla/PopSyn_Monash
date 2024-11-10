"""
Now for each zone we will do adjustment

For a given zone (a vector), we can identify which att is pos and which is neg so we can replace
Inputs:
curr_syn - curr_syn of that zone (matrix) with counts
diff_census - diff from census with neg and pos (vector)
pool - data of all possible values (matrix) with counts
adjusted_atts - adjusted atts (list) so we need to maintain those combinations

Outputs: New syn_pop for that zone

We need to think about how to better this process, not with pairing and simple finding, maybe matrix opt
"""
import pandas as pd
import numpy as np
from PopSynthesis.Methods.IPSF.const import zone_field
from PopSynthesis.Methods.IPSF.utils.condensed_tools import (
    CondensedDF,
    filter_by_SAA_adjusted,
)
from typing import List, Tuple
import itertools
import random


def convert_condensed_by_adjusted_atts(
    condensed: CondensedDF, adjusted_atts: List[str]
) -> pd.DataFrame:
    condensed_records = condensed.get_condensed()
    gb_ids = condensed_records.groupby(adjusted_atts)[condensed.id_col].apply(
        lambda x: list(x)
    )
    gb_counts = condensed_records.groupby(adjusted_atts)[condensed.count_col].sum()
    merged_new_condensed = pd.concat([gb_ids, gb_counts], axis=1)
    merged_new_condensed[condensed.id_col] = merged_new_condensed[
        condensed.id_col
    ].apply(lambda x: sum(x, []))
    return merged_new_condensed


def sample_syn_and_pool_adjust(
    condensed_syn: CondensedDF,
    condensed_pool: CondensedDF,
    adjusted_atts: List[str],
    n_adjust: int,
) -> Tuple[List[int], List[int], int]:
    converted_syn = convert_condensed_by_adjusted_atts(
        condensed_syn, adjusted_atts
    ).add_prefix("syn_")
    converted_pool = convert_condensed_by_adjusted_atts(
        condensed_pool, adjusted_atts
    ).add_prefix("pool_")
    combined_syn_pool = pd.merge(
        converted_syn, converted_pool, left_index=True, right_index=True
    )
    # sample using the syn
    chosen_col_to_sample = f"syn_{condensed_syn.count_col}"
    sub_to_sample = combined_syn_pool[chosen_col_to_sample].reset_index()
    # sample for the comb of prev
    sample_results = sub_to_sample.sample(
        n=n_adjust, weights=chosen_col_to_sample, replace=True
    )
    check_sample_value = sample_results.groupby(adjusted_atts).count()
    check_sample_value = check_sample_value.rename(
        columns={chosen_col_to_sample: "sample_count"}
    )
    updated_combined_syn_pool = pd.concat(
        [combined_syn_pool, check_sample_value], axis=1
    ).fillna(0)
    # Process to sample for each case of prev comb
    updated_combined_syn_pool["decided_sample"] = updated_combined_syn_pool.apply(
        lambda r: min(
            r[f"syn_{condensed_syn.count_col}"],
            r[f"pool_{condensed_pool.count_col}"],
            r["sample_count"],
        ),
        axis=1,
    )
    updated_combined_syn_pool["Remaining_cannot_sample"] = (
        updated_combined_syn_pool["sample_count"]
        - updated_combined_syn_pool["decided_sample"]
    )
    updated_combined_syn_pool["decided_syn"] = updated_combined_syn_pool.apply(
        lambda r: list(
            np.random.choice(r["syn_ids"], size=int(r["decided_sample"]), replace=False)
        ),
        axis=1,
    )
    updated_combined_syn_pool["decided_pool"] = updated_combined_syn_pool.apply(
        lambda r: list(
            np.random.choice(
                r["pool_ids"], size=int(r["decided_sample"]), replace=False
            )
        ),
        axis=1,
    )
    # get results
    removed_ids_syn = sum(updated_combined_syn_pool["decided_syn"].to_list(), [])
    add_ids_pool = sum(updated_combined_syn_pool["decided_pool"].to_list(), [])
    total_cannot_sample = int(
        updated_combined_syn_pool["Remaining_cannot_sample"].sum()
    )

    return removed_ids_syn, add_ids_pool, total_cannot_sample


def process_neg_pos_states_adjustment():
    NotImplemented


def zone_adjustment(
    att: str,
    curr_syn: pd.DataFrame,
    diff_census: pd.Series,
    pool: pd.DataFrame,
    adjusted_atts: List[str],
) -> pd.DataFrame:
    assert "id"
    assert len(curr_syn[zone_field].unique()) == 1
    zone = curr_syn[zone_field].unique()[0]

    check_syn = curr_syn.drop(columns=[zone_field])

    neg_states = diff_census[diff_census < 0].index.tolist()
    pos_states = diff_census[diff_census > 0].index.tolist()
    zeros_states = diff_census[
        diff_census == 0
    ].index.tolist()  # Only for later processing
    pairs_adjust = list(itertools.product(neg_states, pos_states))
    random.shuffle(pairs_adjust)

    # check_got_adjusted = []
    # PARALLELLLLLLL
    # Dict of lock based on the pandas series idx
    # Segment the curr_syn (to be adjusted), by neg and pos state
    # pos state part will stay the same (store for later concat)
    # neg will also create a dict of df (as we only care about it)
    # each neg df can only be access by 1 process and it will be updated (thread safe required)
    # same for the
    # should convert all into multiprocessing.Manager().dict()
    # concat all again

    segmented_curr_syn_neg = {
        neg_state: check_syn[check_syn[att] == neg_state] for neg_state in neg_states
    }
    # This can help reduce overhead as we don't want multiple copies of the large pool
    segmented_pool_pos_state = {
        pos_state: pool[pool[att] == pos_state] for pos_state in pos_states
    }
    added_records = []

    for neg_state, pos_state in pairs_adjust:
        neg_val = int(diff_census[neg_state])
        pos_val = int(diff_census[pos_state])
        # Cannot adjust anyway
        if neg_val == 0 or pos_val == 0:
            continue

        # Check neg state
        filtered_syn_pop = segmented_curr_syn_neg[neg_state]
        num_syn_pop = len(filtered_syn_pop)  # must not change
        neg_comb_prev = filtered_syn_pop.set_index(adjusted_atts)
        condensed_pop_check = CondensedDF(filtered_syn_pop)
        # check pos state
        filtered_pool = segmented_pool_pos_state[pos_state]
        pos_comb_prev = filtered_pool.set_index(adjusted_atts)
        condensed_pool_check = CondensedDF(filtered_pool)

        n_adjust = min(abs(neg_val), pos_val)  # the possible to adjust
        possible_prev_comb = set(neg_comb_prev.index) & set(pos_comb_prev.index)
        if len(possible_prev_comb) == 0:
            # No overlapping
            continue

        condensed_pop_check, remaining_pop = filter_by_SAA_adjusted(
            condensed_pop_check, list(possible_prev_comb), adjusted_atts
        )

        to_remove_pop_ids, to_add_pool_ids, n_not_adjusted = sample_syn_and_pool_adjust(
            condensed_pop_check, condensed_pool_check, adjusted_atts, n_adjust
        )

        chosen_records_from_pool = condensed_pool_check.get_sub_records_by_ids(
            to_add_pool_ids
        )
        # update the condensed pop
        condensed_pop_check.remove_identified_ids(to_remove_pop_ids)
        added_records.append(chosen_records_from_pool)  # these will be added later
        # condensed_pop_check.add_new_records(chosen_records_from_pool)

        # final update
        final_neg_syn = pd.concat(
            [condensed_pop_check.get_full_records(), remaining_pop]
        )
        assert num_syn_pop == len(final_neg_syn) + len(chosen_records_from_pool)

        # update the syn pop segmented
        segmented_curr_syn_neg[neg_state] = final_neg_syn

        actual_got_adjusted = n_adjust - n_not_adjusted
        diff_census[neg_state] += actual_got_adjusted
        diff_census[pos_state] -= actual_got_adjusted

        # check_got_adjusted.append(actual_got_adjusted)

    # print(check_got_adjusted)
    ori_num_syn = len(check_syn)
    curr_syn_pos_state = check_syn[check_syn[att].isin(pos_states + zeros_states)]

    neg_syn_updated = [df for df in segmented_curr_syn_neg.values()]
    final_resulted_syn = pd.concat(
        [curr_syn_pos_state] + neg_syn_updated + added_records
    )
    final_resulted_syn[zone_field] = zone
    if len(final_resulted_syn) != ori_num_syn:
        raise ValueError(
            f"Error processing at zone {zone}: expected {ori_num_syn} records, got {len(final_resulted_syn)}"
        )
    return final_resulted_syn
