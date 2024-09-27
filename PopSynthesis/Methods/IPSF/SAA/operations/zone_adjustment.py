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
from PopSynthesis.Methods.IPSF.const import count_field, zone_field
from typing import List
import itertools
import random


def find_combinations_with_prev_atts(filtered_targeted_samples: pd.DataFrame, prev_atts: List[str]) -> pd.DataFrame:
    sub_samples = filtered_targeted_samples[prev_atts + [count_field]]
    gb_cols = sub_samples.columns.difference([count_field]).to_list()
    sub_samples = sub_samples.reset_index()
    gb_df = pd.DataFrame(sub_samples.groupby(gb_cols)["index"].apply(lambda x: list(x)))
    gb_df[count_field] = gb_df["index"].apply(lambda x: len(x))
    # get the list of index
    return gb_df


def add_decided_sample_count(count_df: pd.DataFrame, n_adjust: int, use_pool_weights:bool=True):
    weights_col = "pool_count" if use_pool_weights else "syn_count"
    idx_states = count_df.index.to_list()
    probs_for_sample = count_df[weights_col].to_list() / count_df[weights_col].sum()
    sample_results = list(np.random.choice(idx_states, n_adjust, p=probs_for_sample))

    name_att = count_df.index.name
    count_df = count_df.reset_index()
    count_df["raw_sample_count"] = count_df[name_att].apply(lambda x: sample_results.count(x))
    count_df["decided_sample_count"] = count_df.apply(lambda x: min(x["pool_count"], x["syn_count"], x["raw_sample_count"]), axis=1)
    return count_df.set_index(name_att)


def zone_adjustment(att: str, curr_syn_count: pd.DataFrame, diff_census: pd.Series, pool: pd.DataFrame, adjusted_atts: List[str]) -> pd.DataFrame:
    assert count_field in curr_syn_count.columns
    assert count_field in pool.columns
    assert len(curr_syn_count[zone_field].unique()) == 1
    curr_syn_count = curr_syn_count.drop(columns=[zone_field])

    neg_states = diff_census[diff_census < 0].index.tolist()
    pos_states = diff_census[diff_census > 0].index.tolist()
    pairs_adjust = list(itertools.product(neg_states, pos_states))
    random.shuffle(pairs_adjust)
    
    for neg_state, pos_state in pairs_adjust:
        # Check neg state
        filtered_syn_pop = curr_syn_count[curr_syn_count[att] == neg_state]
        count_neg_in_syn = find_combinations_with_prev_atts(filtered_syn_pop, adjusted_atts)
        neg_val = int(diff_census[neg_state])
        # check pos state
        filtered_pool = pool[pool[att] == pos_state]
        count_pos_in_pool = find_combinations_with_prev_atts(filtered_pool, adjusted_atts)
        pos_val = int(diff_census[pos_state])

        n_adjust = min(abs(neg_val), pos_val) # the possible to adjust
        possible_prev_comb = set(count_neg_in_syn.index) & set(count_pos_in_pool.index)
        if len(possible_prev_comb) == 0:
            # No overlapping 
            continue
        filtered_count_neg = count_neg_in_syn.loc[list(possible_prev_comb)].add_prefix("syn_")
        filtered_count_pos = count_pos_in_pool.loc[list(possible_prev_comb)].add_prefix("pool_")
        combined_filter = pd.concat([filtered_count_neg, filtered_count_pos], axis=1)
        # Sampling from this, will based on the neg from syn
        combined_counts = add_decided_sample_count(combined_filter, n_adjust)
        combined_counts["syn_chosen_idx"] = combined_counts.apply(lambda r: np.random.choice(r["syn_index"], r["decided_sample_count"], replace=False), axis=1)
        combined_counts["pool_chosen_idx"] = combined_counts.apply(lambda r: np.random.choice(r["pool_index"], r["decided_sample_count"], replace=True), axis=1)
        syn_idx = [item for sublist in combined_counts["syn_chosen_idx"] for item in sublist]
        pool_idx = [item for sublist in combined_counts["pool_chosen_idx"] for item in sublist]
        pool_chosen_samples = filtered_pool.loc[pool_idx]
        print(pool_chosen_samples)
        # Now we need to remove the neg and add the pos, this requires sampling
        break