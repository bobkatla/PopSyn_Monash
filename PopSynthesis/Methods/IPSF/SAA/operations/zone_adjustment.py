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


def find_combinations_with_prev_atts(filtered_targeted_samples: pd.DataFrame, prev_atts: List[str]):
    sub_samples = filtered_targeted_samples[prev_atts + [count_field]]
    count_samples = sub_samples.groupby(sub_samples.columns.difference([count_field]).to_list(), as_index=False)[count_field].sum()
    return count_samples.set_index(count_samples.columns.difference([count_field]).to_list())

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
        filtered_count_neg = count_neg_in_syn.loc[list(possible_prev_comb)]
        filtered_count_pos = count_pos_in_pool.loc[list(possible_prev_comb)]
        
        
        break