""" Contains the funcs for SAA, help seperate out and easier debugging"""

import polars as pl
import pandas as pd
import numpy as np

from typing import List, Union
from PopSynthesis.Methods.IPSF.const import count_field, zone_field


def sample_from_pl(df: pl.DataFrame, n: int, count_field:str = count_field, with_replacement=True) -> pl.DataFrame:
    # Normalize weights to sum to 1
    weights = df[count_field].to_numpy()
    weights = weights/weights.sum()
    sample_indices = np.random.choice(df.height, size=n, replace=with_replacement, p=weights)
    return df[sample_indices.tolist()]


def init_syn_pop_saa(att:str, marginal_data: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    assert zone_field in marginal_data
    states = list(pool[att].unique())
    assert set(states + [zone_field]) == set(marginal_data.columns)

    if count_field not in pool.columns:
        pool = pool.with_columns([pl.lit(1).alias(count_field)])

    sub_pops = []
    for state in states:
        sub_pool = pool.filter(pl.col(att) == state)
        if len(sub_pool) == 0:
            print(
                f"WARNING: cannot see {att}_{state} in the pool, sample by the rest"
            )
            sub_pool = pool  # if there are none, we take all
        for zone in marginal_data[zone_field]:
            condition = marginal_data.filter(pl.col(zone_field) == zone)
            census_val = condition.select(state).to_numpy()[0,0]

            sub_syn_pop = sample_from_pl(sub_pool, census_val)

            sub_syn_pop = sub_syn_pop.with_columns([pl.lit(zone).alias(zone_field)])
            sub_syn_pop = sub_syn_pop.drop(count_field)

            sub_pops.append(sub_syn_pop)
    return pl.concat(sub_pops)


def adjust_atts_state_match_census(att: str, curr_syn_pop: Union[None, pd.DataFrame], census_data_by_att: pd.DataFrame, adjusted_atts: List[str], pool: pd.DataFrame) -> pd.DataFrame:
    if curr_syn_pop is None:
        curr_syn_pop = init_syn_pop_saa(att, census_data_by_att, pool).to_pandas()
    else:
        # Will slowly convert to polars later
        # All the pool and synpop would be in the count format (with weights)
        # This also help confirmed later if we want to use the vista directly
        # now we need update popsyn with SAA, we can k
        # Calulate the diff for each state comparing with census
        # This can just be a value counts and minus
        # Then adjust by add and del
        # We need to search for combinations with each add and del
        # This is to ensure the prev adjust atts are maintained
        # We only can del what can be add and add what can be del
        # Value counts for all states to get the filter
        # Remember to check the weights, the weights for each combination would be the sum
        # Maybe achieve this via a set intersection to filter feasible cases
        # For add, we only care whether they exist or not
        # For del, we only can del what we can del
        # Maybe instead of pairing neg and pos (maybe that while it is slow)
        # We draw the add from a pool, as long as we update the state diff
        # But how to make sure we don't over sample (as we will sample with replacement)
        # So we may need a dict to map to the existing
        # We also need to care about the sampling

        # Loop through the matching combination to add and del
        NotImplemented