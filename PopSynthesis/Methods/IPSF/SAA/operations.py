""" Contains the funcs for SAA, help seperate out and easier debugging"""

import polars as pl
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


def adjust_atts_state_match_census(att: str, curr_syn_pop: Union[None, pl.DataFrame], census_data: pl.DataFrame, adjusted_atts: List[str], pool: pl.DataFrame) -> pl.DataFrame:
    if curr_syn_pop is None:
        curr_syn_pop = init_syn_pop_saa()