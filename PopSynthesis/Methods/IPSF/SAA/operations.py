""" Contains the funcs for SAA, help seperate out and easier debugging"""

import polars as pl

from typing import List, Union


def init_syn_pop_saa(att:str, marginal_data: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    assert "zone_id" in marginal_data
    assert set(pool[att].unique()) == set(marginal_data.columns)
    for state in marginal_data.columns:
        sub_pool = pool.filter(pl.col(att) == state)
        if len(sub_pool) == 0:
            print(
                f"WARNING: cannot see {att}_{state} in the pool, sample by the rest"
            )
            sub_pool = pool  # if there are none, we take all
        tot_state = marginal_data[state]



def adjust_atts_state_match_census(att: str, curr_syn_pop: Union[None, pl.DataFrame], census_data: pl.DataFrame, adjusted_atts: List[str], pool: pl.DataFrame) -> pl.DataFrame:
    if curr_syn_pop is None:
        curr_syn_pop = init_syn_pop_saa()