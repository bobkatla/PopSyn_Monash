""" Contains the funcs for SAA, help seperate out and easier debugging"""

import polars as pl
import pandas as pd
import numpy as np

from typing import List, Union, Tuple, Dict
from PopSynthesis.Methods.IPSF.const import count_field, zone_field
from PopSynthesis.Methods.IPSF.SAA.operations.compare_census import (
    calculate_states_diff,
)
from PopSynthesis.Methods.IPSF.SAA.operations.zone_adjustment import zone_adjustment
import multiprocessing as mp


def process_raw_ipu_init(
    marg: pd.DataFrame, seed: pd.DataFrame
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    atts = [x for x in seed.columns if x not in ["serialno", "sample_geog"]]
    segmented_marg = {}
    zones = marg[marg.columns[marg.columns.get_level_values(0) == zone_field]].values
    zones = [z[0] for z in zones]
    for att in atts:
        sub_marg = marg[marg.columns[marg.columns.get_level_values(0) == att]]
        if sub_marg.empty:
            print(f"Don't have this att {att} in census")
            continue
        sub_marg.columns = sub_marg.columns.droplevel(0)
        sub_marg.loc[:, [zone_field]] = zones
        sub_marg = sub_marg.set_index(zone_field)
        segmented_marg[att] = sub_marg
    new_seed = seed.drop(columns=["sample_geog", "serialno"], errors="ignore")
    return segmented_marg, new_seed


def sample_from_pl(
    df: pl.DataFrame, n: int, count_field: str = count_field, with_replacement=True
) -> pl.DataFrame:
    # Normalize weights to sum to 1
    weights = df[count_field].to_numpy()
    weights = weights / weights.sum()
    sample_indices = np.random.choice(
        df.height, size=n, replace=with_replacement, p=weights
    )
    return df[sample_indices.tolist()]


def init_syn_pop_saa(
    att: str, marginal_data: pd.DataFrame, pool: pd.DataFrame
) -> pl.DataFrame:
    pool = pl.from_pandas(pool)
    marginal_data = pl.from_pandas(marginal_data)
    assert zone_field in marginal_data
    states = list(pool[att].unique())
    assert set(states + [zone_field]) == set(marginal_data.columns)

    if count_field not in pool.columns:
        pool = pool.with_columns([pl.lit(1).alias(count_field)])

    sub_pops = []
    for state in states:
        sub_pool = pool.filter(pl.col(att) == state)
        if len(sub_pool) == 0:
            print(f"WARNING: cannot see {att}_{state} in the pool, sample by the rest")
            sub_pool = pool  # if there are none, we take all
        for zone in marginal_data[zone_field]:
            condition = marginal_data.filter(pl.col(zone_field) == zone)
            census_val = condition.select(state).to_numpy()[0, 0]

            sub_syn_pop = sample_from_pl(sub_pool, census_val)

            sub_syn_pop = sub_syn_pop.with_columns([pl.lit(zone).alias(zone_field)])
            sub_syn_pop = sub_syn_pop.drop(count_field)

            sub_pops.append(sub_syn_pop)
    return pl.concat(sub_pops)


def wrapper_multiprocessing_zones(args):
    att, sub_syn_pop, zone_states_diff, pool, adjusted_atts = args
    # Process row with parameters
    result_zone_syn = zone_adjustment(att, sub_syn_pop, zone_states_diff, pool, adjusted_atts)
    return result_zone_syn


def adjust_atts_state_match_census(
    att: str,
    curr_syn_pop: Union[None, pd.DataFrame],
    census_data_by_att: pd.DataFrame,
    adjusted_atts: List[str],
    pool: pd.DataFrame,
) -> pd.DataFrame:
    print(f"ADJUSTING FOR {att}")
    if curr_syn_pop is None:
        updated_syn_pop = init_syn_pop_saa(att, census_data_by_att, pool).to_pandas()
    else:
        updated_syn_pop = curr_syn_pop

        states_diff_census = calculate_states_diff(
            att, curr_syn_pop, census_data_by_att
        )
        assert (states_diff_census.sum(axis=1) == 0).all()
        # Prepare arguments for each row
        args = [(att, updated_syn_pop[updated_syn_pop[zone_field] == zid], zone_states_diff.copy(deep=True), pool, adjusted_atts) for zid, zone_states_diff in states_diff_census.iterrows()]
        
        # Use multiprocessing Pool
        with mp.Pool(mp.cpu_count()) as pool:
            pop_syn_across_zones = pool.map(wrapper_multiprocessing_zones, args)
        # With state diff we can now do adjustment for each zone, can parallel it?
        # pop_syn_across_zones = []
        # for zid, zone_states_diff in states_diff_census.iterrows():
        #     print(f"DOING {zid}")
        #     sub_syn_pop = updated_syn_pop[updated_syn_pop[zone_field] == zid]
        #     zone_adjusted_syn_pop = zone_adjustment(
        #         att, sub_syn_pop, zone_states_diff, pool, adjusted_atts
        #     )
        #     if zone_adjusted_syn_pop is not None:
        #         pop_syn_across_zones.append(zone_adjusted_syn_pop)

        updated_syn_pop = pd.concat(pop_syn_across_zones)

    return updated_syn_pop
