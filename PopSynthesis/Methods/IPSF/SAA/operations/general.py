""" Contains the funcs for SAA, help seperate out and easier debugging"""

import polars as pl
import pandas as pd
import numpy as np

from typing import List, Union, Tuple, Dict
from PopSynthesis.Methods.IPSF.const import count_field, zone_field
from PopSynthesis.Methods.IPSF.SAA.operations.compare_census import calculate_states_diff
from PopSynthesis.Methods.IPSF.SAA.operations.ILP_zone_ad import ILP_zone_adjustment
from PopSynthesis.Methods.IPSF.utils.condensed import condense_df
import sys
from PopSynthesis.Methods.IPSF.SAA.operations.shared_vars import update_zero_cells

def process_raw_ipu_marg(
    marg: pd.DataFrame, atts: List[str]
) -> Dict[str, pd.DataFrame]:
    segmented_marg = {}
    zones = marg.index.values
    for att in atts:
        sub_marg = marg[marg.columns[marg.columns.get_level_values(0) == att]]
        if sub_marg.empty:
            print(f"Don't have this att {att} in census")
            continue
        sub_marg.columns = sub_marg.columns.droplevel(0)
        sub_marg.loc[:, [zone_field]] = zones
        sub_marg = sub_marg.set_index(zone_field)
        segmented_marg[att] = sub_marg
    return segmented_marg


def process_raw_ipu_init(
    marg: pd.DataFrame, seed: pd.DataFrame
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    atts = [x for x in seed.columns if x not in ["serialno", "sample_geog"]]
    segmented_marg = process_raw_ipu_marg(marg, atts)
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
    att: str, marginal_data: pl.DataFrame, pool: pl.DataFrame
) -> pl.DataFrame:
    assert zone_field in marginal_data
    states = list(pool[att].unique())
    assert set(states + [zone_field]) <= set(marginal_data.columns)

    zero_cells_cases = set(marginal_data.columns) - set(states + [zone_field])
    if len(zero_cells_cases) > 0:
        for case in zero_cells_cases:
            update_zero_cells(att, case)
        states = states + list(zero_cells_cases)

    if count_field not in pool.columns:
        pool = pool.with_columns([pl.lit(1).alias(count_field)])

    sub_pops = []
    for state in states:
        sub_pool = pool.filter(pl.col(att) == state)
        if len(sub_pool) == 0:
            sub_pool = pool  # handle zero-cells
        for zone in marginal_data[zone_field]:
            condition = marginal_data.filter(pl.col(zone_field) == zone)
            census_val = condition.select(state).to_numpy()[0, 0]

            sub_syn_pop = sample_from_pl(sub_pool, census_val)

            sub_syn_pop = sub_syn_pop.with_columns([pl.lit(zone).alias(zone_field)])
            sub_syn_pop = sub_syn_pop.drop(count_field)

            sub_pops.append(sub_syn_pop)
    return pl.concat(sub_pops)


def adjust_atts_state_match_census(
    att: str,
    curr_syn_pop: Union[None, pl.DataFrame],
    census_data_by_att: pl.DataFrame,
    adjusted_atts: List[str],
    pool_count: pl.DataFrame,
    include_value: bool = False,
) -> pd.DataFrame:
    print(f"ADJUSTING FOR {att}")
    updated_syn_pop = None
    if curr_syn_pop is None:
        updated_syn_pop = init_syn_pop_saa(att, census_data_by_att, pool_count)
    else:
        states_diff_census = calculate_states_diff(
            att, curr_syn_pop, census_data_by_att
        )
        assert (states_diff_census.select(pl.exclude([zone_field])).sum_horizontal()==0).all()
        # With state diff we can now do adjustment for each zone, can parallel it?
        pop_syn_across_zones = []
        records_err = {}
        for zone_marg in states_diff_census.iter_rows(named=True):
            zid = zone_marg.pop(zone_field)
            sys.stdout.write(f"\rDOING zone {zid}")
            sys.stdout.flush()
            sub_syn_pop = curr_syn_pop.filter(pl.col(zone_field) == zid)
            if not sub_syn_pop.is_empty():
                condensed_syn = condense_df(sub_syn_pop)
                zone_adjusted_syn_pop, err_remain = ILP_zone_adjustment(
                    att, condensed_syn, zone_marg, pool_count, adjusted_atts, include_value=include_value
                )
                records_err[zid] = err_remain
                if zone_adjusted_syn_pop is not None:
                    assert len(zone_adjusted_syn_pop) == len(sub_syn_pop)
                    pop_syn_across_zones.append(zone_adjusted_syn_pop.select(curr_syn_pop.columns))
        print()
        updated_syn_pop = pl.concat(pop_syn_across_zones)
        assert len(updated_syn_pop) == len(curr_syn_pop)

    return updated_syn_pop
