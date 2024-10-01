"""
Let's use this to compare with census to find the diff 

Should we include the check here as well (check whether we can del that values or not)
"""

import pandas as pd
from PopSynthesis.Methods.IPSF.const import zone_field, count_field


def calculate_states_diff(
    att: str, syn_pop: pd.DataFrame, sub_census: pd.DataFrame
) -> pd.DataFrame:
    """ This calculate the differences between current syn_pop and the census at a specific geo_lev """
    sub_syn_pop_count = syn_pop[[zone_field, att]].value_counts().reset_index()
    tranformed_sub_syn_count = sub_syn_pop_count.pivot(
        index=zone_field, columns=att, values=count_field
    ).fillna(0)
    sub_census = sub_census.set_index(zone_field)
    # Always census is the ground truth, check for missing and fill
    missing_zones = set(sub_census.index) - set(tranformed_sub_syn_count.index)
    missing_states = set(sub_census.columns) - set(tranformed_sub_syn_count.columns)

    for z in missing_zones:
        tranformed_sub_syn_count.loc[z] = 0
    for s in missing_states:
        tranformed_sub_syn_count[s] = 0

    results = sub_census - tranformed_sub_syn_count
    # no nan values
    assert not results.isna().any().any()
    return results
