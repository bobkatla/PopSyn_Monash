"""
Let's use this to compare with census to find the diff 

Should we include the check here as well (check whether we can del that values or not)
"""

import polars as pl
from PopSynthesis.Methods.IPSF.const import zone_field, count_field


def calculate_states_diff(
    att: str, syn_pop: pl.DataFrame, sub_census: pl.DataFrame
) -> pl.DataFrame:
    """ This calculate the differences between current syn_pop and the census at a specific geo_lev """
    sub_syn_pop_count = syn_pop.group_by([zone_field, att]).len(name=count_field)
    tranformed_sub_syn_count = sub_syn_pop_count.pivot(
        index=zone_field, columns=att, values=count_field
    ).fill_nan(0)
    # Always census is the ground truth, check for missing and fill
    # Missing zones
    missing_zones = set(sub_census[zone_field]) - set(
        tranformed_sub_syn_count[zone_field]
    )
    add_missing_zones = pl.DataFrame(
        {zone_field: list(missing_zones)}
        | {
            att: [0] * len(missing_zones)
            for att in tranformed_sub_syn_count.columns
            if att != zone_field
        },
        schema=tranformed_sub_syn_count.schema,
    )
    tranformed_sub_syn_count = tranformed_sub_syn_count.vstack(add_missing_zones)

    # Missing states
    missing_states = set(sub_census.columns) - set(tranformed_sub_syn_count.columns)
    tranformed_sub_syn_count = tranformed_sub_syn_count.with_columns(
        [pl.lit(0).alias(s) for s in missing_states]
    )

    sub_census = sub_census.to_pandas().set_index(zone_field).fillna(0)
    tranformed_sub_syn_count = (
        tranformed_sub_syn_count.to_pandas().set_index(zone_field).fillna(0)
    )
    results = sub_census - tranformed_sub_syn_count
    # no nan values
    assert not results.isna().any().any()
    return pl.from_pandas(results.reset_index())
