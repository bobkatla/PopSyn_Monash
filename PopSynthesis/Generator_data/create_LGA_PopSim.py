'''
This file a quick design to get the needed data for the Population Sim run
'''

import pandas as pd


LS_ATTS_CENSUS = [

]

LS_ATTS_HH = [

]

LS_ATTS_PP = [

]


def get_census_popsim(ls_atts, ls_zones=[]):
    NotImplemented


def get_hh_seed(ls_atts):
    NotImplemented


def get_pp_seed(ls_atts):
    NotImplemented


def simple_get_geo(ls_zones, state_num=2):
    # This will simple create a geo_cross with 2 cols, 1 ls_zones, 1 all the state num
    NotImplemented


def get_state_agg(df, state_nume=2):
    # This will create a df that simply have the State as total/ aggregated of the given df
    NotImplemented


if __name__ == "__main__":
    seed_hh_raw = get_hh_seed(LS_ATTS_HH)
    seed_pp_raw = get_pp_seed(LS_ATTS_PP)

    ls_zones = None
    census_LGA = get_census_popsim(LS_ATTS_CENSUS, ls_zones=ls_zones)
    geo_cross = simple_get_geo(ls_zones=ls_zones)
    census_state = get_state_agg(census_LGA)

    seed_hh_raw.to_csv("hh_seed.csv")
    seed_pp_raw.to_csv("pp_seed.csv")
    census_LGA.to_csv("LGA_controls.csv")
    geo_cross.to_csv("geo_cross_walk.csv")
    census_state.to_csv("state_controls.csv")
