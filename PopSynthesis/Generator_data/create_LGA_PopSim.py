'''
This file a quick design to get the needed data for the Population Sim run
'''

import pandas as pd
import geopandas as gpd


LS_ATTS_CENSUS = [
    "LGA_CODE_2021",
    "LGA_NAME_2021",
    "Tot_P_P",
    "Tot_P_M",
    "Tot_P_F",
    'Num_MVs_per_dweling_0_MVs',
    'Num_MVs_per_dweling_1_MVs',
    'Num_MVs_per_dweling_2_MVs',
    'Num_MVs_per_dweling_3_MVs',
    'Num_MVs_per_dweling_4mo_MVs',
    'Num_MVs_per_dweling_Tot',
    'Num_MVs_NS',
    'Total_dwelings',
]

LS_ATTS_HH = [
    'hhid',
    'homeLGA',
    'totalvehs',
    'hhsize',
    'wdhhwgt_LGA'
]

LS_ATTS_PP = [
    'hhid',
    'persid',
    "sex",
    "persinc"
]


def get_census_popsim(ls_atts, ls_zones=[]):
    gdf_p = gpd.read_file("./data/source2/CENSUS/G01_VIC_GDA2020.gpkg", layer="G01_LGA_2021_VIC") # sex and agegroup
    gdf_h = gpd.read_file("./data/source2/CENSUS/G34_VIC_GDA2020.gpkg", layer="G34_LGA_2021_VIC") # number veh
    # NOTE: combine later with other tables, now just this
    final_gdf = gdf_h.merge(gdf_p, how='inner')[ls_atts]
    final_gdf['LGA_NAME_2021'] = final_gdf['LGA_NAME_2021'].str.replace(r"\(.*\)","", regex=True).str.replace(" ", "")
    # The step with ls_zones to have only the zones in VISTA (greater Mel and geelong)
    df = pd.DataFrame(final_gdf)
    final_df = df[df['LGA_NAME_2021'].isin(ls_zones)]
    final_df = final_df.rename(columns={'LGA_NAME_2021': 'LGA'})
    return final_df


def get_hh_seed(ls_atts):
    df = pd.read_csv("./data/source2/VISTA/H_VISTA_1220_LGA_V1.csv")
    # Process the VISTA to match with LG
    df['homeLGA'] = df['homeLGA'].str.replace(r"\(.*\)","", regex=True).str.replace(" ", "")
    df = df[ls_atts]
    df = df[df['wdhhwgt_LGA'].notnull()]
    df = df.rename(columns={'homeLGA': 'LGA'})
    df['hh_num'] = df.index
    df['State'] = 2
    return df


def get_pp_seed(ls_atts, dict_lga, dict_new_id):
    df = pd.read_csv("./data/source2/VISTA/P_VISTA_1220_LGA_V1.csv")
    df = df[ls_atts]
    df['hh_num'] = df['hhid'].map(dict_new_id)
    # an extra step to filter out persons corresponding with the available households only
    df = df[df['hh_num'].notnull()]
    df['LGA'] = df['hhid'].map(dict_lga)
    df['State'] = 2
    return df


def simple_get_geo(ls_zones, state_num=2):
    # This will simple create a geo_cross with 2 cols, 1 ls_zones, 1 all the state num
    d = {"LGA": ls_zones}
    df = pd.DataFrame(data=d)
    df['State'] = state_num
    # df['Area'] = ['GreaterGeelongLarge' if x =='GreaterGeelong' else 'GreaterMelbourne' for x in df['LGA']]
    return df


def get_state_agg(df, state_num=2):
    # This will create a df that simply have the State as total/ aggregated of the given df
    # remove the zone col
    df = df.drop(['LGA_CODE_2021', 'LGA'], axis=1)
    df = df.aggregate(func='sum')
    df['State'] = state_num
    df = pd.DataFrame(df).transpose()
    return df


def get_mid_file_agg(df_lga):
    # Process Geelong only
    geelong = 'GreaterGeelong'
    geelong_df = df_lga[df_lga['LGA'] == geelong]
    geelong_df = geelong_df.rename(columns={'LGA': 'Area'})
    geelong_df['Area'] = 'GreaterGeelongLarge'
    geelong_df = geelong_df.drop(['LGA_CODE_2021'], axis=1)

    geelong_index = geelong_df.index[0]
    # Process Melbourne
    df = df_lga.drop(['LGA_CODE_2021', 'LGA'], axis=1)
    df = df.drop([geelong_index])
    df_mel = df.aggregate(func='sum')
    df_mel['Area'] = 'GreaterMelbourne'
    df_mel = pd.DataFrame(df_mel).transpose()

    final_df = pd.concat([geelong_df, df_mel])
    return final_df


if __name__ == "__main__":
    seed_hh_raw = get_hh_seed(LS_ATTS_HH)

    dict_lga = dict(zip(seed_hh_raw['hhid'], seed_hh_raw['LGA']))
    dict_new_id = dict(zip(seed_hh_raw['hhid'], seed_hh_raw['hh_num']))
    seed_pp_raw = get_pp_seed(LS_ATTS_PP, dict_lga=dict_lga, dict_new_id=dict_new_id)

    ls_zones = seed_hh_raw['LGA'].unique()
    census_LGA = get_census_popsim(LS_ATTS_CENSUS, ls_zones=ls_zones)
    geo_cross = simple_get_geo(ls_zones=ls_zones)
    census_state = get_state_agg(census_LGA)
    # mid_df = get_mid_file_agg(census_LGA)

    seed_hh_raw.to_csv("hh_seed.csv", index=False)
    seed_pp_raw.to_csv("pp_seed.csv", index=False)
    census_LGA.to_csv("LGA_controls.csv", index=False)
    geo_cross.to_csv("geo_cross_walk.csv", index=False)
    census_state.to_csv("state_controls.csv", index=False)
    # mid_df.to_csv("area_controls.csv", index=False)
