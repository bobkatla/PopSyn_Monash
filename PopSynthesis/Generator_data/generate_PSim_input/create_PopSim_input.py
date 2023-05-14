import pandas as pd
import geopandas as gpd
from paras import seed_atts_H, seed_atts_P, census_atts, loc_file_census, loc_file_vista


def get_seed_P(atts, dict_new):
    df = pd.read_csv(f"{loc_file_vista}P_VISTA_1220_SA1.csv")
    df = df[atts]
    # Currently, the dict will be: 0-> hhid, the rest we will map
    hh_id = dict_new[0]
    for i in range(1, len(dict_new)):
        to_map = dict_new[i] 
        dict_map = dict(zip(hh_id, to_map))
        df[to_map.name] = df['hhid'].map(dict_map)
    # an extra step to filter out persons corresponding with the available households only
    df = df[df['hh_num'].notnull()]
    return df


def get_seed_H(atts):
    df = pd.read_csv(f"{loc_file_vista}H_VISTA_1220_SA1.csv")
    # Process the VISTA to match with LG
    df = df[atts]
    df = df[df['wdhhwgt_sa3'].notnull()]
    df['hh_num'] = df.index
    df = df.rename(columns={
        "homesa1": "SA1",
        "homesa2": "SA2",
        "homesa3": "SA3",
        "homesa4": "SA4",
    })
    return df


def get_census_sa(atts, sa_level):
    inside_atts = atts.copy()
    inside_atts.extend([
        f"{sa_level}_CODE_2021",
        f"{sa_level}_NAME_2021"
    ])
    gdf_p = gpd.read_file(f"{loc_file_census}G01_VIC_GDA2020.gpkg", layer=f"G01_{sa_level}_2021_VIC") # sex and agegroup
    gdf_h = gpd.read_file(f"{loc_file_census}G34_VIC_GDA2020.gpkg", layer=f"G34_{sa_level}_2021_VIC") # number veh
    # NOTE: combine later with other tables, now just this
    final_gdf = gdf_h.merge(gdf_p, how='inner')
    # The step with ls_zones to have only the zones in VISTA (greater Mel and geelong)
    df = pd.DataFrame(final_gdf)
    return df[inside_atts]


def get_geo_cross():
    # Maybe create my own MB file would be easier then clean from past, rule is simple, 
    # SA4: State is 2, sa4 is 200-299, sa3 is plus 2 digits, sa2 is plus more 4 digits, sa1 is plus more 2 digits
    df_mb = pd.read_csv("../data/source/MB_2016_VIC.csv")
    df_mb = df_mb[[
        "SA1_MAINCODE_2016",
        "SA2_MAINCODE_2016",
        "SA3_CODE_2016",
        "SA4_CODE_2016",
        "STATE_CODE_2016"
    ]]
    df_mb = df_mb.drop_duplicates()
    df_mb = df_mb.rename(columns={
        "SA1_MAINCODE_2016": "SA1",
        "SA2_MAINCODE_2016": "SA2",
        "SA3_CODE_2016": "SA3",
        "SA4_CODE_2016": "SA4",
        # "STATE_CODE_2016": "STATE"
    })
    return df_mb


def get_ls_needed_df(seed_atts_P, seed_atts_H, census_atts):
    seed_data_H = get_seed_H(seed_atts_H)

    dict_new = [
        seed_data_H['hhid'], 
        seed_data_H['hh_num'], 
        seed_data_H['SA1'], 
        seed_data_H['SA2'],
        seed_data_H['SA3'], 
        seed_data_H['SA4']
    ]
    seed_data_P = get_seed_P(seed_atts_P, dict_new=dict_new)

    census_data_sa1 = get_census_sa(census_atts, sa_level="SA1")
    census_data_sa2 = get_census_sa(census_atts, sa_level="SA2")
    census_data_sa3 = get_census_sa(census_atts, sa_level="SA3")
    census_data_sa4 = get_census_sa(census_atts, sa_level="SA4")

    geo_cross = get_geo_cross()

    return (
        (seed_data_P, "P_sample.csv",),
        (seed_data_H, "P_sample.csv",),
        (census_data_sa1, "P_sample.csv",),
        (census_data_sa2, "P_sample.csv",),
        (census_data_sa3, "P_sample.csv",),
        (census_data_sa4, "P_sample.csv",),
        (geo_cross, "P_sample.csv",),
    )


def output_csv(ls_to_csv, out_loc="./"):
    for f in ls_to_csv:
        f[0].to_csv(f"{out_loc}{f[1]}", index=False)


def main():
    ls_to_csv = get_ls_needed_df(seed_atts_P, seed_atts_H, census_atts)
    output_csv(ls_to_csv, out_loc="./")


def test():
    NotImplemented


if __name__ == "__main__":
    # main()
    test()
