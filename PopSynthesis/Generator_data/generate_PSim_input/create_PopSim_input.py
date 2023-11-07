import pandas as pd
import geopandas as gpd
from paras import seed_atts_H, seed_atts_P, census_atts, loc_file_census, loc_file_vista, loc_file_convert


def convert_2016_2021(df, sa_lev):
    convert_map = pd.read_csv(f"{loc_file_convert}CG_{sa_lev}_2016_{sa_lev}_2021.csv")

    old_name = f"{sa_lev}_CODE"
    if sa_lev in ("SA1", "SA2"):
        old_name = f"{sa_lev}_MAINCODE"

    convert_map = convert_map[[
        f"{old_name}_2016",
        f"{sa_lev}_CODE_2021"
    ]]
    # The last row is problematic
    convert_map = convert_map[:-1]
    
    convert_map[f"{sa_lev}_CODE_2021"] = convert_map[f"{sa_lev}_CODE_2021"].astype("float")
    # convert_map = convert_map[convert_map[f"{old_name}_2016"]!=convert_map[f"{sa_lev}_CODE_2021"]]

    dict_check = dict(zip(convert_map[f"{old_name}_2016"], convert_map[f"{sa_lev}_CODE_2021"]))
    df[sa_lev] = df[sa_lev].map(dict_check, na_action="ignore")

    return df


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

    # Process converting 2016-2021
    for sa in ("SA1", "SA2", "SA3", "SA4"):
        df = convert_2016_2021(df, sa)

    df["hh_num"] = df["hh_num"].astype("int")

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
    # Process converting 2016-2021
    for sa in ("SA1", "SA2", "SA3", "SA4"):
        df = convert_2016_2021(df, sa)

    df["hh_num"] = df["hh_num"].astype("int")

    return df


def census_process_extra(df):
    # This may create error if these atts are in not in, but this is just quick development
    df["TOT_CASUAL_GUESS"] = df["P_Tot_Emp_Tot"] - (df["P_Emp_PartT_Tot"] + df["P_Emp_FullT_Tot"])
    df["TOT_NOT_WORKING"] = df["Tot_P_P"] - df["P_Tot_Emp_Tot"]
    df["TOT_NOT_FT_WORK"] = df["Tot_P_P"] - df["P_Emp_FullT_Tot"]
    df["TOT_NOT_PT_WORK"] = df["Tot_P_P"] - df["P_Emp_PartT_Tot"]
    df["TOT_NOT_CASUAL_WORK"] = df["Tot_P_P"] - df["TOT_CASUAL_GUESS"]
    return df

def get_census_sa(atts, sa_level, ls_filter_zone=None):
    inside_atts = atts.copy()
    inside_atts.extend([
        f"{sa_level}_CODE_2021",
        f"{sa_level}_NAME_2021"
    ])
    gdf_p = gpd.read_file(f"{loc_file_census}G01_VIC_GDA2020.gpkg", layer=f"G01_{sa_level}_2021_VIC") # sex and agegroup
    gdf_h = gpd.read_file(f"{loc_file_census}G34_VIC_GDA2020.gpkg", layer=f"G34_{sa_level}_2021_VIC") # number veh
    # NOTE: combine later with other tables, now just this
    final_gdf = gdf_h.merge(gdf_p, how='inner')

    
    gdf_num_p = gpd.read_file(f"{loc_file_census}G35_VIC_GDA2020.gpkg", layer=f"G35_{sa_level}_2021_VIC") # number people inside
    final_gdf = final_gdf.merge(gdf_num_p, how='inner')

    gdf_employ = gpd.read_file(f"{loc_file_census}G46B_VIC_GDA2020.gpkg", layer=f"G46B_{sa_level}_2021_VIC") # num employment
    final_gdf = final_gdf.merge(gdf_employ, how='inner')

    # The step with ls_zones to have only the zones in VISTA (greater Mel and geelong)
    df = pd.DataFrame(final_gdf)[inside_atts]
    df = df.rename(columns={
        f"{sa_level}_CODE_2021": sa_level,
        f"{sa_level}_NAME_2021": "NAME"
    })
    
    if ls_filter_zone:
        df = df[df[sa_level].astype("float").isin(ls_filter_zone)]

    # Extra step of processing to match seed with census
    df = census_process_extra(df)

    return df


def get_geo_cross(ls_highest_level=None):
    # Maybe create my own MB file would be easier then clean from past, rule is simple, 
    # SA4: State is 2, sa4 is 200-299, sa3 is plus 2 digits, sa2 is plus more 4 digits, sa1 is plus more 2 digits
    df_mb = pd.read_csv("../data/source/MB_2021.csv")
    df_mb = df_mb[[
        "SA1_CODE_2021",
        "SA2_CODE_2021",
        "SA3_CODE_2021",
        "SA4_CODE_2021",
        "STATE_CODE_2021",
    ]]
    df_mb = df_mb.drop_duplicates()
    df_mb = df_mb.rename(columns={
        "SA1_CODE_2021": "SA1",
        "SA2_CODE_2021": "SA2",
        "SA3_CODE_2021": "SA3",
        "SA4_CODE_2021": "SA4",
        "STATE_CODE_2021": "STATE"
    })
    # getting Victoria only
    df_mb = df_mb[df_mb["STATE"]==2]

    # getting only SA4 that exist in sample
    if ls_highest_level:
        df_mb = df_mb[df_mb["SA4"].isin(ls_highest_level)]

    return df_mb.astype("float")


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

    # We can assume that we only need for SA4 higest level that exist in sample
    geo_cross = get_geo_cross(list(seed_data_H["SA4"].unique()))

    ls_results = [
        [seed_data_P, "P_sample.csv"],
        [seed_data_H, "H_sample.csv"],
        [geo_cross, "geo_cross.csv"],
    ]

    for sa in ("SA1", "SA2", "SA3", "SA4"):
        ls_zones = list(geo_cross[sa].unique())
        data_sa = get_census_sa(census_atts, sa_level=sa, ls_filter_zone=ls_zones)
        ls_results.append([data_sa, f"census_{sa}.csv"])

    return ls_results


def output_csv(ls_to_csv, out_loc="./"):
    for f in ls_to_csv:
        f[0].to_csv(f"{out_loc}{f[1]}", index=False)


def convert_to_int(list_files):
    ls_to_int = ("SA2", "SA3", "SA4")
    for f in list_files:
        df, name = f
        df = df.dropna()
        for t in ls_to_int:
            # print(name, t)
            if "census" not in name or t in name:
                df = df[df[t].notnull()]
                df[t] = df[t].astype("int")
        f[0] = df
    return list_files


def main():
    ls_to_csv = get_ls_needed_df(seed_atts_P, seed_atts_H, census_atts)
    ls_to_csv = convert_to_int(ls_to_csv)
    output_csv(ls_to_csv, out_loc="./data/")


def test():
    atts_hhsz = [
        "Num_Psns_UR_6mo_Total",
        "Num_Psns_UR_5_Total",
        "Num_Psns_UR_4_Total",
        "Num_Psns_UR_3_Total",
        "Num_Psns_UR_2_Total",
        "Num_Psns_UR_1_Total",
        "Total_dwelings"
    ]
    a = get_census_sa(census_atts, "POA")
    a.to_csv("POA_marg.csv", index=False)


if __name__ == "__main__":
    # main()
    test()
