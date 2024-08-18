import pandas as pd
import geopandas as gpd


def get_census_popsim(ls_atts, ls_zones=[]):
    gdf_p = gpd.read_file(
        "./data/source2/CENSUS/G01_VIC_GDA2020.gpkg", layer="G01_LGA_2021_VIC"
    )  # sex and agegroup
    gdf_h = gpd.read_file(
        "./data/source2/CENSUS/G34_VIC_GDA2020.gpkg", layer="G34_LGA_2021_VIC"
    )  # number veh
    # NOTE: combine later with other tables, now just this
    final_gdf = gdf_h.merge(gdf_p, how="inner")[ls_atts]
    final_gdf["LGA_NAME_2021"] = (
        final_gdf["LGA_NAME_2021"]
        .str.replace(r"\(.*\)", "", regex=True)
        .str.replace(" ", "")
    )
    # The step with ls_zones to have only the zones in VISTA (greater Mel and geelong)
    df = pd.DataFrame(final_gdf)
    final_df = df[df["LGA_NAME_2021"].isin(ls_zones)]
    final_df = final_df.rename(columns={"LGA_NAME_2021": "LGA"})
    return final_df


def get_hh_seed(ls_atts, dict_of_new_cols):
    df = pd.read_csv("./data/source2/VISTA/H_VISTA_1220_LGA_V1.csv")
    # Process the VISTA to match with LG
    df = df[ls_atts]
    df = df[df["wdhhwgt_LGA"].notnull()]
    df["hh_num"] = df.index
    return df


def get_pp_seed(ls_atts, dict_lga, dict_new_id):
    df = pd.read_csv("./data/source2/VISTA/P_VISTA_1220_LGA_V1.csv")
    df = df[ls_atts]
    df["hh_num"] = df["hhid"].map(dict_new_id)
    # an extra step to filter out persons corresponding with the available households only
    df = df[df["hh_num"].notnull()]
    df["LGA"] = df["hhid"].map(dict_lga)
    return df


def main():
    df = pd.read_csv("./data/source2/VISTA/T_VISTA_1220_LGA_V1.csv")
    df = df[df["origplace2"] == "Survey Home"]
    hhid_SA1_dict = pd.Series(df["origSA1"].values, index=df["hhid"]).to_dict()
    hhid_SA2_dict = pd.Series(df["origSA2"].values, index=df["hhid"]).to_dict()
    hhid_SA3_dict = pd.Series(df["origSA3"].values, index=df["hhid"]).to_dict()
    hhid_SA4_dict = pd.Series(df["origSA4"].values, index=df["hhid"]).to_dict()
    df_hh = pd.read_csv("./data/source2/VISTA/H_VISTA_1220_LGA_V1.csv")
    df_hh = df_hh[df_hh["hhid"].notna()][["hhid", "homeLGA"]]
    df_hh["SA1"] = df_hh["hhid"].map(hhid_SA1_dict)
    df_hh["SA2"] = df_hh["hhid"].map(hhid_SA2_dict)
    df_hh["SA3"] = df_hh["hhid"].map(hhid_SA3_dict)
    df_hh["SA4"] = df_hh["hhid"].map(hhid_SA4_dict)
    print(df_hh[df_hh["SA1"].isna()])

    # Real nice, got the org place, filter to get at home then get the orgiSA1,2,3,4


if __name__ == "__main__":
    main()
    # gdf = gpd.read_file("./data/source2/CENSUS/G01_VIC_GDA2020.gpkg", layer="G01_POA_2021_VIC")
    # print(gdf.columns)
    # gdf = gdf[gdf['geometry'].notna()]
    # gdf['coords'] = gdf['geometry'].apply(lambda x: x.representative_point().coords[:])
    # gdf.plot(column='Tot_P_P', cmap='OrRd', edgecolor='k', legend=True)
    # for idx, row in gdf.iterrows():
    #     plt.annotate(text=row['POA_NAME_2021'], xy=row['coords'][0], horizontalalignment='center', fontsize=3)
    # plt.show()
