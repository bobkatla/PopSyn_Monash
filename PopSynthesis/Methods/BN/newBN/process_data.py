import geopandas as gpd
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def get_hh_census():
    gdf_vel = gpd.read_file(
        "../../../Generator_data/data/source2/CENSUS/G34_VIC_GDA2020.gpkg",
        layer="G34_LGA_2021_VIC",
    )
    gdf_num_pp = gpd.read_file(
        "../../../Generator_data/data/source2/CENSUS/G35_VIC_GDA2020.gpkg",
        layer="G35_LGA_2021_VIC",
    )
    final_df = gdf_vel.merge(gdf_num_pp, how="inner")
    final_df["LGA_NAME_2021"] = (
        final_df["LGA_NAME_2021"]
        .str.replace(r"\(.*\)", "", regex=True)
        .str.replace(" ", "")
    )

    # Solving the unstate one so it can match
    final_df["Num_MVs_per_dweling_0_MVs_new"] = (
        final_df["Num_MVs_per_dweling_0_MVs"] + final_df["Num_MVs_NS"]
    )
    # print(final_df.columns)
    final_df = final_df.rename(columns={"LGA_NAME_2021": "LGA"})
    return pd.DataFrame(final_df)


def get_hh_census_sa1():
    gdf_vel = gpd.read_file(
        "../../../Generator_data/data/source2/CENSUS/G34_VIC_GDA2020.gpkg",
        layer="G34_SA1_2021_VIC",
    )
    gdf_num_pp = gpd.read_file(
        "../../../Generator_data/data/source2/CENSUS/G35_VIC_GDA2020.gpkg",
        layer="G35_SA1_2021_VIC",
    )
    final_df = gdf_vel.merge(gdf_num_pp, how="inner")

    # Solving the unstate one so it can match
    final_df["Num_MVs_per_dweling_0_MVs_new"] = (
        final_df["Num_MVs_per_dweling_0_MVs"] + final_df["Num_MVs_NS"]
    )
    # print(final_df.columns)
    final_df = final_df.rename(columns={"SA1_NAME_2021": "SA1"})
    final_df["SA1"] = final_df["SA1"].astype("float")
    return pd.DataFrame(final_df)


def get_hh_seed(atts=None):
    atts_seed = (
        atts if atts else ["hhid", "homeLGA", "hhsize", "totalvehs", "wdhhwgt_LGA"]
    )
    df_hh = pd.read_csv(
        "../../../Generator_data/data/source2/VISTA/H_VISTA_1220_LGA_V1.csv"
    )
    # df_pp = pd.read_csv("../../../Generator_data/data/source2/VISTA/H_VISTA_1220_LGA_V1.csv")

    df_hh_seed = df_hh[atts_seed]
    # extra processing to match
    df_hh_seed["homeLGA"] = (
        df_hh_seed["homeLGA"]
        .str.replace(r"\(.*\)", "", regex=True)
        .str.replace(" ", "")
    )
    df_hh_seed = df_hh_seed.dropna(subset=["wdhhwgt_LGA"])
    df_hh_seed = df_hh_seed.rename(columns={"homeLGA": "LGA", "wdhhwgt_LGA": "_weight"})
    return df_hh_seed


def get_hh_seed_sa1(atts=None):
    atts_seed = (
        atts if atts else ["hhid", "homesa1", "hhsize", "totalvehs", "wdhhwgt_sa3"]
    )
    df_hh = pd.read_csv(
        "../../../Generator_data/data/source/VISTA_1220_SA1_v3/H_VISTA_1220_SA1.csv"
    )
    # df_pp = pd.read_csv("../../../Generator_data/data/source2/VISTA/H_VISTA_1220_LGA_V1.csv")

    df_hh_seed = df_hh[atts_seed]
    # extra processing to match
    df_hh_seed = df_hh_seed.dropna(subset=["wdhhwgt_sa3"])
    df_hh_seed = df_hh_seed.rename(columns={"homesa1": "SA1", "wdhhwgt_sa3": "_weight"})
    df_hh_seed["SA1"] = df_hh_seed["SA1"].astype("float")
    return df_hh_seed


def main():
    df_hh_census = get_hh_census()
    df_hh_seed = get_hh_seed()

    # cut down the census to only have matching record with seed (only greater melbourne and geelong)
    df_hh_census = df_hh_census[
        df_hh_census["LGA"].isin(list(df_hh_seed["LGA"].unique()))
    ]

    df_con_hh = pd.read_csv("./controls/hh_con.csv")
    atts_census = list(df_con_hh["tot_name"].unique())
    atts_census.append("LGA")

    df_hh_census = df_hh_census[atts_census]

    households = df_hh_seed
    print(eval("households.hhsize>=6"))

    for ex, att, state in zip(
        df_con_hh["expression"], df_con_hh["att"], df_con_hh["state"]
    ):
        bool_ex = eval(ex)
        df_hh_seed.loc[bool_ex, att] = state
    print(df_hh_seed)

    return df_hh_census, df_hh_seed, df_con_hh


if __name__ == "__main__":
    main()
