import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'


def get_hh_census():
    gdf_vel = gpd.read_file("../../../Generator_data/data/source2/CENSUS/G34_VIC_GDA2020.gpkg", layer="G34_LGA_2021_VIC")
    gdf_num_pp = gpd.read_file("../../../Generator_data/data/source2/CENSUS/G35_VIC_GDA2020.gpkg", layer="G35_LGA_2021_VIC")
    final_df = gdf_vel.merge(gdf_num_pp, how="inner")
    final_df['LGA_NAME_2021'] = final_df['LGA_NAME_2021'].str.replace(r"\(.*\)","", regex=True).str.replace(" ", "")

    # Solving the unstate one so it can match
    final_df['Num_MVs_per_dweling_0_MVs_new'] = final_df['Num_MVs_per_dweling_0_MVs'] + final_df['Num_MVs_NS']
    # print(final_df.columns)
    return pd.DataFrame(final_df)


def get_hh_seed():
    atts_seed = [
        'hhid',
        'homeLGA',
        'hhsize',
        'totalvehs',
        'wdhhwgt_LGA'
    ]
    df_hh = pd.read_csv("../../../Generator_data/data/source2/VISTA/H_VISTA_1220_LGA_V1.csv")
    # df_pp = pd.read_csv("../../../Generator_data/data/source2/VISTA/H_VISTA_1220_LGA_V1.csv")

    df_hh_seed = df_hh[atts_seed]
    # extra processing to match
    df_hh_seed['homeLGA'] = df_hh_seed['homeLGA'].str.replace(r"\(.*\)","", regex=True).str.replace(" ", "")
    df_hh_seed = df_hh_seed.dropna(subset=['wdhhwgt_LGA'])
    return df_hh_seed


def main():
    df_hh_census = get_hh_census()
    df_hh_seed = get_hh_seed()

    # cut down the census to only have matching record with seed (only greater melbourne and geelong)
    df_hh_census = df_hh_census[df_hh_census['LGA_NAME_2021'].isin(list(df_hh_seed['homeLGA'].unique()))]

    df_con_hh = pd.read_csv("./controls/hh_con.csv")
    
    return df_hh_census, df_hh_seed, df_con_hh


if __name__ == "__main__":
    main()
