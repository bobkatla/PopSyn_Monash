"""
File to double check after we got simple data
Mostly will the case of zero-cell (data exists in the census but not in sample)
--> we can aggregate or remove --> aggregate, maybe moveup 
Also we have the issue that in the sample, some code (sa1 and sa2) are old codes, just maybe has new code for 2021 now

Something todo, defo:
1. Taking only the Metro Mel (Greater Mel and Greater Geelong), they do exist all in VISTA
2. Check what SA don't exist in sample, alot sa1 yes, but maybe not sa2
3. Deal with zero cell (aggre how, maybe just combine with nearest SA)
"""

import geopandas as gpd
from paras import loc_file_census


def main():
    # df_census = pd.read_csv("./data/census_SA2.csv")
    # df_H = pd.read_csv("./data/H_sample.csv")
    # count=0
    # for z in df_census["SA2"]:
    #     if z not in df_H["SA2"]:
    #         print(z, "not")
    #         count += 1
    # print(count)
    sa_level = "SA4"
    gdf_p = gpd.read_file(
        f"{loc_file_census}G01_VIC_GDA2020.gpkg", layer=f"G01_{sa_level}_2021_VIC"
    )  # sex and agegroup
    gdf_h = gpd.read_file(
        f"{loc_file_census}G34_VIC_GDA2020.gpkg", layer=f"G34_{sa_level}_2021_VIC"
    )  # number veh
    # NOTE: combine later with other tables, now just this
    final_gdf = gdf_h.merge(gdf_p, how="inner")

    income_h = gpd.read_file(
        f"{loc_file_census}G33_VIC_GDA2020.gpkg", layer=f"G33_{sa_level}_2021_VIC"
    )
    print(final_gdf.columns)


if __name__ == "__main__":
    main()
