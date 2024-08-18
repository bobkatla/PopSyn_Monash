import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df_mel = pd.read_csv("BN_Melbourne_hh_2021.csv")
    sum_syn = df_mel[["LGA", "totalvehs"]].groupby("LGA").sum()
    sum_syn["LGA_NAME_2021"] = sum_syn.index
    gdf_vel = gpd.read_file(
        "../../../Generator_data/data/source2/CENSUS/G34_VIC_GDA2020.gpkg",
        layer="G34_LGA_2021_VIC",
    )
    gdf_vel["LGA_NAME_2021"] = (
        gdf_vel["LGA_NAME_2021"]
        .str.replace(r"\(.*\)", "", regex=True)
        .str.replace(" ", "")
    )
    final_df = gdf_vel.merge(sum_syn, how="inner")

    # final_df.plot(column='Num_MVs_per_dweling_Tot', cmap='OrRd', edgecolor='k', legend=True)
    final_df.plot(column="totalvehs", cmap="OrRd", edgecolor="k", legend=True)
    plt.show()
