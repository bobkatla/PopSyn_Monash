import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import fiona


if __name__ == "__main__":
	# a = fiona.listlayers('G01_VIC_GDA2020.gpkg')
	# print(a)
	# ['G01_GCCSA_2021_VIC', 'G01_UCL_2021_VIC', 'G01_SUA_2021_VIC', 'G01_SOSR_2021_VIC', 'G01_SOS_2021_VIC', 'G01_CED_2021_VIC', 'G01_LGA_2021_VIC', 'G01_POA_2021_VIC', 'G01_SA1_2021_VIC', 'G01_SA2_2021_VIC', 'G01_SA3_2021_VIC', 'G01_SA4_2021_VIC', 'G01_SAL_2021_VIC', 'G01_SED_2021_VIC', 'G01_STE_2021_VIC']
	gdf = gpd.read_file("./data/source2/CENSUS/G01_VIC_GDA2020.gpkg", layer="G01_LGA_2021_VIC")
	df = pd.read_csv("./data/source2/VISTA/H_VISTA_1220_LGA_V1.csv")
	
    # Process the VISTA to match with LG
	df['homeLGA'] = df['homeLGA'].str.replace(r"\(.*\)","", regex=True).str.replace(" ", "")
	gdf['LGA_NAME_2021'] = gdf['LGA_NAME_2021'].str.replace(r"\(.*\)","", regex=True).str.replace(" ", "")
	count_vista = dict(df['homeLGA'].value_counts())

	# I checked: all of the value in VISTA match with the one in CENSUS
	# t = list(gdf['LGA_NAME_2021'])
	# for a in count_vista:
	# 	if a not in t:
	# 		print(a)

	gdf["VISTA_count"] = gdf['LGA_NAME_2021'].map(count_vista, na_action='ignore').fillna(0)
	# gdf.plot()
	gdf.plot(column='VISTA_count', cmap='OrRd', edgecolor='k', legend=True)
	plt.show()
