seed_atts_P =[
    'hhid',
    'persid',
    "sex",
    "carlicence",
    "age",
    "fulltimework",
    "parttimework",
    "casualwork",
    "anywork",
    "studying",
    "wdperswgt_sa3"
] 
seed_atts_H = [
    'hhid',
    'homesa1',
    'homesa2',
    'homesa3',
    'homesa4',
    'totalvehs',
    'hhsize',
    "dwelltype",
    "owndwell",
    "hhinc",
    'wdhhwgt_sa3'
]
census_atts = [
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
    "Age_0_4_yr_P",
    "Age_5_14_yr_P",
    "Age_15_19_yr_P",
    "Age_20_24_yr_P",
    "Age_25_34_yr_P",
    "Age_35_44_yr_P",
    "Age_45_54_yr_P",
    "Age_55_64_yr_P",
    "Age_65_74_yr_P",
    "Age_75_84_yr_P",
    "Age_85ov_P",
    'Num_Psns_UR_1_Total',
    "Num_Psns_UR_2_Total",
    "Num_Psns_UR_3_Total",
    "Num_Psns_UR_4_Total",
    "Num_Psns_UR_5_Total",
    "Num_Psns_UR_6mo_Total",
    "P_LFS_NS_Tot", # labor force not stated
    "P_Not_in_LF_Tot", # not in labor force
    "P_Tot_LF_Tot", # tot labor force
    "P_Tot_Unemp_Tot", # unemployed (should be in labor force)
    "P_Tot_Emp_Tot", # employed
    "P_Hours_wkd_NS_Tot", # hour work not stated
    "P_Emp_awy_f_wrk_Tot", # away from work
    "P_Emp_PartT_Tot", # parttime
    "P_Emp_FullT_Tot", # full time
]

# Assuming all sample/VISTA data are in the same folder
loc_file_vista = "../data/source2/VISTA/SA/"
# Assuming all census data pack are in the same place, this is the .gpkg file
loc_file_census = "../data/source2/CENSUS/"

loc_file_convert = "../data/source2/CONVERT/"