seed_atts_P =[
    'hhid',
    'persid',
    "sex",
    "persinc",
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
]

# Assuming all sample/VISTA data are in the same folder
loc_file_vista = "../data/source2/VISTA/SA/"
# Assuming all census data pack are in the same place, this is the .gpkg file
loc_file_census = "../data/source2/CENSUS/"