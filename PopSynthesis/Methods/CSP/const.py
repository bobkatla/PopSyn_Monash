from pathlib import Path


DATA_FOLDER = Path(__file__).parent / "data"
OUTPUT_FOLDER = Path(__file__).parent / "output"

HHID = "serialno" # same as in the seed as well as later in the synthesis
ZONE_ID = "zone_id"

HH_SZ = "hhsize" # we constrainted by this
HH_ATTS = [
    "dwelltype",
    "hhinc",
    "totalvehs",
    "owndwell"
]
RELATIONSHIP = "relationship"
PP_ATTS = [
    "age",
    "sex",
    "persinc",
    "nolicence",
    "anywork",
]

hh_seed_file_path = DATA_FOLDER / "hh_sample_ipu.csv"
pp_seed_file_path = DATA_FOLDER / "pp_sample_ipu.csv"