from pathlib import Path


DATA_FOLDER = Path(__file__).parent / "data"
OUTPUT_FOLDER = Path(__file__).parent / "output"

HHID = "hhid"
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