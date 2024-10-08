from pathlib import Path
import logging
import os
from PopSynthesis.Methods.connect_HH_PP.paras_dir import log_dir
import datetime

# ct stores current time
ct = datetime.datetime.now()
ct = str(ct).replace(".", "-").replace(":", "-").replace(" ", "-")

# create logger
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger("connect_hh_pp")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(os.path.join(log_dir, f"connect_hh_pp_{ct}.log"))
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)

LS_HH_INC = [
    "$1-$149 ($1-$7,799)",
    "$150-$299 ($7,800-$15,599)",
    "$300-$399 ($15,600-$20,799)",
    "$400-$499 ($20,800-$25,999)",
    "$500-$649 ($26,000-$33,799)",
    "$650-$799 ($33,800-$41,599)",
    "$800-$999 ($41,600-$51,999)",
    "$1,000-$1,249 ($52,000-$64,999)",
    "$1,250-$1,499 ($65,000-$77,999)",
    "$1,500-$1,749 ($78,000-$90,999)",
    "$1,750-$1,999 ($91,000-$103,999)",
    "$2,000-$2,499 ($104,000-$129,999)",
    "$2,500-$2,999 ($130,000-$155,999)",
    "$3,000-$3,499 ($156,000-$181,999)",
    "$3,500-$3,999 ($182,000-$207,999)",
    "$4,000-$4,499 ($208,000-$233,999)",
    "$4,500-$4,999 ($234,000-$259,999)",
    "$5,000-$5,999 ($260,000-$311,999)",
    "$6,000-$7,999 ($312,000-$415,999)",
    "$8,000 or more ($416,000 or more)",
]

HH_ATTS = ["hhid", "dwelltype", "owndwell", "hhinc", "totalvehs", "hhsize"]

PP_ATTS = [
    "persid",
    "hhid",
    "age",
    "sex",
    "relationship",
    "persinc",
    "nolicence",
    "anywork",
]
NOT_INCLUDED_IN_BN_LEARN = ["serialno", "persid", "relationship", "sample_geog"]

zone_field = "zone_id"
hhid_field = "serialno"
count_field = "count"  # THIS IS THE DEFAULT FROM value_counts in pandas

POOL_SIZE = int(1e7)

data_dir = Path(__file__).parent.resolve() / "data" / "raw"
processed_dir = Path(__file__).parent.resolve() / "data" / "processed"
output_dir = Path(__file__).parent.resolve() / "output"
