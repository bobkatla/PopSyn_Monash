from pathlib import Path

zone_field = "zone_id"
hhid_field = "hhid"
count_field = "saa_count"

POOL_SIZE = 1e7

data_dir = Path(__file__).resolve() / "data" / "raw"
processed_dir = Path(__file__).resolve() / "data" / "processed"
output_dir = Path(__file__).resolve() / "output"