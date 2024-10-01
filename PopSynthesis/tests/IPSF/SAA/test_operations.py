from PopSynthesis.Methods.IPSF.SAA.operations.general import init_syn_pop_saa
from PopSynthesis.Methods.IPSF.const import zone_field

from pathlib import Path
import polars as pl
import pandas as pd


data_folder = Path(__file__).parent.parent.parent.resolve() / "test_data"
expected_folder = None

hh_marg = pd.read_csv(data_folder / "hh_marg_minor_test.csv", header=[0, 1])
hh_sample = pd.read_csv(data_folder / "hh_sample_minors.csv")


def test_init_syn_pop():
    atts = [x for x in hh_sample.columns if x not in ["serialno", "sample_geog"]]
    segmented_marg = {}
    zones = hh_marg[
        hh_marg.columns[hh_marg.columns.get_level_values(0) == zone_field]
    ].values
    zones = [z[0] for z in zones]
    for att in atts:
        sub_marg = hh_marg[hh_marg.columns[hh_marg.columns.get_level_values(0) == att]]
        sub_marg.columns = sub_marg.columns.droplevel(0)
        sub_marg.loc[:, [zone_field]] = zones
        segmented_marg[att] = pl.from_pandas(sub_marg)
    results_run = init_syn_pop_saa(
        "hhsize", segmented_marg["hhsize"], pl.from_pandas(hh_sample)
    )
    for zone in zones:
        print(segmented_marg["hhsize"].filter(pl.col(zone_field) == zone))
        print(results_run.filter(pl.col(zone_field) == zone)["hhsize"].value_counts())


test_init_syn_pop()
