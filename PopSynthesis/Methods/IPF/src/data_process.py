"""
Process data to have the formated input for the IPF
"""
import pandas as pd


def process_data(seed, census, zone_lev, control, hh=True):
    # We are approaching this zone by zone
    seed_type = "households" if hh else "persons"
    control = control[control["seed_table"] == seed_type]
    for census_att, exp in zip(control["control_field"], control["expression"]):
        for index, row in census.iterrows():
            # Get the census val for that zone
            zone = row[zone_lev]
            census_val = row[census_att]

            househodlds, persons = seed[seed[zone_lev]==zone], seed[seed[zone_lev]==zone]

            filter_on_exp = eval(exp)
            seed_val = filter_on_exp.value_counts()[True]
            # Because we only need the count so the below is not needed but maybe in the future
            # df_test = seed[filter_on_exp]
            print(seed_val)


def test():
    hh = pd.read_csv("../data/H_sample.csv")
    pp = pd.read_csv("../data/P_sample.csv")
    con = pd.read_csv("../controls/controls.csv")
    census = pd.read_csv("../data/census_SA3.csv")
    process_data(pp, census, "SA3", con, False)


if __name__ == "__main__":
    test()