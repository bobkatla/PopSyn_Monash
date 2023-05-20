"""
Process data to have the formated input for the IPF
"""
import pandas as pd


def get_sample_counts(df, ls_to_rm=None, ls_to_have=None):
    if ls_to_rm is None:
        ls_to_rm = [
            "hhid",
            "persid",
            "wdperswgt_sa3",
            "hh_num",
            "SA1",
            "SA2",
            "SA3",
            "SA4",
            "wdhhwgt_sa3"
        ]
    for to_rm in ls_to_rm:
        if to_rm in df.columns:
            df = df.drop(columns=to_rm)
    if ls_to_have:
        df = df[ls_to_have]
    return df.value_counts()


def process_data(seed, census, zone_lev, control, hh=True):
    # We are approaching this zone by zone
    seed_type = "households" if hh else "persons"
    control = control[control["seed_table"] == seed_type]

    final_results = {}
    for index, row in census.iterrows():
        # zone by zone
        zone = row[zone_lev]
        househodlds, persons, inside = seed[seed[zone_lev]==zone], seed[seed[zone_lev]==zone], seed[seed[zone_lev]==zone]

        ls_tups = []
        margi_val = []
        ls_to_have=[]

        for census_att, exp, att, state in zip(control["control_field"], control["expression"], control["att"], control["state"]):
            # Get the census val for that zone
            margi_val.append(row[census_att])

            ls_tups.append((att, state,))

            if att not in ls_to_have: ls_to_have.append(att)

            # filter_on_exp = eval(exp)
            # seed_val = filter_on_exp.value_counts()[True]
            # Because we only need the count so the below is not needed but maybe in the future
            # df_test = seed[filter_on_exp]

        # joint dist for IPF but only the bone
        j_cou = get_sample_counts(inside, ls_to_have=ls_to_have)

        midx = pd.MultiIndex.from_tuples(ls_tups)
        marginals = pd.Series(margi_val, index=midx)

        total = row["Total_dwelings"] if hh else row["Tot_P_P"]

        final_results[zone] = {
            "total": total,
            "seed": j_cou,
            "census": marginals
        }
    return final_results


def get_test_data():
    hh = pd.read_csv("../data/H_sample.csv")
    pp = pd.read_csv("../data/P_sample.csv")
    con = pd.read_csv("../controls/controls.csv")
    census_sa3 = pd.read_csv("../data/census_SA3.csv")
    return hh, pp, con, census_sa3


def test():
    hh, pp, con, census_sa3 = get_test_data()
    a = process_data(pp, census_sa3, "SA3", con, False)
    print(a)


if __name__ == "__main__":
    test()