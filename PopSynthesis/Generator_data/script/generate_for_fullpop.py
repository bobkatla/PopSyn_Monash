"""
This is the main file, to generate the full population and sample file
"""
import pandas as pd
import numpy as np


ATTRIBUTES = [
    "age",
    "hhsize",
    "relationship",
    "carlicence",
    "sex",
    "persinc",
    "dwelltype",
    "totalvehs",
    "wdhhwgt_sa3",
]


def flatten_connect_pp_hh(
    atts: list, hh_df: pd.DataFrame, pp_df: pd.DataFrame
) -> pd.DataFrame:
    # Connect hh and pp, hh is the base, connect with the main person (default the one did the survey)
    merge_df = pd.merge(pp_df, hh_df, on=["hhid"])
    df_atts = merge_df[atts].dropna()
    df_filter_only_self = df_atts[
        df_atts["relationship"] == "Self"
    ]  # NOTE: as these are people who do the survey so they will kinda be biased
    min_age, max_age = 0, 120  # NOTE: checked and min age is 15, max age is 116
    # Bin age into groups
    interval = 10
    bins = pd.IntervalIndex.from_tuples(
        [(begin, begin + interval) for begin in range(min_age, max_age + 1, interval)]
    )
    df_filter_only_self["age"] = pd.cut(
        df_filter_only_self["age"], bins
    ).dropna()  # This will give err if there are NaN
    df_filter_only_self = df_filter_only_self.drop(columns="relationship")
    return df_filter_only_self


def round_to_realise_df(flatten_df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    raw_non_weight = flatten_df.drop(columns=weight_col).to_numpy()
    weights = flatten_df[weight_col].to_numpy()
    cols = flatten_df.drop(columns=weight_col).columns

    final_raw = []
    for i in range(len(weights)):
        d = raw_non_weight[i]
        w = int(float(weights[i]))  # Simple rounding, can fix it to be better
        final_raw.append(np.repeat([d], w, axis=0))

    processed_final = np.concatenate(final_raw, axis=0)
    final_data = pd.DataFrame(processed_final, columns=cols)
    return final_data


def process_pop_to_get_control_and_marginal(realised_pop: pd.DataFrame) -> pd.DataFrame:
    # NOTE: this is mainly use in case we have a whole population data, or testing using the seed data
    d_marg = {"total": len(realised_pop)}
    d_con = {
        "target": [],
        "geography": [],
        "seed_table": [],
        "importance": [],
        "att": [],
        "state": [],
        "control_field": [],
        "expression": [],
    }
    for att in realised_pop.columns:
        data_seri = realised_pop[att]
        att_freq = data_seri.value_counts(sort=False).sort_index()
        for val in att_freq.index:
            tot_name = f"{att}__{val}"
            d_marg[tot_name] = att_freq[val]

            d_con["target"].append("NA")
            d_con["geography"].append("NA")
            d_con["seed_table"].append("flat_table")
            d_con["importance"].append(1000)
            d_con["att"].append(att)
            d_con["state"].append(val)
            d_con["control_field"].append(tot_name)
            d_con["expression"].append(f"flat_table.{att} == '{val}'")
    df_marg = pd.DataFrame(data=d_marg, index=[0])
    df_con = pd.DataFrame(data=d_con)

    return df_marg, df_con


def main():
    # This will be the 2021 file, will output full population with rounded to make it as the base
    data_loc = "../data/source2/VISTA/SA/"
    # import hh file
    hh_df = pd.read_csv(data_loc + "H_VISTA_1220_SA1.csv")
    # import pp file
    pp_df = pd.read_csv(data_loc + "P_VISTA_1220_SA1.csv")
    # connect to file -> flaten
    flatten_df = flatten_connect_pp_hh(atts=ATTRIBUTES, hh_df=hh_df, pp_df=pp_df)
    # rounding and retrive the full population
    realised_pop = round_to_realise_df(flatten_df, weight_col="wdhhwgt_sa3")
    # process and output the control and marginal file
    marginal_df, control_df = process_pop_to_get_control_and_marginal(realised_pop)

    # output the 3 files, this will be use for everything!!!
    output_loc = "../data/new_data_proces_2021/"
    realised_pop.to_csv(output_loc + "full_population_2021.csv", index=False)
    marginal_df.to_csv(output_loc + "marginal_2021.csv", index=False)
    control_df.to_csv(output_loc + "controls_2021.csv", index=False)


if __name__ == "__main__":
    main()
