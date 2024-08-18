import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State

from PopSynthesis.Methods.BN.newBN.process_data import (
    get_hh_census_sa1,
    get_hh_seed_sa1,
)
from PopSynthesis.Methods.BN.TBN.utils import learn_struct_BN_score


def sampling_data(model, census, ls_lga):
    inference = BayesianModelSampling(model)
    result = []
    for lga, tot in zip(census["LGA"], census["Total_Total"]):
        print(lga, tot)
        if lga in ls_lga:
            final_syn = inference.rejection_sample(
                size=tot, evidence=[State("LGA", lga)]
            )
            result.append(final_syn)
    final_df = pd.concat(result)
    return final_df


def sampling_data_new(model, census):
    inference = BayesianModelSampling(model)
    result = []
    for sa1, tot in zip(census["SA1"], census["Total_Total"]):
        print(sa1, tot)
        final_syn = inference.rejection_sample(size=tot, evidence=[State("SA1", sa1)])
        result.append(final_syn)
    final_df = pd.concat(result)
    return final_df


def get_states(census, name_geo, seed, ls_atts):
    state_names = {}
    state_names[name_geo] = list(census[name_geo].unique())
    for att in ls_atts:
        if att != "wdhhwgt_sa3" and att != "homesa1":
            state_names[att] = list(seed[att].unique())
    return state_names


def check_census_match_sa1(census, seed, geo):
    for state in seed[geo].unique():
        if state in census[geo].unique():
            print(state, "YES")
        else:
            print(state, "NO")
            # NOTE: future will map them to their current SA1 instead of just drop
            seed = seed.drop(seed[seed[geo] == state].index)
    return seed


def main():
    atts = [
        "homesa1",
        "hhsize",
        "dwelltype",
        "hhinc",
        "owndwell",
        "adultbikes",
        "kidsbikes",
        "cars",
        "fourwds",
        "utes",
        "vans",
        "trucks",
        "mbikes",
        "othervehs",
        "aveage",
        "wdhhwgt_sa3",
    ]
    df_hh_seed = get_hh_seed_sa1(atts)
    df_hh_census = get_hh_census_sa1()
    df_hh_seed = check_census_match_sa1(df_hh_census, df_hh_seed, "SA1")

    state_names = get_states(df_hh_census, "SA1", df_hh_seed, atts)

    model = learn_struct_BN_score(df_hh_seed, state_names=state_names, show_struct=True)
    para_learn = BayesianEstimator(model=model, data=df_hh_seed)
    ls_CPDs = para_learn.get_parameters(prior_type="BDeu", weighted=True)
    model.add_cpds(*ls_CPDs)

    final_df = sampling_data_new(model, df_hh_census)
    final_df.to_csv("BN_Melbourne_hh_full_BL.csv", index=False)


if __name__ == "__main__":
    main()
