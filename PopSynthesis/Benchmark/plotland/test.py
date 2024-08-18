import pandas as pd
from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score, learn_para_BN
from pgmpy.sampling import BayesianModelSampling


def main():
    file_loc = "./"
    syn_pop = pd.read_csv("synthetic_2021_HH.csv")
    census = pd.read_csv("census_SA1.csv")
    ls_census_name = census["SA1"].unique()
    ls_syn_pop_name = syn_pop["SA1"].unique()

    print(len(ls_census_name), len(ls_syn_pop_name))

    ls_not_exist = []
    for name in ls_census_name:
        if name not in ls_syn_pop_name:
            ls_not_exist.append(name)

    df = census[census["SA1"].isin(ls_not_exist)]
    df = df[["SA1", "Total_dwelings"]]
    print(df["Total_dwelings"].sum())

    new_pop = syn_pop.drop(columns="SA1")
    model = learn_struct_BN_score(new_pop, show_struct=False)
    model = learn_para_BN(model, new_pop)

    to_impute_df = []
    for sa1, tot_hh in zip(df["SA1"], df["Total_dwelings"]):
        inference = BayesianModelSampling(model)
        extra_df = inference.forward_sample(size=tot_hh)
        extra_df["SA1"] = sa1
        to_impute_df.append(extra_df)

    final_to_impute = pd.concat(to_impute_df)
    final_df = pd.concat([final_to_impute, syn_pop])

    # final_df.to_csv("./new_syn_2021_HH.csv")
    # print(final_df)

    check_new = []
    a = final_df["SA1"].unique()
    for name in ls_census_name:
        if name not in a:
            check_new.append(name)
    print(len(check_new))

    # Confirmed, all SA1 in synpop exist in census but 296 census not exist
    # After that imputation there are 279 still not exist, simple because they are 0 houses there


if __name__ == "__main__":
    main()
