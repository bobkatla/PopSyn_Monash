import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
import pylab as plt
from benchmark import SRMSE
import matplotlib.pyplot as plt

location_of_processed_data = "../../data/data_processed"


def input():
    df_h_ori = pd.read_csv(location_of_processed_data + "/h_test_seed.csv")
    ATTRIBUTES = ["HHSIZE", "CARS", "TOTALVEHS", "CW_ADHHWGT_SA3", "SA3"]
    df = df_h_ori[ATTRIBUTES].dropna()
    df = df.rename(columns={"CW_ADHHWGT_SA3": "_weight"})
    return df


def input_SA3_20904():
    df = input()
    df = df[df["SA3"] == 20904]
    return df


def tot_SA3_20904():
    df_tot = pd.read_csv(location_of_processed_data + "/SA3_controls.csv")
    df = df_tot[df_tot["SA3_CODE_2016"] == 20904]
    df = df.drop(columns="SA3_CODE_2016")
    return df


pri_counts = {
    "HHSIZE": [
        [
            501.7725737,
            3410.708717,
            4819.874374,
            1590.543415,
            291.03364,
            291.03364,
            291.03364,
        ],
        [
            876.5782217,
            5958.382619,
            8420.143166,
            2778.620815,
            508.4250594,
            508.4250594,
            508.4250594,
        ],
        [
            607.0480092,
            4126.299533,
            5831.118113,
            1924.250674,
            352.0945564,
            352.0945564,
            352.0945564,
        ],
        [
            662.1282605,
            4500.697623,
            6360.202215,
            2098.846767,
            384.0417111,
            384.0417111,
            384.0417111,
        ],
        [
            284.5887677,
            1934.440903,
            2733.672944,
            902.1034908,
            165.0646315,
            165.0646315,
            165.0646315,
        ],
        [
            74.08271386,
            503.5639076,
            711.6159647,
            234.8310347,
            42.96879304,
            42.96879304,
            42.96879304,
        ],
        [
            74.08271386,
            503.5639076,
            711.6159647,
            234.8310347,
            42.96879304,
            42.96879304,
            42.96879304,
        ],
    ],
    "TOTALVEHS": [[2985], [20290], [28673], [9462], [1731.333], [1731.333], [1731.333]],
}


def compare_dist(model, data):
    # All  the vars list and the corresponding totol control field
    # DAG to get the parents of each nodes and then we get the cardinality of each to calculate the shape
    # Maybe need to fix this with state setting so we have the correct probability (tempo fix, divide it)
    pseudo_counts = {}
    for var in model.nodes():
        # pseudo_counts[var] = (model.get_cpds(var).get_values() * 68732) - pri_counts[var]
        pseudo_counts[var] = pri_counts[var]
    state_names = {}
    for var in model.nodes():
        state_names.update(model.get_cpds(var).state_names)

    _est = BayesianEstimator(model, data, state_names=state_names)
    cpds = _est.get_parameters(prior_type="dirichlet", pseudo_counts=pseudo_counts)
    return cpds


if __name__ == "__main__":
    # Prep
    df_h_all = input()
    df_struc_learn = df_h_all.drop(columns="_weight")
    df_struc_learn = df_struc_learn.drop(columns="SA3")
    df_struc_learn = df_struc_learn.drop(columns="CARS")

    tot_df_SA3_20904 = tot_SA3_20904()
    df_h_SA3_20904 = input_SA3_20904()

    con_df = pd.read_csv("controls_files/hh_controls.csv")

    # struct learning
    # Do I need to set up the states?
    est = HillClimbSearch(df_struc_learn)
    best_DAG = est.estimate(scoring_method=BicScore(df_struc_learn))
    model = BayesianNetwork(best_DAG)

    # model.fit()
    # Para learning
    # Auto got weights
    state_names = {"HHSIZE": [1, 2, 3, 4, 5, 6, 7], "TOTALVEHS": [0, 1, 2, 3, 4, 5, 6]}

    Y = []
    # first run
    para_learn = BayesianEstimator(
        model=model, data=df_h_SA3_20904, state_names=state_names
    )
    ls_CPDs = para_learn.get_parameters(
        prior_type="dirichlet", pseudo_counts=pri_counts
    )
    model.add_cpds(*ls_CPDs)
    inference = BayesianModelSampling(model)
    syn_data = inference.forward_sample(size=68732)
    Y.append(SRMSE(syn_data, tot_df_SA3_20904, con_df))

    for i in range(20):
        cpds = compare_dist(model, syn_data)
        for c in cpds:
            c.values = np.absolute(c.values)
            c.normalize()
            model.add_cpds(c)
        inference = BayesianModelSampling(model)
        syn_data = inference.forward_sample(size=68732)
        Y.append(SRMSE(syn_data, tot_df_SA3_20904, con_df))
    X = list(range(1, len(Y) + 1))
    plt.plot(X, Y)
    plt.xlabel("Iteration")
    plt.ylabel("SRMSE")
    plt.show()
    # nx.draw_circular(model ,with_labels=True)
    # plt.show()
