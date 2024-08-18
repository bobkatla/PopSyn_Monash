import synthpop.ipf.ipf as ipf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PopSynthesis.Benchmark.legacy.checker import update_SRMSE
from multiprocessing import Process, Lock, Array


def IPF_sampling(constraints):
    # constraints.to_csv('./Joint_dist_result_IPF.csv')
    ls = None
    for i in constraints.index:
        if constraints[i]:
            # TODO: instead of just rounding like this, use the papers method of int rounding
            ls_repeat = np.repeat([i], int(constraints[i]), axis=0)
            if ls is None:
                ls = ls_repeat
            else:
                ls = np.concatenate((ls, ls_repeat), axis=0)
    return pd.DataFrame(ls, columns=constraints.index.names)


def IPF_training(df, sample_rate):
    atts = df.columns
    ls_tups = []
    margi_val = []

    for att in atts:
        counts = df[att].value_counts()
        indexs = list(counts.index)
        for i, c in enumerate(counts):
            ls_tups.append((att, indexs[i]))
            margi_val.append(c)

    # Margi dist for IPF
    marginal_midx = pd.MultiIndex.from_tuples(ls_tups)
    marginals = pd.Series(margi_val, index=marginal_midx)

    # joint dist for IPF but only the bone
    j_cou = df.value_counts()
    j_idx = list(j_cou.index)
    # To solve zero cell by making a extremely small number
    j_vals = [1e-25] * len(j_idx)

    # Fill up the vals for joint from sample
    N = df.shape[0]
    one_percent = int(N / 100)
    seed_df = df.sample(n=sample_rate * one_percent).copy()

    seed_cou = seed_df.value_counts()
    seed_idx = list(seed_cou.index)
    for idx in seed_idx:
        i = j_idx.index(idx)
        j_vals[i] = seed_cou[idx]

    joint_dist_midx = pd.MultiIndex.from_tuples(j_idx, names=atts)
    joint_dist = pd.Series(j_vals, index=joint_dist_midx)

    constraints, iterations = ipf.calculate_constraints(
        marginals, joint_dist, tolerance=1e-5
    )
    return IPF_sampling(constraints)
    # print(iterations)


def multi_thread_f(df, s_rate, re_arr, l):
    print(f"START THREAD FOR SAMPLE RATE {s_rate}")
    check_time = 10
    re = 0
    for _ in range(check_time):
        sampling_df = IPF_training(df, s_rate)
        re += update_SRMSE(df, sampling_df)
    re = re / check_time
    # Calculate the SRMSE
    l.acquire()
    try:
        # NOTE: this is depends on the range we put the array, it should be same size but accessing the index is diff
        re_arr[s_rate - 1] = re
        print(f"DONE {s_rate}")
    finally:
        l.release()


def plot_SRMSE_IPF(original):
    # Maybe will not make this fixed like this
    X = range(1, 100)

    results = Array("d", X)
    lock = Lock()
    hold_p = []

    for i in X:
        p = Process(target=multi_thread_f, args=(original, i, results, lock))
        p.start()
        hold_p.append(p)
    for p in hold_p:
        p.join()

    print("DONE ALL, PLOTTING NOW")
    Y = results[:]
    plt.plot(X, Y)
    plt.xlabel("Percentages of sampling rate")
    plt.ylabel("SRMSE")
    plt.savefig("./img_data/IPF_SRMSE_final.png")
    plt.show()


if __name__ == "__main__":
    ATTRIBUTES = ["AGEGROUP", "CARLICENCE", "SEX", "PERSINC", "DWELLTYPE", "TOTALVEHS"]

    # import data
    p_original_df = pd.read_csv(
        "../../Generator_data/data/source/VISTA_2012_16_v1_SA1_CSV/P_VISTA12_16_SA1_V1.csv"
    )
    # Only have record of the main person (the person that did the survey)
    p_self_df = p_original_df[p_original_df["RELATIONSHIP"] == "Self"]
    h_original_df = pd.read_csv(
        "../../Generator_data/data/source/VISTA_2012_16_v1_SA1_CSV/H_VISTA12_16_SA1_V1.csv"
    )

    orignal_df = pd.merge(p_self_df, h_original_df, on=["HHID"])
    df = orignal_df[ATTRIBUTES].dropna()

    make_like_paper = True
    if make_like_paper:
        df.loc[df["TOTALVEHS"] == 0, "TOTALVEHS"] = "NO"
        df.loc[df["TOTALVEHS"] != "NO", "TOTALVEHS"] = "YES"

        df.loc[df["CARLICENCE"] == "No Car Licence", "CARLICENCE"] = "NO"
        df.loc[df["CARLICENCE"] != "NO", "CARLICENCE"] = "YES"

    result_sample = IPF_training(df, 5)
    print(update_SRMSE(df, result_sample))
    # print(SRMSE(df, result_sample))
    # print(result_sample)
    # plot_SRMSE_IPF(df)
