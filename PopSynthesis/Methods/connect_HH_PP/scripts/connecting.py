import pandas as pd
import os, glob

from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score, learn_para_BN
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
import random
from numpy.random import multinomial


def _constrained_sum_sample_pos(n, total):
    # print(n, total)
    # """Return a randomly chosen list of n positive integers summing to total.
    # Each such list is equally likely to occur."""
    # dividers = sorted(random.sample(range(0, total), n-1))
    # return [a - b for a, b in zip(dividers + [total], [0] + dividers)]
    result = multinomial(total, [1/n] * n)
    # print(result)
    return result


def reject_samp_veh(BN, df_marg, zone_lev):
    inference = BayesianModelSampling(BN)
    ls_total_veh = [
        ("Num_MVs_per_dweling_4mo_MVs", [4, 5, 6, 7, 8, 9]), 
        ("Num_MVs_per_dweling_3_MVs", [3]), 
        ("Num_MVs_per_dweling_2_MVs", [2]), 
        ("Num_MVs_per_dweling_1_MVs", [1]),
        ("Num_MVs_per_dweling_0_MVs", [0]),
        ("Num_MVs_NS", None)]
    ls_all = []
    for zone in df_marg[zone_lev]:
        print(f"DOING {zone}")
        ls_re = []
        zone_info = df_marg[df_marg[zone_lev]==zone]
        assert len(zone_info) == 1
        for totveh_label in ls_total_veh:
            n_totvehs = zone_info[totveh_label[0]].iat[0]
            evidence = [[State('totalvehs', state)] for state in totveh_label[1]] if totveh_label[1] is not None else None
            # Weird case of multiple
            if evidence:
                if len(evidence) > 1:
                    ls_sz = _constrained_sum_sample_pos(len(evidence), n_totvehs)
                    for i in range(len(evidence)):
                        syn = inference.rejection_sample(evidence=evidence[i], size=ls_sz[i], show_progress=True)
                        ls_re.append(syn)   
                else:
                    syn = inference.rejection_sample(evidence=evidence[0], size=n_totvehs, show_progress=True)
                    ls_re.append(syn)
            else:
                syn = inference.forward_sample(size=n_totvehs, show_progress=True)
                ls_re.append(syn)
        if ls_re == []: continue
        final_for_zone = pd.concat(ls_re, axis=0)
        final_for_zone["SA1"] = zone
        ls_all.append(final_for_zone)
    final_result = pd.concat(ls_all, axis=0)
    return final_result


def main():
    # path = r'../data' # use your path
    # all_files = glob.glob(os.path.join(path , "connect*"))
    # for file in all_files:
    #     print(f"DOING {file}")
    #     df = pd.read_csv(file)
    #     # drop all the ids as they are not needed for in BN learning
    #     id_cols = [x for x in df.columns if "hhid" in x or "persid" in x]
    #     df = df.drop(columns=id_cols)
    #     print("Learn BN")
    #     model = learn_struct_BN_score(df, show_struct=True)
    #     model = learn_para_BN(model, df)
    #     print("Doing the sampling")

    #learning to get the HH only with main person
    df_seed = pd.read_csv("../data/connect_hh_main.csv")
    # drop all the ids as they are not needed for in BN learning
    id_cols = [x for x in df_seed.columns if "hhid" in x or "persid" in x]
    df_seed = df_seed.drop(columns=id_cols)
    print("Learn BN")
    model = learn_struct_BN_score(df_seed, show_struct=False)
    model = learn_para_BN(model, df_seed)
    print("Doing the sampling")
    census_df = pd.read_csv("../data/census_sa1.csv")
    final_syn_pop = reject_samp_veh(BN=model, df_marg=census_df, zone_lev="SA1")
    final_syn_pop.to_csv("SynPop_hh_main_sa1.csv", index=False)

if __name__ == "__main__":
    main()