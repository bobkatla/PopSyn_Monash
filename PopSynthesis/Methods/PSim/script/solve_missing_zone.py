"""
To generate SA1 we need to do with zero-cell issue,
one way (here) is to do BN and generate the seed for those area.
We can start with just random sample, but in the future,
each zone should be calibrated to fit with the census (or some other data)
"""

import pandas as pd
import bnlearn as bn
from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
import networkx as nx
import pylab as plt


def _learn_struct_BN_score(df, state_names=None, black_ls=None, show_struct=False, method=HillClimbSearch, scoring_method='bicscore'):
    learn_df = df
    if '_weight' in df:
        learn_df = df.drop(columns=['_weight'])
    est = method(learn_df, state_names=state_names)
    best_DAG = est.estimate(scoring_method=scoring_method, black_list=black_ls)
    model = BayesianNetwork(best_DAG)
    if show_struct: 
        nx.draw_circular(model ,with_labels=True)
        plt.show()
    return model


def _learn_para_BN(model, data_df):
    para_learn = BayesianEstimator(
            model=model,
            data=data_df
        )
    ls_CPDs = para_learn.get_parameters(weighted='_weight' in data_df)
    model.add_cpds(*ls_CPDs)
    return model


def _learn_BN(df_seed, root_att=None):
    # Should return the BN learnt from the data
    black_ls = None
    if root_att is not None:
        ls_atts = df_seed.columns
        black_ls = [(att, root_att) for att in ls_atts if att != root_att]
    # learn the struct   

    model = _learn_struct_BN_score(df_seed, black_ls=black_ls, show_struct=False)
    model = _learn_para_BN(model, df_seed)

    return model


def _sampling_BN(BN, ls_zone, df_marg):
    # Should return a pd.DataFrame
    inference = BayesianModelSampling(BN)
    age_values = []
    ls_re = []
    for zone in ls_zone:
        zone_info = df_marg[df_marg["SA1"]==zone]
        assert len(zone_info) == 1
        n = int(zone_info["Total_dwelings"])
        # evidence
        # syn = inference.rejection_sample(evidence=evidence, size=n, show_progress=False)

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
        

def _tempo_test(BN, df_marg, zone_lev):
    inference = BayesianModelSampling(BN)
    ls_hhsize = [
        ("Num_Psns_UR_6mo_Total", [6, 7, 8, 9, 10, 11]), 
        ("Num_Psns_UR_5_Total", [5]), 
        ("Num_Psns_UR_4_Total", [4]), 
        ("Num_Psns_UR_3_Total", [3]), 
        ("Num_Psns_UR_2_Total", [2]), 
        ("Num_Psns_UR_1_Total", [1])]
    ls_all = []
    for zone in df_marg[zone_lev]:
        print(f"DOING {zone}")
        ls_re = []
        zone_info = df_marg[df_marg[zone_lev]==zone]
        assert len(zone_info) == 1
        for hhsize_label in ls_hhsize:
            n_hhsz = int(zone_info[hhsize_label[0]])
            evidence = [[State('hhsize', state)] for state in hhsize_label[1]]
            # Weird case of multiple
            if len(evidence) > 1:
                ls_sz = _constrained_sum_sample_pos(len(evidence), n_hhsz)
                for i in range(len(evidence)):
                    syn = inference.rejection_sample(evidence=evidence[i], size=ls_sz[i], show_progress=True)
                    ls_re.append(syn)   
            else:
                syn = inference.rejection_sample(evidence=evidence[0], size=n_hhsz, show_progress=True)
                ls_re.append(syn)
        if ls_re == []: continue
        final_for_zone = pd.concat(ls_re, axis=0)
        final_for_zone["SA1"] = zone
        ls_all.append(final_for_zone)
    final_result = pd.concat(ls_all, axis=0)
    return final_result

    

def _extract_missing_zones(ls_available:list[str], ls_zones:list[str], zone_lev:str):
    # Assuming they are the same type
    assert ls_available.dtype == ls_zones.dtype
    ls_available = list(ls_available)
    ls_zones = list(ls_zones)
    # Should return a list of missing zones
    ls_unavai = []
    for zone in ls_available:
        if zone not in ls_zones:
            ls_unavai.append(zone)
    assert len(ls_available) == len(ls_unavai) + len(ls_zones)
    return ls_unavai


def _sample_some_missing(BN, old_synthetic_loc, census_data, zone_lev):
    inference = BayesianModelSampling(BN)
    old_df = pd.read_csv(old_synthetic_loc)
    ls_sa1 = old_df[zone_lev]
    old_counts = ls_sa1.value_counts()
    ls_new_df = [old_df]
    for zone in census_data[zone_lev]:
        print(f"DOINGGGG {zone}")
        zone_info = census_data[census_data[zone_lev]==zone]
        n_tot = int(zone_info["Total_dwelings"])
        exist_already = old_counts[zone] if zone in old_counts else 0
        n_to_sample = n_tot - exist_already
        assert n_to_sample >= 0
        re = inference.forward_sample(size=n_to_sample, show_progress=True)
        ls_new_df.append(re)
    final_df = pd.concat(ls_new_df, axis=0)
    print(final_df)
    return final_df


def main():
    # Import data
    df_seed_H = pd.read_csv("../data/H_sample.csv")
    # df_seed_P = pd.read_csv("../data/p_sample.csv")

    name_zone_lev = "POA"
    df_census = pd.read_csv(f"../data/census_{name_zone_lev}.csv")
    # ls_all_zones = df_census[name_zone_lev].astype("Int64").unique()
    # Extract missing zones

    df_seed = df_seed_H
    # print(df_seed_H["hhsize"].unique()) # till 11
    df_seed = df_seed.rename(columns={"wdhhwgt_sa3": "_weight"})

    # ls_zones = df_seed[name_zone_lev].astype("Int64").unique()
    # ls_missing_zones = _extract_missing_zones(ls_available=ls_all_zones, ls_zones=ls_zones ,zone_lev=name_zone_lev)
    df_seed = df_seed.drop(columns=["SA1", "SA2", "SA3", "SA4", "hhid", "hh_num"])
    # Learn BN
    BN = _learn_BN(df_seed=df_seed)
    # a = _sample_some_missing(BN, "./synthetic_2021_HH.csv", df_census, name_zone_lev)

    # dummy_seed = _sampling_BN(BN, ls_zone=ls_missing_zones, df_marg=df_census)
    final_re = _tempo_test(BN, df_census, name_zone_lev)
    final_re.to_csv("./synthetic_2021_HH_POA.csv", index=False)
    # Combine dummy seed and original seed
    # Output the new seed, this will be the new input
    

if __name__ == "__main__":
    main()