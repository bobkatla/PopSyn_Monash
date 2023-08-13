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

    model = _learn_struct_BN_score(df_seed, black_ls=black_ls, show_struct=True)
    model = _learn_para_BN(model, df_seed)

    return model


def _sampling_BN(BN, ls_zone, n):
    # Should return a pd.DataFrame
    ls_re = []
    

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


def main():
    # Import data
    df_seed_H = pd.read_csv("../data/H_sample.csv")
    # df_seed_P = pd.read_csv("../data/p_sample.csv")

    name_zone_lev = "SA1"
    df_census = pd.read_csv(f"../data/census_{name_zone_lev}.csv")
    ls_all_zones = df_census[name_zone_lev].astype("Int64").unique()
    # Extract missing zones

    df_seed = df_seed_H
    df_seed = df_seed.rename(columns={"wdhhwgt_sa3": "_weight"})

    ls_zones = df_seed[name_zone_lev].astype("Int64").unique()
    ls_missing_zones = _extract_missing_zones(ls_available=ls_all_zones, ls_zones=ls_zones ,zone_lev=name_zone_lev)
    # Learn BN
    BN = _learn_BN(df_seed=df_seed)
    print(BN)
    dummy_seed = _sampling_BN(BN, ls_zone=ls_missing_zones)
    # Combine dummy seed and original seed
    # Output the new seed, this will be the new input
    

if __name__ == "__main__":
    main()