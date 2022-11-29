import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
import numpy as np
import networkx as nx
import pylab as plt

data_location = "../../../Generator_data/data/data_processed_here/"


def convert_to_prob(d1_arr):
    new_arr = []
    sum_arr = sum(d1_arr)
    for val in d1_arr:
        new_arr.append(val / sum_arr)
    return new_arr


def get_state_names(con_df):
    # return the dict of att and their states (order is important)
    state_names = {}
    ls_atts = con_df['att'].unique()
    for att in ls_atts:
        df_att = con_df[con_df['att']==att]
        state_names[att] = list(df_att['state'])
    return state_names


def cal_count_states(con_df, tot_df):
    #  calculate the prior of each 
    state_names = get_state_names(con_df)
    final_count = {}
    for att in state_names:
        ls_states = state_names[att]
        ls_count = [] # note that this would match with the ls_states
        for state in ls_states:
            tot_name = con_df[(con_df['att']==att) & (con_df['state']==state)]['tot_name']
            ls_count.append(tot_df[tot_name.iloc[0]].iloc[0])
        ls_prob = convert_to_prob(ls_count)
        final_count[att] = {
            'card': len(ls_states),
            'states': ls_states,
            'count': ls_count,
            'probs': ls_prob
        }
    return final_count


def learn_struct_BN_score(df, state_names=None, show_struct=False, method=HillClimbSearch, scoring_method='bicscore'):
    est = method(df, state_names=state_names)
    best_DAG = est.estimate(scoring_method=scoring_method)
    model = BayesianNetwork(best_DAG)
    if show_struct: 
        nx.draw_circular(model ,with_labels=True)
        plt.show()
    return model


def multiply_ls_arr(ls_arr):
    # NOTE: order matters, list of 1d arr, return 1 1d arr
    l_re = 1
    for i, arr in enumerate(ls_arr): 
        if len(arr) == 0: ls_arr.pop(i)
        else: l_re *= len(arr)

    re = [1 for _ in range(l_re)]
    mod_num = 1

    while ls_arr:
        last_arr = ls_arr.pop()
        l = len(last_arr)
        for i, val in enumerate(re):
            pos = int(((i % (mod_num*l)) - (i % mod_num)) / mod_num)
            re[i] = val * last_arr[pos]
        mod_num *= l
    
    return re


def get_prior(raw_model, con_df, tot_df):
    # return the list of prior counts and cpds, based on the census
    pri_counts = {}
    cpds = []

    final_counts = cal_count_states(con_df, tot_df)
    total = tot_df['total'].iloc[0]

    for att in final_counts:
        pa = raw_model.get_parents(att)
        att_probs = final_counts[att]['probs']
        vals_2d_matrix = []
        for prob in att_probs:
            ls_vals_states = [[prob]]
            for p in pa: ls_vals_states.append(final_counts[p]['probs'])
            final_vals = multiply_ls_arr(ls_vals_states)
            vals_2d_matrix.append(final_vals)
        
        pri_counts[att] = np.array(vals_2d_matrix) * total

        if pa: evidence_card = [final_counts[p]['card'] for p in pa]
        else: pa, evidence_card = None, None
        cpd_att = TabularCPD(
            att,
            final_counts[att]['card'],
            np.array(vals_2d_matrix),
            evidence=pa,
            evidence_card=evidence_card
        )
        cpds.append(cpd_att)
    
    return pri_counts, cpds


def main():
    con_df = pd.read_csv(data_location + "flat_con.csv")
    tot_df = pd.read_csv(data_location + "flat_marg.csv")
    seed_data = pd.read_csv(data_location + "flatten_seed_data.csv").astype(str)

    state_names = get_state_names(con_df)
    print(state_names.keys())

    model = learn_struct_BN_score(seed_data, state_names=state_names, show_struct=False)
    prior_counts, prior_cpds = get_prior(model, con_df, tot_df)
    # para_learn = BayesianEstimator(
    #     model=model,
    #     data=seed_data,
    #     state_names=state_names
    # )
    # ls_CPDs = para_learn.get_parameters()
    # print(ls_CPDs)
    model.add_cpds(*prior_cpds)

    # for att in state_names:
    #     print(model.get_parents(att))
    #     print(model.get_cpds(att).state_names)
    #     print(model.get_cpds(att))
    #     # print(model.get_cpds(att).get_values())
    # print(model.nodes())


if __name__ == '__main__':
    main()
