import pandas as pd
from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
import numpy as np
import networkx as nx
import pylab as plt
from itertools import chain
from PopSynthesis.Benchmark.legacy.checker import total_RMSE_flat, update_SRMSE

data_location = "../../../Generator_data/data/data_processed_here/"


def convert_to_prob(d1_arr):
    return [float(i)/sum(d1_arr) for i in d1_arr]


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
    learn_df = df
    if '_weight' in df:
        learn_df = df.drop(columns=['_weight'])
    est = method(learn_df, state_names=state_names)
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
    state_names = get_state_names(con_df)
    pri_counts = {}
    cpds = []

    final_counts = cal_count_states(con_df, tot_df)
    total = tot_df['total'].iloc[0]

    for att in final_counts:
        pa = sorted(raw_model.get_parents(att)) # DO this to match with the way they do it in pgmpy, dunno why they sort
        att_probs = final_counts[att]['probs']
        vals_2d_matrix = []
        for prob in att_probs:
            ls_vals_states = [[prob]]
            for p in pa: ls_vals_states.append(final_counts[p]['probs'])
            final_vals = multiply_ls_arr(ls_vals_states)
            vals_2d_matrix.append(final_vals)
        
        pri_counts[att] = np.array(vals_2d_matrix) * total

        evidence, evidence_card = None, None
        if pa: 
            evidence = pa
            evidence_card = [final_counts[p]['card'] for p in pa]
        cpd_att = TabularCPD(
            att,
            final_counts[att]['card'],
            np.array(vals_2d_matrix),
            state_names={var: state_names[var] for var in chain([att], pa)},
            evidence=evidence,
            evidence_card=evidence_card
        )
        cpd_att.normalize()
        cpds.append(cpd_att)
    
    return pri_counts, cpds

def compare_dist(model, data, pri_counts):
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
    cpds = _est.get_parameters(
            prior_type="dirichlet", pseudo_counts=pseudo_counts
        )
    return cpds


def dirichlet_loop_BN(model, prior_counts, n, con_df, tot_df, actual_df, ite=20, plot=False, selecting=True, select_ite=10):
    Y1, Y2 = [], []

    for _ in range(ite):
        inference = BayesianModelSampling(model)

        # simple selection
        syn_data = None
        if selecting:
            possible_pop = {}
            for _ in range(select_ite):
                syn = inference.forward_sample(size=n)
                possible_pop[total_RMSE_flat(syn, tot_df, con_df)] = syn
            syn_data = possible_pop[min(possible_pop)]
        else: syn_data = inference.forward_sample(size=n)
        
        # Y1.append(total_RMSE_flat(syn_data, tot_df, con_df))
        # Y2.append(update_SRMSE(actual_df, syn_data))
        cpds = compare_dist(model, syn_data, prior_counts)
        for c in cpds:
            c.normalize()
            model.add_cpds(c)
    if plot:
        fig, axs = plt.subplots(2)
        X = list(range(1, len(Y1)+1))
        axs[0].plot(X, Y1)
        axs[1].plot(X, Y2)
        plt.xlabel('Iteration')
        plt.ylabel('err')
        plt.show()
    return model


def loop_to_test(con_df, tot_df, ori_data, step=5, plot=False):
    N = len(ori_data)
    one_per = int(N/100)

    X, Y1, Y2 = [], [], []
    for i in range(1, 100, step):
        seed_data = ori_data.sample(n=(one_per*i))

        state_names = get_state_names(con_df)

        model = learn_struct_BN_score(
            seed_data, 
            state_names=state_names, 
            scoring_method='bicscore', 
            show_struct=False
            )
        prior_counts, prior_cpds = get_prior(model, con_df, tot_df)

        # para_learn = MaximumLikelihoodEstimator(model, seed_data)
        # ls_CPDs = para_learn.get_parameters()

        # para_learn = BayesianEstimator(
        #     model=model,
        #     data=seed_data,
        #     state_names=state_names
        # )
        # ls_CPDs = para_learn.get_parameters(
        #     prior_type='dirichlet',
        #     pseudo_counts = prior_counts
        # )
        # model.add_cpds(*ls_CPDs)
        model.add_cpds(*prior_cpds)

        final_model = dirichlet_loop_BN(
            model, 
            prior_counts, 
            tot_df['total'].iloc[0], 
            con_df, 
            tot_df, 
            ori_data,
            ite=10,
            select_ite=15,
            plot=False
            )

        inference = BayesianModelSampling(model)
        final_syn = inference.forward_sample(size=N)
        Y1.append(total_RMSE_flat(final_syn, tot_df, con_df))
        Y2.append(update_SRMSE(ori_data, final_syn))
        X.append(i)

    print('RMSE with Census', Y1)
    print('SRMSE', Y2)
    if plot:
        fig, axs = plt.subplots(2)
        axs[0].plot(X, Y1, label='RMSE with Census')
        axs[1].plot(X, Y2, label='SRMSE')
        plt.legend(loc="upper left")
        plt.xlabel('Iteration')
        plt.ylabel('Err')
        plt.show()


def test():
    con_df = pd.read_csv(data_location + "flat_con.csv")
    tot_df = pd.read_csv(data_location + "flat_marg.csv")
    ori_data = pd.read_csv(data_location + "flatten_seed_data.csv").astype(str)

    loop_to_test(con_df, tot_df, ori_data, plot=True)


if __name__ == '__main__':
    test()
