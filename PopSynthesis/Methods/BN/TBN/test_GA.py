import pandas as pd
import numpy as np
import math

from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import State

from PopSynthesis.Methods.BN.TBN.utils import learn_struct_BN_score, compare_dist, get_prior, get_state_names, cal_count_states
from PopSynthesis.Benchmark.checker import total_RMSE_flat, update_SRMSE

"""
    Some parameters:
    - How to pick parents and the individuals for next round (maybe not the same as eval func)
    - How many solutions per gen
    - How to do crossover (producing offsprings)
    - How to mutate and where/who, to what extent at what rate
    - How many generations or the termination criteria
    - Heuristic approach, how? 
"""


def learn_BN_diriclet(data_df, con_df, tot_df):
    # this will learn the whole thing in one
    # structure learning
    state_names = get_state_names(con_df)
    model = learn_struct_BN_score(
            data_df, 
            state_names=state_names, 
            scoring_method='bicscore', 
            show_struct=False
            )
    prior_counts, prior_cpds = get_prior(model, con_df, tot_df)
    # parameter learning
    para_learn = BayesianEstimator(
            model=model,
            data=data_df,
            state_names=state_names
        )
    ls_CPDs = para_learn.get_parameters(
        prior_type='dirichlet',
        pseudo_counts = prior_counts
    )
    model.add_cpds(*ls_CPDs)
    return model


def sample_BN(model, n, typeOf='forward', evidence=None):
    inference = BayesianModelSampling(model)
    syn = None
    if typeOf == 'forward':
        if evidence: print("Using forward sampling, the evidence you provided would not have any effect")
        syn = inference.forward_sample(size=n)
    elif typeOf == 'rejection':
        syn = inference.rejection_sample(evidence=evidence, size=n)
    else:
        print("ERRORRRRR, not sure what type of sampling is that")
    return syn


def eval_func(indi, tot_df, con_df):
    score = total_RMSE_flat(indi, tot_df, con_df)
    return score


def softmax(arr_prob):
    exps = np.exp(arr_prob - np.max(arr_prob))
    return exps / np.sum(exps)


def cross_entropy(dist_target, dist_check):
    # Note that they will be processed using softmax, the reason is to solve the 0 value from syn_pop
    # dist_target = softmax(dist_target)
    # dist_check = softmax(dist_check)

    # H(target, approx/check)
    # H(P, Q) = – sum x in X P(x) * log(Q(x))

    assert len(dist_target) == len(dist_check)

    # They will be 2 same size array for the same atts, 
    # The order matters, assuming that for the same position is the same event/state

    result = 0
    for i in range(len(dist_target)):
        result += dist_target[i] * math.log(dist_check[i], 2) # in bits, change to e if want nats
    return -result


def get_dist_syn_pop(syn_pop_seri, ls_states):
    val_count = syn_pop_seri.value_counts(normalize=True)
    # order matters, has to match with others, that is why there is the ls_states
    re = []
    assert len(ls_states) >= len(val_count)
    for state in ls_states:
        re.append(val_count[state] if state in val_count else 0)
    return re
    

def best_fit_atts(syn_pop, con_df, tot_df, num_att=1):
    att_char = cal_count_states(con_df, tot_df)
    final_re = {}
    for att in att_char:
        census_dist = att_char[att]['probs']
        syn_dist = get_dist_syn_pop(syn_pop[att], ls_states=att_char[att]['states'])
        # Have to make the syn_dist as target cause' it may have 0, if we swap there will be err as log(0) is undefined
        final_re[att] = cross_entropy(syn_dist, census_dist)
    sort_result = sorted(final_re.items(), key=lambda item: item[1])
    assert num_att <= len(sort_result)
    return [sort_result[i][0] for i in range(num_att)]


def partition_df(df, frac=0.5):
    # parition randomly
    frac_df = df.sample(frac=frac)
    rest_df = df.drop(frac_df.index)
    return frac_df, rest_df


def mutation(indi, BN_model, con_df, tot_df, partition_rate=0.25, num_keep_atts=3, num_child=5):
    arr_solutions = []
    # find the best fit atts
    ls_best_atts = best_fit_atts(indi, con_df, tot_df, num_att=num_keep_atts)
    ls_atts = list(indi.columns)
    index_best_atts = [ls_atts.index(att) for att in ls_best_atts]

    # create children from mutation
    for _ in range(num_child):
        # partition randomly based on the ratio
        mut_part, rest_part = partition_df(indi, frac=partition_rate)
        # BN inference for the rest of them atts in mutation, this has to be rejection sample
        # create new df based on mutation part
        final_list_df = [rest_part]

        mut_part = mut_part.to_numpy() # convert for better performance
        for record in mut_part:
            # create evidence
            evidence = [State(ls_atts[i], record[i]) for i in index_best_atts]
            new_rec = sample_BN(
                model=BN_model, 
                n=1, # NOTE: can try further test of instead of having only 1, we can create more and select the best of mutation (maybe most different one?)
                typeOf='rejection',
                evidence=evidence)
            final_list_df.append(new_rec)
        # combine again
        final_child = pd.concat(final_list_df, ignore_index=True)
        arr_solutions.append(final_child)
    
    return arr_solutions


def crossover(pa1, pa2, partition_rate=0.4):
    offspring = []
    # partition randomly based on the ratio for pa1
    # partition randomly based on the ratio for pa2
    # swap
    return offspring


def eval_loop(first_gen):
    # Select the individuals for the operations (the criteria is unknown)
    # Producing offsprings (reproduction/ crossover)
    # Mutate offspring
    # Eval each of individual with the eval func
    # Select the "best" for next round (or replacement)
    NotImplemented


def EvoProg():
    # Initial solutions/ population
    # Run loop
    # Pick the final solution
    NotImplemented


data_location = "../../../Generator_data/data/data_processed_here/"


def main():
    con_df = pd.read_csv(data_location + "flat_con.csv")
    tot_df = pd.read_csv(data_location + "flat_marg.csv")
    ori_data = pd.read_csv(data_location + "flatten_seed_data.csv").astype(str)


if __name__ == "__main__":
    main()