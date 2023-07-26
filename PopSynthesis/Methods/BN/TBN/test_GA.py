import math
import time
import random

import pandas as pd
import numpy as np
import networkx as nx
import pylab as plt

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

def learn_para_BN_dirichlet(model, data_df, state_names, prior_counts):
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
    model = learn_para_BN_dirichlet(model, data_df, state_names, prior_counts)
    
    return model


def sample_BN(model, n, typeOf='forward', evidence=None):
    inference = BayesianModelSampling(model)
    syn = None
    if typeOf == 'forward':
        if evidence: print("Using forward sampling, the evidence you provided would not have any effect")
        syn = inference.forward_sample(size=n)
    elif typeOf == 'rejection':
        syn = inference.rejection_sample(evidence=evidence, size=n, show_progress=False)
    else:
        print("ERRORRRRR, not sure what type of sampling is that")
    return syn


def eval_func(indi, con_df, tot_df):
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

    # create children from mutation
    for _ in range(num_child):
        # partition randomly based on the ratio
        mut_part, rest_part = partition_df(indi, frac=partition_rate)
        # BN inference for the rest of them atts in mutation, this has to be rejection sample
        # create new df based on mutation part
        final_list_df = [rest_part]

        count_combine_best_att = mut_part.groupby(ls_best_atts)[ls_atts[0]].count()
        for state_combine in count_combine_best_att.index:
            print(state_combine)
            # create evidence
            evidence = [State(att, state) for att, state in zip(ls_best_atts,state_combine)]
            num_to_sample = count_combine_best_att[state_combine]
            new_rec = sample_BN(
                model=BN_model, 
                n=num_to_sample, # NOTE: can try further test of instead of having only 1, we can create more and select the best of mutation (maybe most different one?)
                typeOf='rejection',
                evidence=evidence)
            final_list_df.append(new_rec)
        # combine again
        final_child = pd.concat(final_list_df, ignore_index=True)
        arr_solutions.append(final_child)
    
    return arr_solutions


def crossover(pa1, pa2, partition_rate=0.4):
    # partition randomly based on the ratio for pa1
    swap_pa1, keep_pa1 = partition_df(pa1, frac=partition_rate)
    # partition randomly based on the ratio for pa2
    swap_pa2, keep_pa2 = partition_df(pa2, frac=partition_rate)
    # swap
    offspring1 = pd.concat([keep_pa1, swap_pa2], ignore_index=True)
    offspring2 = pd.concat([keep_pa2, swap_pa1], ignore_index=True)
    return [offspring1, offspring2]


def eval_ls_solutions(ls_sol, con_df, tot_df, eval_func, n=1):
    assert n <= len(ls_sol)
    # Should return the list of best solutions
    check = []
    for sol in ls_sol:
        re = eval_func(
            indi=sol,
            con_df=con_df,
            tot_df=tot_df
        )
        check.append((sol, re))
    sort_result = sorted(check, key=lambda indi: indi[1])
    return [sort_result[i][0] for i in range(n)]


def EvoProg_check_loop(seed_data, ori_data, con_df, tot_df, num_pop=10, random_rate=0.2, num_gen=1000, err_converg=math.inf, crossover_time=3):
    assert random_rate < 1
    num_random = max(int(num_pop * random_rate), 1) if random_rate > 0 else 0
    num_best = num_pop - num_random
    # Initial solutions/ population
    N = tot_df['total'].iloc[0]
    model = learn_BN_diriclet(seed_data, con_df, tot_df) # NOTE: this model is quite good as it does incorporate census data

    # This is because I want to test only para update
    state_names = get_state_names(con_df)
    prior_counts, prior_cpds = get_prior(model, con_df, tot_df)

    initial_pop = sample_BN(model, n=N)

    ###### TEST
    nx.draw_circular(model ,with_labels=True)
    plt.show()
    
    check_RMSD=[] 
    check_SRMSE=[]
    ######## TEST

    # Run loop
    solutions = [initial_pop]
    counter = 0
    err_score = math.inf
    while counter < num_gen and err_score >= err_converg:
        print(f"RUNNING FOR GEN {counter}")

        #TODO: defo can optimise the work on eval solution, will work on it later

        # pick the best solution
        best_sol = eval_ls_solutions(solutions, con_df, tot_df, n=len(solutions))

        ######### TEST
        test_score = eval_func(best_sol[0], con_df=con_df, tot_df=tot_df)
        print("best at the moment", test_score)
        check_RMSD.append(test_score)
        check_SRMSE.append(update_SRMSE(ori_data, best_sol[0]))
        ########## TEST

        # Mutate offspring, mutate all using the BN of the best, best one will get mutate more
        for i in range(len(solutions)):
            mutation_offsp = mutation(
                indi=best_sol[i],
                BN_model=model,
                con_df=con_df,
                tot_df=tot_df,
                partition_rate=0.2,
                num_keep_atts=int(len(state_names)/3), # A more robust way/ dynamic to declare this
                num_child=(num_pop-i) # this is to make sure that the population size is correct
            )
            solutions.extend(mutation_offsp)

        # Producing offsprings (reproduction/ crossover)
        for _ in range(crossover_time):
            best_pa_sol = eval_ls_solutions(solutions, con_df, tot_df, n=2)
            cross_offsp = crossover(
                pa1=best_pa_sol[0],
                pa2=best_pa_sol[1],
                partition_rate=0.4
            )
            solutions.extend(cross_offsp)
        # Select the "best" for next round (or replacement)
        sorted_solutions = eval_ls_solutions(solutions, con_df, tot_df, n=len(solutions))

        # Having some random solutions from the worst to increase diversity
        worst_solutions = sorted_solutions[num_best:]
        random_solutions = random.sample(worst_solutions, k=num_random)

        solutions = sorted_solutions[:num_best]
        solutions.extend(random_solutions)
        
        # select the "best" only 1 for BN learning
        model = learn_para_BN_dirichlet(model, sorted_solutions[0], state_names, prior_counts)
        counter += 1
    # Pick the final solution, can create BN as well
    result = eval_ls_solutions(solutions, con_df, tot_df)[0]
    nx.draw_circular(model ,with_labels=True)
    plt.show()

    ###### TEST
    print(check_RMSD) 
    print(check_SRMSE)
    np.save('Testing/GA_results_RMSD_2', np.array(check_RMSD))
    np.save('Testing/GA_results_SRMSE_2', np.array(check_SRMSE))
    ####### TEST
    
    return result


def EvoProg(seed_data, con_df, tot_df, num_pop=10, random_rate=0.2, num_gen=1000, err_converg=math.inf, crossover_time=3):
    assert random_rate < 1
    num_random = max(int(num_pop * random_rate), 1) if random_rate > 0 else 0
    num_best = num_pop - num_random
    # Initial solutions/ population
    N = tot_df['total'].iloc[0]
    model = learn_BN_diriclet(seed_data, con_df, tot_df) # NOTE: this model is quite good as it does incorporate census data

    # This is because I want to test only para update
    state_names = get_state_names(con_df)
    prior_counts, prior_cpds = get_prior(model, con_df, tot_df)

    initial_pop = sample_BN(model, n=N)

    # Run loop
    solutions = [initial_pop]
    counter = 0
    err_score = math.inf
    while counter < num_gen and err_score >= err_converg:
        print(f"RUNNING FOR GEN {counter}")

        #TODO: defo can optimise the work on eval solution, will work on it later

        # pick the best solution
        best_sol = eval_ls_solutions(solutions, con_df, tot_df, n=len(solutions))

        # Mutate offspring, mutate all using the BN of the best, best one will get mutate more
        print(f"GA - gen {counter}: mutation")
        for i in range(len(solutions)):
            mutation_offsp = mutation(
                indi=best_sol[i],
                BN_model=model,
                con_df=con_df,
                tot_df=tot_df,
                partition_rate=0.2,
                num_keep_atts=int(len(state_names)/3), # A more robust way/ dynamic to declare this
                num_child=(num_pop-i) # this is to make sure that the population size is correct
            )
            solutions.extend(mutation_offsp)

        # Producing offsprings (reproduction/ crossover)
        print(f"GA - gen {counter}: crossover")
        for _ in range(crossover_time):
            best_pa_sol = eval_ls_solutions(solutions, con_df, tot_df, n=2)
            cross_offsp = crossover(
                pa1=best_pa_sol[0],
                pa2=best_pa_sol[1],
                partition_rate=0.4
            )
            solutions.extend(cross_offsp)
        # Select the "best" for next round (or replacement)
        sorted_solutions = eval_ls_solutions(solutions, con_df, tot_df, n=len(solutions))

        print(f"GA - gen {counter}: selection")
        # Having some random solutions from the worst to increase diversity
        worst_solutions = sorted_solutions[num_best:]
        random_solutions = random.sample(worst_solutions, k=num_random)

        solutions = sorted_solutions[:num_best]
        solutions.extend(random_solutions)
        
        # select the "best" only 1 for BN learning
        model = learn_para_BN_dirichlet(model, sorted_solutions[0], state_names, prior_counts)
        counter += 1
    # Pick the final solution, can create BN as well
    result = eval_ls_solutions(solutions, con_df, tot_df)[0]
    
    return result


data_location = "../../../Generator_data/data/data_processed_here/"


def main():
    ori_data = pd.read_csv(data_location + "flatten_seed_data.csv").astype(str)
    con_df = pd.read_csv(data_location + "flat_con.csv")
    tot_df = pd.read_csv(data_location + "flat_marg.csv")
    seed_data = ori_data.sample(n=1000, ignore_index=True)
    start = time.time()
    final_pop = EvoProg_check_loop(seed_data, ori_data, con_df, tot_df, num_gen=20, random_rate=0.3)
    end = time.time()
    print("elapsed time in second", end - start)
    final_pop.to_csv("GA.csv", index=False)


if __name__ == "__main__":
    main()