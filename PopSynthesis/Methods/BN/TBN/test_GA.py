import pandas as pd
import math

from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork

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


def sample_BN(model, n, typeOf='forward'):
    inference = BayesianModelSampling(model)
    syn = None
    if typeOf == 'forward':
        syn = inference.forward_sample(size=n)
    else:
        print("ERRORRRRR")
    return syn


def eval_func(indi, tot_df, con_df):
    score = total_RMSE_flat(indi, tot_df, con_df)
    return score


def cross_entropy(dist_target, dist_check):
    # H(target, approx/check)
    # H(P, Q) = – sum x in X P(x) * log(Q(x))

    assert len(dist_target) == len(dist_check)

    # They will be 2 same size array for the same atts, 
    # The order matters, assuming that for the same position is the same event/state

    result = 0
    for i in range(len(dist_target)):
        result += dist_target[i] * math.log(dist_check[i], 2) # in bits, change to e if want nats
    return -result


def mutation(indi, BN_model, partition_rate=0.25, num_keep_atts=3, num_child=5):
    arr_solutions = []
    # find the best fit atts
    for _ in range(num_child):
        # partition randomly based on the ratio
        # BN inference for the rest of them atts in mutation
        # combine again
        arr_solutions.append(None)
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

    a = cal_count_states(con_df, tot_df)
    target = a['SEX']['probs']
    check = [0.3, 0.7]
    print(cross_entropy(target, check))


if __name__ == "__main__":
    main()