import numpy as np
import pandas as pd

from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork

from PopSynthesis.Benchmark.checker import total_RMSE_flat, update_SRMSE
from PopSynthesis.Methods.BN.TBN.test_GA import EvoProg
from PopSynthesis.Methods.IPF.IPF import IPF_training


def BN_learning_normal(seed_data, N):
    est = HillClimbSearch(seed_data)
    best_DAG = est.estimate(scoring_method=BicScore(seed_data))
    model = BayesianNetwork(best_DAG)
    para_learn = MaximumLikelihoodEstimator(model=model, data=seed_data)
    ls_CPDs = para_learn.get_parameters()
    model.add_cpds(*ls_CPDs)
    inference = BayesianModelSampling(model)
    syn_data = inference.forward_sample(size=N)
    return syn_data


def run_loop_test_all(ori_data, con_df, tot_df, sample_range = range(1, 101, 5), num_repeat=10):
    N = len(ori_data)
    bn_re_SRMSE, evol_re_SRMSE, ipf_re_SRMSE = [], [], []
    bn_re_RMSD, evol_re_RMSD, ipf_re_RMSD = [], [], []
    for r in sample_range:
        sample_rate = r/100
        bn_SRMSE, evol_SRMSE, ipf_SRMSE = 0, 0, 0
        bn_RMSD, evol_RMSD, ipf_RMSD = 0, 0, 0
        for j in range(num_repeat):
            seed_data = ori_data.sample(frac=sample_rate)
            print(f"RATE {r} - repeat {j} - DOING EVO")
            evo_sol = EvoProg(
                seed_data=seed_data,
                con_df=con_df,
                tot_df=tot_df,
                num_pop=10,
                random_rate=0.3,
                num_gen=20,
                crossover_time=2
            )
            evol_SRMSE += update_SRMSE(evo_sol, ori_data)
            evol_RMSD += total_RMSE_flat(evo_sol, tot_df, con_df)

            print(f"RATE {r} - repeat {j} - DOING BN")
            bn_sol = BN_learning_normal(seed_data=seed_data, N=N)
            bn_SRMSE += update_SRMSE(bn_sol, ori_data)
            bn_RMSD += total_RMSE_flat(bn_sol, tot_df, con_df)

            print(f"RATE {r} - repeat {j} - DOING IPF")
            ipf_sol = IPF_training(seed_data, r)
            ipf_SRMSE += update_SRMSE(ipf_sol, ori_data)
            ipf_RMSD += total_RMSE_flat(ipf_sol, tot_df, con_df)

        print(f"RATE {r} - FINALISING...")
        evol_re_SRMSE.append(evol_SRMSE/num_repeat)
        bn_re_SRMSE.append(bn_SRMSE/num_repeat)
        ipf_re_SRMSE.append(ipf_SRMSE/num_repeat)

        evol_re_RMSD.append(evol_RMSD/num_repeat)
        bn_re_RMSD.append(bn_RMSD/num_repeat)
        ipf_re_RMSD.append(ipf_RMSD/num_repeat)
    
    print("PREP RESULTS")
    _SRMSE = [ipf_re_SRMSE, bn_re_SRMSE, evol_re_SRMSE]
    _RMSD = [ipf_re_RMSD, bn_re_RMSD, evol_re_RMSD]
    return np.array([_SRMSE, _RMSD])


def main():
    # Getting the needed data, original and census
    data_location = "Generator_data/data/data_processed_here/"
    
    ori_data = pd.read_csv(data_location + "flatten_seed_data.csv").astype(str)
    con_df = pd.read_csv(data_location + "flat_con.csv")
    tot_df = pd.read_csv(data_location + "flat_marg.csv")

    # Running sample, maybe for 1 sample we will have multiple and avg out
    results = run_loop_test_all(
        ori_data, con_df, tot_df, 
        sample_range=range(1, 10)
    )
    np.save('Final_results_1_10', results)
    

if __name__ == "__main__":
    main()
