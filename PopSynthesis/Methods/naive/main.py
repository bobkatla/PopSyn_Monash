"""
simple based on the weights of the seed data and sample from it
"""
import pandas as pd
import numpy as np
from PopSynthesis.Benchmark.CompareFullPop.utils import sampling_from_full_pop, realise_full_pop_based_on_weight, condense_pop
from PopSynthesis.Methods.IPF.src.data_process import get_marg_val_from_full
from PopSynthesis.Benchmark.CompareFullPop.compare import SRMSE_based_on_counts
from PopSynthesis.Benchmark.CompareCensus.compare import compare_RMS_census


def eval_based_on_full_pop(loc_data, range_sample=np.linspace(0.01, 0.1, 10)):
    seed_df_hh = pd.read_csv(loc_data + "H_sample.csv")
    seed_df_pp = pd.read_csv(loc_data + "P_sample.csv")
    
    # fake_HH_seed_data, fake_PP_seed_data = wrapper_get_all(seed_df_hh, seed_df_pp, sample_rate=0.01, name_weights_in_hh="wdhhwgt_sa3", new_name_weights_in_hh='_weight', shared_ids_name='hhid')
    
    # NOTE: for now, only HH data
    to_drop_cols = ["hh_num", "hhid", "SA1", "SA2", "SA3", "SA4"]
    seed_df_hh = seed_df_hh.drop(columns=to_drop_cols)
    full_df_hh = realise_full_pop_based_on_weight(seed_df_hh, weight_col="wdhhwgt_sa3")
    n = len(full_df_hh)
    results = []
    results_rmse = []
    marginals = get_marg_val_from_full(full_df_hh)
    for rate in range_sample:
        print(f"PROCESSING rate {rate}")
        seed_df = sampling_from_full_pop(full_df_hh, rate=1) # shuffle the data
        seed_df = sampling_from_full_pop(seed_df, rate=1) # shuffle the data
        seed_df = seed_df.head(int(rate * n))
        new_seed_df = condense_pop(seed_df, "_weight")
        print("Doing the sample now")
        # Rounding the contraints
        syn_pop = new_seed_df.sample(n=n, weights="_weight", replace=True)
        syn_pop = syn_pop.drop(columns="_weight")
        print("Calculate SRMSE now")
        SRMSE = SRMSE_based_on_counts(full_df_hh.value_counts(), syn_pop.value_counts())
        results.append(SRMSE)
        results_rmse.append(compare_RMS_census(marginals, get_marg_val_from_full(syn_pop)))
        print(f"Done rate {rate} with score of {SRMSE}")
    return results, results_rmse


def main():
    loc_data = "./data/"
    min_rate, max_rate, tot = 0.1, 1, 10
    results, results_rmse = eval_based_on_full_pop(loc_data=loc_data, range_sample=np.linspace(min_rate, max_rate, tot))
    np.save(f'./output/result_naive_{min_rate}_{max_rate}.npy', np.asarray(results))
    print(results)
    np.save(f'./output/result_rmse_naive_{min_rate}_{max_rate}.npy', np.asarray(results_rmse))
    print(results_rmse)


if __name__ == "__main__":
    main()