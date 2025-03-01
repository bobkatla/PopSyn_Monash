"""
The main file to do the work of generating using IPF-based
"""
from paras import loc_data, loc_output
from PopSynthesis.Methods.IPF.src.IPF import (
    eval_based_on_full_pop,
    simple_synthesize_all_zones,
)
from PopSynthesis.Methods.IPF.src.data_process import simple_load_data
import pandas as pd
import numpy as np
import os
import pathlib
import time

def test_IPF_on_full_pop():
    min_rate, max_rate, tot = 0.1, 1, 10
    results, results_rmse = eval_based_on_full_pop(
        loc_data=loc_data, range_sample=np.linspace(min_rate, max_rate, tot)
    )
    np.save(f"./output/result_IPF_{min_rate}_{max_rate}.npy", np.asarray(results))
    print(results)
    np.save(
        f"./output/result_rmse_IPF_{min_rate}_{max_rate}.npy", np.asarray(results_rmse)
    )
    print(results_rmse)


def run_IPF(marg_file, sample_file):
    marg, samples, xwalks = simple_load_data(marg_file, sample_file)
    synthetic_agents = simple_synthesize_all_zones(marg, samples, xwalks)
    return synthetic_agents


def calculate_expected_sum(marg_file):
    marg = pd.read_csv(marg_file, header=[0,1])
    n_atts = len(set(marg.columns.get_level_values(0)) - {"zone_id", "sample_geog"})
    marg.columns = marg.columns.to_flat_index()
    cols_to_sum = [col for col in marg.columns if col[0] not in ["zone_id", "sample_geog"]]
    sum_tot = marg[cols_to_sum].sum().sum() / n_atts
    return int(sum_tot)


def main():
    name_f = lambda x: os.path.join(
        pathlib.Path(__file__).parent.resolve(), loc_data, f"{x}.csv"
    )
    hh_marg_file = name_f("hh_marginals_ipu")
    # expected sum
    expected_sum = calculate_expected_sum(hh_marg_file)
    hh_sample_file = name_f("HH_pool")
    # time the IPF
    start_time = time.time()
    synthetic_hh = run_IPF(hh_marg_file, hh_sample_file)
    end_time = time.time()
    
    synthetic_hh.to_csv(
        os.path.join(
            pathlib.Path(__file__).parent.resolve(), loc_output, "IPF_using_BN_pool.csv"
        ),
        index=False,
    )
    
    print(f"Time taken: {end_time - start_time} seconds")
    assert expected_sum == len(synthetic_hh)

if __name__ == "__main__":
    main()
