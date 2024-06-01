"""
The main file to do the work of generating using IPF-based
"""
from paras import loc_data, loc_output
from PopSynthesis.Methods.IPF.src.IPF import eval_based_on_full_pop, simple_synthesize_all_zones
from PopSynthesis.Methods.IPF.src.data_process import simple_load_data
import numpy as np
import os
import pathlib


def main():
    min_rate, max_rate, tot = 0.1, 1, 10
    results, results_rmse = eval_based_on_full_pop(loc_data=loc_data, range_sample=np.linspace(min_rate, max_rate, tot))
    np.save(f'./output/result_IPF_{min_rate}_{max_rate}.npy', np.asarray(results))
    print(results)
    np.save(f'./output/result_rmse_IPF_{min_rate}_{max_rate}.npy', np.asarray(results_rmse))
    print(results_rmse)


def test_new_only_hh():
    name_f = lambda x: os.path.join(pathlib.Path(__file__).parent.resolve(), loc_data, f'{x}.csv')
    hh_marg_file = name_f("hh_marginals_ipu")
    hh_sample_file = name_f("hh_sample_ipu")
    hh_marg, hh_sample, xwalks = simple_load_data(hh_marg_file, hh_sample_file)
    synthetic_hh = simple_synthesize_all_zones(hh_marg, hh_sample, xwalks)
    print(synthetic_hh)
    synthetic_hh.to_csv(os.path.join(pathlib.Path(__file__).parent.resolve(), loc_output, "IPF_re_HH_only.csv"), index=False)

if __name__ == "__main__":
    # main()
    test_new_only_hh()
