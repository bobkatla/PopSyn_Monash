"""
The main file to do the work of generating using IPF-based
"""
from paras import loc_data, loc_controls, loc_output
from PopSynthesis.Methods.IPF.src.IPF import eval_based_on_full_pop
import numpy as np


def main():
    min_rate, max_rate, tot = 0.1, 1, 10
    results, results_rmse = eval_based_on_full_pop(loc_data=loc_data, range_sample=np.linspace(min_rate, max_rate, tot))
    np.save(f'./output/result_IPF_{min_rate}_{max_rate}.npy', np.asarray(results))
    print(results)
    np.save(f'./output/result_rmse_IPF_{min_rate}_{max_rate}.npy', np.asarray(results_rmse))
    print(results_rmse)


if __name__ == "__main__":
    main()