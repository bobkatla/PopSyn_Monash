"""
The main file to do the work of generating using IPF-based
"""
from paras import loc_data, loc_controls, loc_output
from PopSynthesis.Methods.IPF.src.IPF import eval_based_on_full_pop
import numpy as np


def main():
    min_rate, max_rate, tot = 0.01, 0.1, 10
    results = eval_based_on_full_pop(loc_data=loc_data, range_sample=np.linspace(min_rate, max_rate, tot))
    data = np.asarray(results)
    np.save(f'result_{min_rate}_{max_rate}.npy', data)
    print(results)


if __name__ == "__main__":
    main()