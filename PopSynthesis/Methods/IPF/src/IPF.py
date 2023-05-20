import numpy as np
import pandas as pd
import synthpop.ipf.ipf as ipf
from PopSynthesis.Methods.IPF.src.data_process import process_data, get_test_data


def IPF_sampling(constraints):
    # constraints.to_csv('./Joint_dist_result_IPF.csv')
    ls = None
    for i in constraints.index:
        if constraints[i]: 
            # TODO: instead of just rounding like this, use the papers method of int rounding
            ls_repeat = np.repeat([i], int(constraints[i]), axis=0)
            if ls is None: ls = ls_repeat
            else: ls = np.concatenate((ls, ls_repeat), axis=0)
    return pd.DataFrame(ls, columns=constraints.index.names)


if __name__ == "__main__":
    hh, pp, con, census_sa3 = get_test_data()
    a = process_data(pp, census_sa3, "SA3", con, False)
    for b in a:
        c = a[b]
        print(c["seed"])  
        print(c["census"])        
        constraints, iterations = ipf.calculate_constraints(c["census"], c["seed"], tolerance=1e-5)
        print(constraints)
        print(IPF_sampling(constraints))
        break
