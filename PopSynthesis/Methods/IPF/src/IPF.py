import numpy as np
import pandas as pd
import synthpop.ipf.ipf as ipf
from PopSynthesis.Methods.IPF.src.data_process import process_data, get_test_data


def IPF_sampling(constraints, rounding=None):
    # constraints.to_csv('./Joint_dist_result_IPF.csv')
    ls = None
    for i in constraints.index:
        if constraints[i]: 
            # TODO: instead of just rounding like this, use the papers method of int rounding
            ls_repeat = np.repeat([i], int(constraints[i]), axis=0)
            if ls is None: ls = ls_repeat
            else: ls = np.concatenate((ls, ls_repeat), axis=0)
    return pd.DataFrame(ls, columns=constraints.index.names)


def IPF_all(seed, census, zone_lev, con, hh=True, tolerence=1e-5):
    dict_zones = process_data(
        seed=seed,
        census=census,
        zone_lev=zone_lev,
        control=con,
        hh=hh
    )
    ls_df = []
    for zone in dict_zones:
        zone_details = dict_zones[zone]      
        constraints, iterations = ipf.calculate_constraints(zone_details["census"], zone_details["seed"], tolerance=tolerence)
        result = IPF_sampling(constraints)
        result[zone_lev] = zone
        ls_df.append(result)
    synthetic_population = pd.concat(ls_df)

    return synthetic_population


if __name__ == "__main__":
    hh, pp, con, census_sa3 = get_test_data()
    fin = IPF_all(
        hh, 
        census_sa3,
        "SA3",
        con,
        True
    )
    print(fin["totalvehs"].value_counts())
    
