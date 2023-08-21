import numpy as np
import pandas as pd
import synthpop.ipf.ipf as ipf
from PopSynthesis.Methods.IPF.src.data_process import process_data, get_test_data, get_marg_val_from_full, get_joint_dist_from_sample
from PopSynthesis.Benchmark.CompareFullPop.utils import wrapper_get_all, sampling_from_full_pop, realise_full_pop_based_on_weight
from PopSynthesis.Benchmark.CompareFullPop.compare import full_pop_SRMSE


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


def IPF_all(seed, census, zone_lev, con, hh=True, tolerence=1e-5, max_iterations=1000):
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
        constraints, iterations = ipf.calculate_constraints(zone_details["census"], zone_details["seed"], tolerance=tolerence, max_iterations=max_iterations)
        result = IPF_sampling(constraints)
        result[zone_lev] = zone
        ls_df.append(result)
        print(f"Done for {zone} with {iterations} iter")
    synthetic_population = pd.concat(ls_df)

    return synthetic_population


def eval_based_on_full_pop(loc_data, tolerance=1e-5, range_sample=np.linspace(0.01, 0.1, 10)):
    seed_df_hh = pd.read_csv(loc_data + "H_sample.csv")
    seed_df_pp = pd.read_csv(loc_data + "P_sample.csv")
    
    # fake_HH_seed_data, fake_PP_seed_data = wrapper_get_all(seed_df_hh, seed_df_pp, sample_rate=0.01, name_weights_in_hh="wdhhwgt_sa3", new_name_weights_in_hh='_weight', shared_ids_name='hhid')
    
    # NOTE: for now, only HH data
    to_drop_cols = ["hh_num", "hhid", "SA1", "SA2", "SA3", "SA4"]
    seed_df_hh = seed_df_hh.drop(columns=to_drop_cols)
    full_df_hh = realise_full_pop_based_on_weight(seed_df_hh, weight_col="wdhhwgt_sa3")
    marginals = get_marg_val_from_full(full_df_hh)
    results = []
    for rate in range_sample:
        print(f"PROCESSING rate {rate}")
        seed_df = sampling_from_full_pop(full_df_hh, rate=rate)
        joint_dist = get_joint_dist_from_sample(seed_df=seed_df, full_pop=full_df_hh)
        print("Doing the IPF now")
        constraints, iterations = ipf.calculate_constraints(marginals, joint_dist, tolerance=tolerance)
        syn_pop = IPF_sampling(constraints=constraints)
        print("Calculate SRMSE now")
        SRMSE = full_pop_SRMSE(full_df_hh, syn_pop)
        results.append(SRMSE)
        print(f"Done rate {rate} with {iterations} iters, got score of {SRMSE}")
    return results


if __name__ == "__main__":
    hh, pp, con, census_sa3 = get_test_data()
    fin = IPF_all(
        hh, 
        census_sa3,
        "SA3",
        con,
        True,
        max_iterations=100000000000000
    )
    fin.to_csv("test_hh.csv")
    print(fin)
    
