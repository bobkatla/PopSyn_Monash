import numpy as np
import pandas as pd
import synthpop.ipf.ipf as ipf
from synthpop import categorizer as cat
from PopSynthesis.Methods.IPF.src.data_process import process_data, get_test_data, get_marg_val_from_full, get_joint_dist_from_sample, get_marg_from_constraints
from PopSynthesis.Benchmark.CompareFullPop.utils import sampling_from_full_pop, realise_full_pop_based_on_weight
from PopSynthesis.Benchmark.CompareFullPop.compare import SRMSE_based_on_counts
from PopSynthesis.Benchmark.CompareCensus.compare import compare_RMS_census
from PopSynthesis.Generator_data.generate_combine_census.utils import TRS


def IPF_sampling(constraints, round_basic=True, tot=None):
    # constraints.to_csv('./Joint_dist_result_IPF.csv')
    round_vals = None
    if round_basic:
        constraints = constraints.apply(lambda x: int(x))
    else:
        ls_vals = constraints.to_list()
        round_vals = TRS(ls_vals, int(tot))

    ls = None
    for count, i in enumerate(constraints.index):
        if count % int(len(constraints.index)/10) == 0:
            check = (count / len(constraints.index)) * 100
            print(f"SAMPLING got till {round(check, 2)}%")

        if constraints[i]: 
            # TODO: instead of just rounding like this, use the papers method of int rounding
            to_sample = constraints[i] if round_basic else round_vals[count]
            ls_repeat = np.repeat([i], to_sample, axis=0)
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


def eval_based_on_full_pop(loc_data, tolerance=1e-5,  max_iterations=10000, range_sample=np.linspace(0.01, 0.1, 10)):
    seed_df_hh = pd.read_csv(loc_data + "H_sample.csv")
    seed_df_pp = pd.read_csv(loc_data + "P_sample.csv")
    
    # fake_HH_seed_data, fake_PP_seed_data = wrapper_get_all(seed_df_hh, seed_df_pp, sample_rate=0.01, name_weights_in_hh="wdhhwgt_sa3", new_name_weights_in_hh='_weight', shared_ids_name='hhid')
    
    # NOTE: for now, only HH data
    to_drop_cols = ["hh_num", "hhid", "SA1", "SA2", "SA3", "SA4"]
    seed_df_hh = seed_df_hh.drop(columns=to_drop_cols)
    full_df_hh = realise_full_pop_based_on_weight(seed_df_hh, weight_col="wdhhwgt_sa3")
    N = len(full_df_hh)
    marginals = get_marg_val_from_full(full_df_hh)
    results = []
    results_rmse = []
    for rate in range_sample:
        print(f"PROCESSING rate {rate}")
        seed_df = sampling_from_full_pop(full_df_hh, rate=1) # shuffle the data
        seed_df = sampling_from_full_pop(seed_df, rate=1) # shuffle the data
        seed_df = seed_df.head(int(rate * N))
        joint_dist = get_joint_dist_from_sample(seed_df=seed_df, full_pop=full_df_hh)
        print("Doing the IPF now")
        constraints, iterations = ipf.calculate_constraints(marginals, joint_dist, tolerance=tolerance,  max_iterations=max_iterations)
        # Rounding the contraints
        constraints = constraints.round()
        # print("Doing the sampling")
        # syn_pop = IPF_sampling(constraints=constraints)
        print("Calculate SRMSE now")
        marg_syn = get_marg_from_constraints(constraints=constraints)
        SRMSE = SRMSE_based_on_counts(full_df_hh.value_counts(), constraints)
        results.append(SRMSE)
        results_rmse.append(compare_RMS_census(marginals, marg_syn=marg_syn))
        print(f"Done rate {rate} with {iterations} iters, got score of {SRMSE}")
    return results, results_rmse


def simple_synthesize(marg, joint_dist, marginal_zero_sub=.01, jd_zero_sub=.001):
    # this is the zero marginal problem
    marg = marg.replace(0, marginal_zero_sub)

    # zero cell problem
    joint_dist.frequency = joint_dist.frequency.replace(0, jd_zero_sub)

    # IPF
    constraint, iterations = ipf.calculate_constraints(marg, joint_dist.frequency)
    tot = constraint.sum()

    idx_temp = constraint.index.names
    temp = constraint.reset_index()
    temp = temp[temp["hhinc"]!="Negative income"] # this does not make sense or we can keep this as well
    constraint = temp.set_index(idx_temp)
    constraint = constraint[0]

    # constraint.index = joint_dist.cat_id
    result = IPF_sampling(constraint, round_basic=False, tot=tot)
    return result, iterations


def simple_synthesize_all_zones(marg, samples, xwalk):
    syn_list = []
    for geogs in xwalk:
        print(f"Doing zone {geogs[0]}")
        hhs, joint_dist = cat.joint_distribution(samples[samples.sample_geog == geogs[1]], cat.category_combinations(marg.columns))
        synthetic_results, iterations = simple_synthesize(marg.loc[geogs[0]], joint_dist)
        synthetic_results["zone"] = geogs[0]
        syn_list.append(synthetic_results)
        print(f"Finished zone {geogs[0]} with {iterations} iters")
    all_re = pd.concat(syn_list)
    return all_re


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
    
