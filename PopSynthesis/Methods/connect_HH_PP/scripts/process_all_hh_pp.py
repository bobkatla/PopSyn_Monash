"""
This will process from the begining till ends (probs including process_data)
We need to put the HH process and PP process in one place as we need to update the pool of the HH
after the deleted through cross check with others BN
We need to sample again with the missing one

AFTER THIS IS done (compared with IPU) then we can proceed with the updating to have perfect HH and perfect PP
"""

import pandas as pd
import numpy as np
import pickle
import os

from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator

from PopSynthesis.Methods.connect_HH_PP.paras_dir import processed_data, geo_lev, output_dir
from PopSynthesis.Methods.connect_HH_PP.scripts.const import *
from PopSynthesis.Methods.connect_HH_PP.scripts.sample_hh_main import *
from PopSynthesis.Methods.connect_HH_PP.scripts.sample_pp import *


# Process to have HH and Main Persons from the HH-Main pool
# Process to have Main and Rela Persons, the results of this will update the HH again

pool_sz = int(1e6) # 10 Mils


def get_hh_main_df(pool, marg_hh=None):
    if marg_hh is None:
        df_marg_hh = pd.read_csv(os.path.join(data_dir, "hh_marginals_ipu.csv"), header=[0,1])
    else:
        df_marg_hh = marg_hh
    final_syn_hh_main = samp_from_pool_1layer(pool, df_marg_hh, "totalvehs", geo_lev)
    return final_syn_hh_main


def get_pool(df_seed, state_names):
    print("Learn BN")
    model = learn_struct_BN_score(df_seed, show_struct=False, state_names=state_names)
    model = learn_para_BN(model, df_seed)
    print("Doing the sampling")
    inference = BayesianModelSampling(model)
    pool = inference.forward_sample(size=pool_sz, show_progress=True)
    pool = filter_pool(pool)
    return pool


def main():
    pp_state_names = None
    with open(os.path.join(processed_data, 'dict_pp_states.pickle'), 'rb') as handle:
        pp_state_names = pickle.load(handle)
    hh_state_names = None
    with open(os.path.join(processed_data, 'dict_hh_states.pickle'), 'rb') as handle:
        hh_state_names = pickle.load(handle)
    state_names = hh_state_names | pp_state_names

    #learning to get the HH only with main person
    df_seed = pd.read_csv(os.path.join(processed_data, "connect_hh_main.csv"))
    # drop all the ids as they are not needed for in BN learning
    id_cols = [x for x in df_seed.columns if "hhid" in x or "persid" in x]
    df_seed = df_seed.drop(columns=id_cols)
    pool_hh_main = get_pool(df_seed, state_names)

    all_rela_exist = ALL_RELA.copy()
    all_rela_exist.remove("Self")
    dict_model_inference = inference_model_get(all_rela_exist, pp_state_names)
    dict_pool_sample = pools_get(all_rela_exist, dict_model_inference, pool_sz)

    ls_final_hh = []
    ls_final_pp = []

    check = np.inf
    i = 0
    marg_hh = None
    re_check_to_show = []
    while check > 100:
        print(f"DOING ITE {i} with err == {check}")
        combine_df_hh_main = get_hh_main_df(pool_hh_main, marg_hh)
        # combine_df_hh_main = pd.read_csv(os.path.join(processed_data, f"SynPop_hh_main_{geo_lev}.csv"))
        # final_syn.to_csv(os.path.join(processed_data, f"SynPop_hh_main_{geo_lev}.csv"), index=False)

        # Process the HH and main to have the HH with IDs and People in HH
        hh_df, main_pp_df_all = process_combine_df(combine_df_hh_main)

        store_pp_df = extra_pp_df(main_pp_df_all)
        ls_df_pp = [store_pp_df]

        del_df = []
        for rela in all_rela_exist:
            to_del_df, pop_rela = process_rela_fast(main_pp_df_all, rela, dict_pool_sample[rela])
            cols_main = [f"{x}_main" for x in PP_ATTS if x not in["relationship", "persid", "hhid", geo_lev]]
            rename_cols = {f"{name}_{rela}": name for name in PP_ATTS if name not in["relationship", "persid", "hhid", geo_lev]}
            pop_rela = pop_rela.drop(columns=cols_main)
            pop_rela = pop_rela.rename(columns=rename_cols)
            ls_df_pp.append(pop_rela)
            del_df.append(to_del_df)

        del_df_final = pd.concat(del_df)
        all_df_pp = pd.concat(ls_df_pp)
        all_df_pp_rm = all_df_pp[~all_df_pp["hhid"].isin(del_df_final["hhid"])] # remove those del
        hh_df_rm = hh_df[~hh_df["hhid"].isin(del_df_final["hhid"])]

        ls_final_hh.append(hh_df_rm)
        ls_final_pp.append(all_df_pp_rm)

        # Update the pool
        get_comb = del_df_final.drop(columns=["hhid", "POA"]).value_counts()
        pool_del_index = []
        for comb in get_comb.index:
            q = ""
            idx_c = 0
            for att, state in zip(get_comb.index.names, comb):
                if idx_c != 0:
                    q += " & "
                q += f"{att} == '{state}'"
                idx_c += 1
            pool_del_index += list(pool_hh_main.query(q).index)
        pool_hh_main = pool_hh_main.loc[~pool_hh_main.index.isin(pool_del_index)]
        # Get the new marg to handle the new df
        del_hh = hh_df[hh_df["hhid"].isin(del_df_final["hhid"])]
        marg_new_raw = del_hh.groupby('POA')['totalvehs'].value_counts().unstack().fillna(0)
        convert_marg_dict = {(marg_new_raw.columns.name, state): marg_new_raw[state] for state in marg_new_raw.columns}
        convert_marg_dict[("zone_id", None)] = marg_new_raw.index
        marg_hh = pd.DataFrame(convert_marg_dict)
    
        check = len(del_df_final)
        re_check_to_show.append(check)
        i += 1

    # Process to combine final results of hh and df, mainly change id
    print(f"DOING processing hhid after {i} ite")
    new_ls_hh = []
    new_ls_pp = []
    max_id = None
    for hh, pp in zip(ls_final_hh, ls_final_pp):
        if max_id is not None:
            hh["hhid"] = hh["hhid"] + max_id
            pp["hhid"] = pp["hhid"] + max_id
        max_id = int(max(hh["hhid"]))
        new_ls_hh.append(hh)
        new_ls_pp.append(pp)
    
    final_hh = pd.concat(new_ls_hh)
    final_pp = pd.concat(new_ls_pp)

    # Outputing
    final_pp.to_csv(os.path.join(output_dir, f"syn_pp_final_{geo_lev}.csv"), index=False)
    final_hh.to_csv(os.path.join(output_dir, f"syn_hh_final_{geo_lev}.csv"), index=False)

    print(re_check_to_show)


if __name__ == "__main__":
    main()