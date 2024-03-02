import pandas as pd
import numpy as np
import os
import random
from itertools import chain
from PopSynthesis.Methods.connect_HH_PP.paras_dir import data_dir, processed_data, output_dir
from PopSynthesis.Methods.connect_HH_PP.scripts.adjust_atts_by_counts import *
from PopSynthesis.Methods.connect_HH_PP.scripts.sample_hh_main import *
from PopSynthesis.Methods.connect_HH_PP.scripts.process_all_hh_pp import *
from PopSynthesis.Methods.connect_HH_PP.scripts.get_hh_main_combine import *
import pickle


def segment_df(df, chunk_sz) -> list[pd.DataFrame]:
    start = 0
    ls_df = []
    while start < len(df):
        sub_df = df.iloc[start:start+chunk_sz]
        ls_df.append(sub_df)
        start += chunk_sz
    return ls_df



def get_the_noad_hh(combine_df, df_seed, hh_state_names):
    geo_lev = [x for x in combine_df.columns if x != "count"][0]
    check = combine_df["count"].sum()
    ls_df = []
    while check > 0:
        print(check)
        sample = get_pool(df_seed, hh_state_names, pool_sz=check, special=False)
        ls_df.append(sample)
        check -= len(sample)
    fin_no_ad = pd.concat(ls_df)
    combine_df["ls_zones"] = combine_df.apply(lambda r: [r[geo_lev]] * r["count"], axis=1)
    ls_vals_zone = list(chain.from_iterable(combine_df["ls_zones"]))
    random.shuffle(ls_vals_zone)
    fin_no_ad[geo_lev] = ls_vals_zone
    return fin_no_ad


def main():
    geo_lev = "POA"

    df_seed = pd.read_csv(os.path.join(processed_data, "ori_sample_hh.csv"))
    # drop all the ids as they are not needed for in BN learning
    id_cols = [x for x in df_seed.columns if "hhid" in x or "persid" in x]
    df_seed = df_seed.drop(columns=id_cols)

    with open(os.path.join(processed_data, 'dict_hh_states.pickle'), 'rb') as handle:
        hh_state_names = pickle.load(handle)


    census_data = pd.read_csv(os.path.join(data_dir, "hh_marginals_ipu.csv"), header=[0,1])
    # Sample without adjustment
    tot = census_data[census_data.columns[census_data.columns.get_level_values(0)=="hhsize"]].sum(axis=1)
    ls_zones = census_data[census_data.columns[census_data.columns.get_level_values(0)=="zone_id"]]
    combine_df = pd.concat([tot, ls_zones], axis=1)
    combine_df.columns = ["count", geo_lev]

    all_rela_exist = ALL_RELA.copy()
    all_rela_exist.remove("Self")

    ls_final_hh = []
    ls_final_pp = []

    check = np.inf
    i = 0
    re_check_to_show = []

    # Only importing now
    with open(os.path.join(processed_data, 'dict_pool_sample.pickle'), 'rb') as handle:
        dict_pool_sample = pickle.load(handle)

    ls_final_hh = []
    ls_final_pp = []

    while check > 5:
        print(f"DOING ITE {i} with err == {check}")
        hh_df = get_the_noad_hh(combine_df, df_seed, hh_state_names)
        if "hhid" not in hh_df.columns:
            hh_df["hhid"] = hh_df.index
        
        # Sample hh main
        logger.info("GETTING the main people")
        # To reduce the memory issue, we need to segment the hh_df, the whole point of this is just assigning anw
        ls_to_gb = [x for x in hh_df.columns if x != "hhid"]
        hh_df = hh_df.astype(str)
        hh_df.to_csv(os.path.join(processed_data, "keep_check", "first_hh.csv"), index=False)
        # count_hh_df = hh_df.groupby(ls_to_gb)["hhid"].apply(lambda x: list(x)).reset_index()
        # print(len(count_hh_df))
        ls_sub_df = segment_df(hh_df, chunk_sz=100000)
        # ls_sub_df[0].to_csv("to_test_combine.csv", index=False)

        _ls_df_com, _init_del = [], []
        for sub_df in ls_sub_df:
            _df_com, _del_sub = get_combine_df(sub_df, dict_pool_sample["Main"].value_counts().reset_index())
            if len(_df_com) > 0: _ls_df_com.append(_df_com)
            if len(_del_sub) > 0: _init_del.append(_del_sub)

        combine_df_hh_main = pd.concat(_ls_df_com)
        del_hh = pd.concat(_init_del)
        # Process the HH and main to have the HH with IDs and People in HH
        combine_df_hh_main.to_csv(os.path.join(processed_data, "keep_check", "noad_hh_main.csv"), index=False)
        del_hh.to_csv(os.path.join(processed_data, "keep_check", "noad_init_del.csv"), index=False)
        _, main_pp_df_all = process_combine_df(combine_df_hh_main)

        store_pp_df = extra_pp_df(main_pp_df_all)
        ls_df_pp = [store_pp_df]

        del_df = []
        main_pp_df_all[all_rela_exist + ["hhid"]] = main_pp_df_all[all_rela_exist + ["hhid"]].astype(int)
        for rela in all_rela_exist:
            logger.info(f"Doing {rela} now lah~")
            to_del_df, pop_rela = process_rela_fast(main_pp_df_all, rela, dict_pool_sample[rela].copy()) # fix this

            if len(pop_rela) > 0:
                dict_hhid = dict(zip(hh_df["hhid"], hh_df[geo_lev]))
                pop_rela[geo_lev] = pop_rela.apply(lambda r: dict_hhid[int(r["hhid"])],axis=1)
                cols_main = [f"{x}_main" for x in PP_ATTS if x not in["relationship", "persid", "hhid", geo_lev]]
                rename_cols = {f"{name}_{rela}": name for name in PP_ATTS if name not in["relationship", "persid", "hhid", geo_lev]}
                pop_rela = pop_rela.drop(columns=cols_main)
                pop_rela = pop_rela.rename(columns=rename_cols)
                ls_df_pp.append(pop_rela)
            if to_del_df is not None:
                del_df.append(to_del_df)

        if len(ls_df_pp) == 0:
            raise ValueError("Some reason there are none to concat for pp df")
        all_df_pp = pd.concat(ls_df_pp)
        
        if len(del_df) == 0:
            ls_final_hh.append(hh_df)
            ls_final_pp.append(all_df_pp)
            re_check_to_show.append(0)
            break

        del_df_final = pd.concat(del_df)
        ls_del_id = list(del_df_final["hhid"].astype(str)) + list(del_hh["hhid"].astype(str))
        hh_df["hhid"] = hh_df["hhid"].astype(str)
        
        hh_df_keep = hh_df[~hh_df["hhid"].isin(ls_del_id)]
        hh_df_got_rm = hh_df[hh_df["hhid"].isin(ls_del_id)]
        
        ls_final_hh.append(hh_df_keep)
        ls_final_pp.append(all_df_pp)

        # Checking to get the new combine
        count_poa_rm = hh_df_got_rm[geo_lev].value_counts()
        combine_df = count_poa_rm.reset_index()
        print(combine_df)

        # Finish
        check = len(hh_df_got_rm)
        re_check_to_show.append(check)
        i += 1
        

    # Process to combine final results of hh and df, mainly change id
    print(f"DOING processing hhid after {i} ite")
    new_ls_hh = []
    new_ls_pp = []
    max_id = 1
    for hh, pp in zip(ls_final_hh, ls_final_pp):
        if max_id is not None:
            hh["hhid"] = hh["hhid"] + max_id
            pp["hhid"] = pp["hhid"] + max_id
        max_id = int(max(hh["hhid"])) + 1
        new_ls_hh.append(hh)
        new_ls_pp.append(pp)
    
    final_hh = pd.concat(new_ls_hh)
    final_pp = pd.concat(new_ls_pp)

    # Outputing
    final_pp.to_csv(os.path.join(output_dir, f"hh_no_adjustments.csv"), index=False)
    final_hh.to_csv(os.path.join(output_dir, f"pp_no_adjustments.csv"), index=False)

    print(re_check_to_show)


if __name__ == "__main__":
    main()