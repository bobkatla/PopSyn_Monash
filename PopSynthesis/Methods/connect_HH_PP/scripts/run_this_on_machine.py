"""
After we got the households, now we need 
"""


import pandas as pd
from PopSynthesis.Methods.connect_HH_PP.scripts.adjust_atts import *
from PopSynthesis.Methods.connect_HH_PP.scripts.sample_hh_main import *
from PopSynthesis.Methods.connect_HH_PP.scripts.process_all_hh_pp import *
from PopSynthesis.Methods.connect_HH_PP.scripts.get_hh_main_combine import *


POOL_SZ = int(1e7)


def main():
    df_seed = pd.read_csv(os.path.join(processed_data, "ori_sample_hh.csv"))
    # drop all the ids as they are not needed for in BN learning
    id_cols = [x for x in df_seed.columns if "hhid" in x or "persid" in x]
    df_seed = df_seed.drop(columns=id_cols)

    with open(os.path.join(processed_data, 'dict_hh_states.pickle'), 'rb') as handle:
        hh_state_names = pickle.load(handle)
    with open(os.path.join(processed_data, 'dict_pp_states.pickle'), 'rb') as handle:
        pp_state_names = pickle.load(handle)
    state_names = hh_state_names | pp_state_names

    pool = get_pool(df_seed, hh_state_names, POOL_SZ)
    geo_lev = "POA"
    processed_already = ["hhsize", "totalvehs", "hhinc", "dwelltype"]

    all_rela_exist = ALL_RELA.copy()
    all_rela_exist.remove("Self")
    dict_model_inference = inference_model_get(all_rela_exist, pp_state_names)
    dict_pool_sample = pools_get(all_rela_exist, dict_model_inference, POOL_SZ)
    

    ls_final_hh = []
    ls_final_pp = []

    check = np.inf
    i = 0
    marg_hh = None
    re_check_to_show = []

    # with open(os.path.join(processed_data, 'dict_pool_sample.pickle'), 'rb') as handle:
    #     dict_pool_sample = pickle.load(handle)
    # pool = pd.read_csv(os.path.join(processed_data, 'hh_pool.csv'))

    # print("SAVING incase")
    # pool.to_csv(os.path.join(processed_data, 'hh_pool.csv'))
    # with open(os.path.join(processed_data, 'dict_pool_sample.pickle'), 'wb') as handle:
    #     pickle.dump(dict_pool_sample, handle)

    while check > 10 and i < 20:
        print(f"DOING ITE {i} with err == {check}")
        if marg_hh is None:
            hh_df = pd.read_csv(os.path.join(output_dir, "adjust", "final", "saving_hh_dwelltype.csv"))
        else:
            # Simple create a new func here and get the new marg already
            hh_df = process_data_general(marg_hh, pool, geo_lev, processed_already)
        if "hhid" not in hh_df.columns:
            hh_df["hhid"] = hh_df.index
        combine_df_hh_main, del_hh = get_combine_df(hh_df)
        # Process the HH and main to have the HH with IDs and People in HH
        print("SOME EXTRA PROCESS PP")
        hh_df, main_pp_df_all = process_combine_df(combine_df_hh_main)
        main_pp_df_all[all_rela_exist] = main_pp_df_all[all_rela_exist].astype(float).astype(int)
        # Sample hh main
        print("GETTING the main people")
        del_hh.to_csv(os.path.join(processed_data, "keep_check", f"del_first_{i}.csv"), index=False)
        hh_df.to_csv(os.path.join(processed_data, "keep_check", f"testing_input_hh_{i}.csv"), index=False)
        main_pp_df_all.to_csv(os.path.join(processed_data, "keep_check", f"main_pp_df_{i}.csv"), index=False)

        store_pp_df = extra_pp_df(main_pp_df_all)
        ls_df_pp = [store_pp_df]

        del_df = [del_hh]
        for rela in all_rela_exist:
            print(f"Doing {rela} now lah~")
            to_del_df, pop_rela = process_rela_fast(main_pp_df_all, rela, dict_pool_sample[rela])
            cols_main = [f"{x}_main" for x in PP_ATTS if x not in["relationship", "persid", "hhid", geo_lev]]
            rename_cols = {f"{name}_{rela}": name for name in PP_ATTS if name not in["relationship", "persid", "hhid", geo_lev]}
            if pop_rela is not None:
                pop_rela = pop_rela.drop(columns=cols_main)
                pop_rela = pop_rela.rename(columns=rename_cols)
                ls_df_pp.append(pop_rela)
            if to_del_df is not None:
                del_df.append(to_del_df)

        all_df_pp = pd.concat(ls_df_pp)
        
        if len(del_df) == 0:
            ls_final_hh.append(hh_df)
            ls_final_pp.append(all_df_pp)
            re_check_to_show.append(0)
            break

        del_df_final = pd.concat(del_df)
        all_df_pp_rm = all_df_pp[~all_df_pp["hhid"].isin(del_df_final["hhid"])] # remove those del
        hh_df_rm = hh_df[~hh_df["hhid"].isin(del_df_final["hhid"])]
        
        ls_final_hh.append(hh_df_rm)
        ls_final_pp.append(all_df_pp_rm)

        hh_df_rm.to_csv(os.path.join(processed_data, "keep_check", f"hh_df_{i}.csv"), index=False)
        all_df_pp_rm.to_csv(os.path.join(processed_data, "keep_check", f"pp_df_{i}.csv"), index=False)

        # Update the pool
        print("Updating the pool")
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
            pool_del_index += list(pool.query(q).index)
        pool = pool.loc[~pool.index.isin(pool_del_index)]
        print("Done updating the pool")
        # Get the new marg to handle the new df
        del_hh = hh_df[hh_df["hhid"].isin(del_df_final["hhid"])]
        
        # NOTE: need to rethink the adjust att here, key point is to maintain the marg of given
        ls_temp_hold = []
        for adjust_att in processed_already:
            temp_hold = del_hh.groupby('POA')[adjust_att].value_counts().unstack().fillna(0)
            temp_hold.columns = [(temp_hold.columns.name, x) for x in temp_hold.columns]
            temp_hold = temp_hold.astype(int)
            ls_temp_hold.append(temp_hold)
        marg_new_raw = pd.concat(ls_temp_hold, axis=1)
        convert_marg_dict = {com_col: marg_new_raw[com_col] for com_col in marg_new_raw.columns}
        convert_marg_dict[("zone_id", None)] = marg_new_raw.index
        marg_hh = pd.DataFrame(convert_marg_dict)
        print(marg_hh)
    
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

    # # Outputing
    final_pp.to_csv(os.path.join(output_dir, f"syn_pp_final_{geo_lev}_ad4.csv"), index=False)
    final_hh.to_csv(os.path.join(output_dir, f"syn_hh_final_{geo_lev}_ad4.csv"), index=False)



if __name__ == "__main__":
    main()
