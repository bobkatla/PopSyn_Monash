"""
After we got the households, now we need 
"""


import pandas as pd
from PopSynthesis.Methods.connect_HH_PP.scripts.adjust_atts_by_counts import *
from PopSynthesis.Methods.connect_HH_PP.scripts.sample_hh_main import *
from PopSynthesis.Methods.connect_HH_PP.scripts.process_all_hh_pp import *
from PopSynthesis.Methods.connect_HH_PP.scripts.get_hh_main_combine import *
import logging
logger = logging.getLogger("connect_hh_pp")


def main():
    geo_lev = "POA"
    processed_already = ["hhsize", "totalvehs", "hhinc", "dwelltype"]

    all_rela_exist = ALL_RELA.copy()
    all_rela_exist.remove("Self")

    ls_final_hh = []
    ls_final_pp = []

    check = np.inf
    i = 0
    marg_hh = None
    re_check_to_show = []

    # Only importing now
    with open(os.path.join(processed_data, 'dict_pool_counts.pickle'), 'rb') as handle:
        dict_pool_sample = pickle.load(handle)
    pool = pd.read_csv(os.path.join(processed_data, "save_pools",'final_pool_count.csv'), index_col=0)
    pool = pool.astype(str)
    cols_pool = [x for x in pool.columns if x != "count"]

    while check > 10 and i < 20:
        logger.info(f"DOING ITE {i} with err == {check}")
        if marg_hh is None:
            hh_df = pd.read_csv(os.path.join(output_dir, "adjust", "final", "saving_hh_dwelltype.csv"), low_memory=False)
        else:
            raise ValueError("No this is GOOD, you got here")
            # Simple create a new func here and get the new marg already
            hh_df = process_data_general(marg_hh, pool, geo_lev, processed_already) # fix this
        
        if "hhid" not in hh_df.columns:
            hh_df["hhid"] = hh_df.index
        
        # Sample hh main
        logger.info("GETTING the main people")
        combine_df_hh_main, del_hh = get_combine_df(hh_df, dict_pool_sample["Main"].copy()) # fix this
        # Process the HH and main to have the HH with IDs and People in HH
        logger.info("SOME EXTRA PROCESS PP")
        _, main_pp_df_all = process_combine_df(combine_df_hh_main)
        main_pp_df_all[all_rela_exist] = main_pp_df_all[all_rela_exist].astype(float).astype(int)

        del_hh.to_csv(os.path.join(processed_data, "keep_check", f"del2_first_{i}.csv"), index=False)
        main_pp_df_all.to_csv(os.path.join(processed_data, "keep_check", f"main2_pp_df_{i}.csv"), index=False)

        store_pp_df = extra_pp_df(main_pp_df_all)
        ls_df_pp = [store_pp_df]

        del_df = []
        for rela in all_rela_exist:
            logger.info(f"Doing {rela} now lah~")
            to_del_df, pop_rela = process_rela_using_count(main_pp_df_all, rela, dict_pool_sample[rela].copy(), geo_lev) # fix this

            if len(pop_rela) > 0:
                ls_df_pp.append(pop_rela)
            if len(to_del_df) > 0:
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

        hh_df_got_rm.to_csv(os.path.join(processed_data, "keep_check", f"del2_df_hh_{i}.csv"), index=False)
        hh_df_keep.to_csv(os.path.join(processed_data, "keep_check", f"hh2_keep_df_{i}.csv"), index=False)
        all_df_pp.to_csv(os.path.join(processed_data, "keep_check", f"pp2_df_{i}.csv"), index=False)

        ########
        # Update the pool, fix this
        logger.info("Updating the pool")
        temp_pool = pool.set_index(cols_pool)
        temp_keep = hh_df_got_rm.set_index(cols_pool)
        pool = temp_pool[~temp_pool.index.isin(temp_keep.index)]
        pool = pool.reset_index()
        logger.info("Done updating the pool")
        # Get the new marg to handle the new df
        
        # NOTE: need to rethink the adjust att here, key point is to maintain the marg of given
        ls_temp_hold = []
        for adjust_att in processed_already:
            temp_hold = hh_df_got_rm.groupby(geo_lev)[adjust_att].value_counts().unstack().fillna(0)
            temp_hold.columns = [(temp_hold.columns.name, x) for x in temp_hold.columns]
            temp_hold = temp_hold.astype(int)
            ls_temp_hold.append(temp_hold)
        marg_new_raw = pd.concat(ls_temp_hold, axis=1)
        convert_marg_dict = {com_col: marg_new_raw[com_col] for com_col in marg_new_raw.columns}
        convert_marg_dict[("zone_id", None)] = marg_new_raw.index
        marg_hh = pd.DataFrame(convert_marg_dict)
    
        check = len(hh_df_got_rm)
        re_check_to_show.append(check)
        i += 1

        pool.to_csv(os.path.join(processed_data, "keep_check", f"updated_pool_{i}.csv"), index=False)
        marg_hh.to_csv(os.path.join(processed_data, "keep_check", f"updated_marg_{i}.csv"), index=False)

    # Process to combine final results of hh and df, mainly change id
    logger.info(f"DOING processing hhid after {i} ite")
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

    print(re_check_to_show)



if __name__ == "__main__":
    main()
