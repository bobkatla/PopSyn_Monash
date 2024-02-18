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

    all_rela_exist = [x for x in ALL_RELA if x != "Self"]

    with open(os.path.join(processed_data, 'dict_pool_sample.pickle'), 'rb') as handle:
        dict_pool_sample = pickle.load(handle)
    

    i = 16
        
    logger.info(f"Doing adjust ite {i}")
    
    logger.info("Importing prev run")
    pool = pd.read_csv(os.path.join(processed_data, "keep_check", f"updated_pool_{i-1}.csv"))
    marg_hh = pd.read_csv(os.path.join(processed_data, "keep_check", f"updated_marg_{i-1}.csv"), header=[0,1])
    pool = pool.astype(str)
    pool["count"] = pool["count"].astype(int)
    cols_pool = [x for x in pool.columns if x != "count"]

    p_h_df = os.path.join(processed_data, "keep_check", f"adjusted_hh_new_{i}.csv")
    if os.path.exists(p_h_df):
        print(f"ye we have the hh for ite {i}")
        hh_df = pd.read_csv(p_h_df)
    else:
        hh_df = process_data_general(marg_hh, pool, geo_lev, processed_already)
        # hh_df.to_csv(p_h_df, index=False)
    
    # hh_df = pd.read_csv(r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\PopSynthesis\Methods\connect_HH_PP\output\testland\saving_hh_dwelltype.csv")

    if "hhid" not in hh_df.columns:
        hh_df["hhid"] = hh_df.index
    # Sample hh main
    logger.info("GETTING the main people")
    combine_df_hh_main, del_hh = get_combine_df(hh_df, dict_pool_sample["Main"].copy().value_counts().reset_index())
    # Process the HH and main to have the HH with IDs and People in HH
    logger.info("SOME EXTRA PROCESS PP")
    _, main_pp_df_all = process_combine_df(combine_df_hh_main)
    main_pp_df_all[all_rela_exist] = main_pp_df_all[all_rela_exist].astype(float).astype(int)

    # del_hh.to_csv(os.path.join(processed_data, "keep_check", f"del2_first_{i}.csv"), index=False)
    # main_pp_df_all.to_csv(os.path.join(processed_data, "keep_check", f"main2_pp_df_{i}.csv"), index=False)

    # del_hh = pd.read_csv(os.path.join(processed_data, "keep_check", f"del2_first_{i}.csv"))
    # main_pp_df_all = pd.read_csv(os.path.join(processed_data, "keep_check", f"main2_pp_df_{i}.csv"))

    store_pp_df = extra_pp_df(main_pp_df_all)
    ls_df_pp = [store_pp_df]

    del_df = []
    for rela in all_rela_exist:
        logger.info(f"Doing {rela} now lah~")
        to_del_df, pop_rela = process_rela_fast(main_pp_df_all, rela, dict_pool_sample[rela].copy()) # fix this

        if pop_rela is not None:
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
    
    if len(del_df) > 0:
        del_df_final = pd.concat(del_df)
        ls_del_id = list(del_df_final["hhid"].astype(str)) + list(del_hh["hhid"].astype(str))
    else:
        ls_del_id = list(del_hh["hhid"].astype(str))
    hh_df["hhid"] = hh_df["hhid"].astype(str)
    
    hh_df_keep = hh_df[~hh_df["hhid"].isin(ls_del_id)]
    hh_df_got_rm = hh_df[hh_df["hhid"].isin(ls_del_id)]

    # hh_df_got_rm.to_csv(os.path.join(processed_data, "keep_check", f"del2_df_hh_{i}.csv"), index=False)
    # hh_df_keep.to_csv(os.path.join(processed_data, "keep_check", f"hh2_keep_df_{i}.csv"), index=False)
    # all_df_pp.to_csv(os.path.join(processed_data, "keep_check", f"pp2_df_{i}.csv"), index=False)

    check = len(hh_df_got_rm)
    print(check)

    if check == 0:
        print("DONE, no more, check == 0")
    else:
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
        # Very simple, take the diff from original to the current
        ls_temp_hold = []
        for adjust_att in cols_pool:
            temp_hold = hh_df_keep.groupby(geo_lev)[adjust_att].value_counts().unstack().fillna(0)
            temp_hold.columns = [(temp_hold.columns.name, x) for x in temp_hold.columns]
            temp_hold = temp_hold.astype(int)
            ls_temp_hold.append(temp_hold)
        marg_new_raw = pd.concat(ls_temp_hold, axis=1)
        convert_marg_dict = {com_col: marg_new_raw[com_col] for com_col in marg_new_raw.columns}
        convert_marg_dict[("zone_id", None)] = marg_new_raw.index
        marg_got_kept = pd.DataFrame(convert_marg_dict)
        marg_hh = marg_hh.set_index(marg_hh[marg_hh.columns[marg_hh.columns.get_level_values(0)=="zone_id"]])
        marg_got_kept = marg_got_kept.set_index(marg_got_kept[marg_hh.columns[marg_hh.columns.get_level_values(0)=="zone_id"]])
        print(marg_hh)
        print(marg_got_kept)
        print(marg_hh  - marg_got_kept)

        # pool.to_csv(os.path.join(processed_data, "keep_check", f"updated_pool_{i}.csv"), index=False)
        # marg_hh.to_csv(os.path.join(processed_data, "keep_check", f"updated_marg_{i}.csv"), index=False)


if __name__ == "__main__":
    main()
