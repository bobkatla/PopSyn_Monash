"""
After we got the households, now we need 
"""


import pandas as pd
from PopSynthesis.Methods.connect_HH_PP.scripts.adjust_atts_by_counts import *
from PopSynthesis.Methods.connect_HH_PP.scripts.sample_hh_main import *
from PopSynthesis.Methods.connect_HH_PP.scripts.process_all_hh_pp import *
from PopSynthesis.Methods.connect_HH_PP.scripts.get_hh_main_combine import *
from PopSynthesis.Methods.connect_HH_PP.scripts.utils import *
import logging

logger = logging.getLogger("connect_hh_pp")


def main():
    geo_lev = "POA"
    processed_already = [
        "hhsize",
        "totalvehs",
        "hhinc",
        "dwelltype",
    ]  # Maybe don't need to run for that
    # processed_already = ["hhsize", "totalvehs"]

    all_rela_exist = AVAILABLE_RELATIONSHIPS.copy()
    all_rela_exist.remove("Main")

    ls_final_hh = []
    ls_final_pp = []

    check = np.inf
    i = 0
    re_check_to_show = []

    # Only importing now
    # Process original census
    ori_census = pd.read_csv(
        os.path.join(data_dir, "hh_marginals_ipu.csv"), header=[0, 1]
    )
    ls_drop = list(
        ori_census.columns[
            ori_census.columns.get_level_values(0).isin(["zone_id", "sample_geog"])
        ]
    )
    new_ori_census = ori_census.set_index(
        ori_census[
            ori_census.columns[ori_census.columns.get_level_values(0) == "zone_id"]
        ].iloc[:, 0]
    )
    new_ori_census.index = new_ori_census.index.astype(str)
    marg_hh = new_ori_census.drop(columns=ls_drop)

    # with open(os.path.join(processed_data, "dict_pool_sample.pickle"), "rb") as handle:
    #     dict_pool_sample = pickle.load(handle)

    ############# TEST #################
    with open(os.path.join(processed_data, "dict_hh_states.pickle"), "rb") as handle:
        hh_state_names = pickle.load(handle)
    with open(os.path.join(processed_data, "dict_pp_states.pickle"), "rb") as handle:
        pp_state_names = pickle.load(handle)
    state_names = hh_state_names | pp_state_names

    # learning to get the HH only with main person
    df_seed_hh_main = pd.read_csv(os.path.join(processed_data, "connect_hh_main.csv"))
    # drop all the ids as they are not needed for in BN learning
    id_cols_hh_main = [
        x for x in df_seed_hh_main.columns if "hhid" in x or "persid" in x
    ]
    df_seed_hh_main = df_seed_hh_main.drop(columns=id_cols_hh_main)
    logger.info("GETTING the pool HH and Main")
    pool_hh_main = get_pool(df_seed_hh_main, state_names, int(1e5), special=False)
    dict_model_inference = inference_model_get(all_rela_exist, pp_state_names)
    dict_pool_sample = pools_get(all_rela_exist, dict_model_inference, int(1e4))
    dict_pool_sample["Main"] = pool_hh_main

    # chosen_poa = ["3000", "3002"]
    # marg_hh = marg_hh.loc[chosen_poa]
    #############################################

    # Create pool, pool here is the household count, will fix alot later

    pool = pd.read_csv(
        os.path.join(processed_data, "new_save_pools", "final_pool_count_new_rela.csv")
    )
    pool = pool.astype(str)
    pool["count"] = pool["count"].astype(int)
    cols_pool = [x for x in pool.columns if x != "count"]

    while check > 10 and i < 50:
        logger.info(f"DOING ITE {i} with err == {check}")
        if i == 0:  # init begin by having the adjust hh_df
            hh_df = pd.read_csv(
                os.path.join(output_dir, "adjust", "saving_hh_test2_dwelltype.csv"),
                low_memory=False,
            )
            # hh_df = hh_df.astype(str)
            # hh_df = hh_df[hh_df["POA"].isin(chosen_poa)]
        else:
            # Simple create a new func here and get the new marg already
            hh_df = process_data_general(
                marg_hh, pool, geo_lev, processed_already, is_ipu_data=False
            )
            hh_df.to_csv(
                os.path.join(
                    processed_data, "keep_check", f"new_rela_adjusted4_hh_new_{i}.csv"
                ),
                index=False,
            )

        if "hhid" not in hh_df.columns:
            hh_df = hh_df.reset_index(drop=True)
            hh_df["hhid"] = hh_df.index
            assert len(hh_df["hhid"].unique()) == len(hh_df)  # gotta be unique

        # Sample hh main
        logger.info("GETTING the main people")
        ls_to_gb = [x for x in hh_df.columns if x != "hhid"]
        hh_df = hh_df.astype(str)
        count_hh_df = (
            hh_df.groupby(ls_to_gb)["hhid"].apply(lambda x: list(x)).reset_index()
        )
        ls_sub_df = segment_df(count_hh_df, chunk_sz=100000)

        _ls_df_com, _init_del = [], []
        for sub_df in ls_sub_df:
            _df_com, _del_sub = get_combine_df(
                sub_df, dict_pool_sample["Main"].value_counts().reset_index()
            )
            if len(_df_com) > 0:
                _ls_df_com.append(_df_com)
            if len(_del_sub) > 0:
                _init_del.append(_del_sub)

        combine_df_hh_main = (
            pd.concat(_ls_df_com) if len(_ls_df_com) > 0 else pd.DataFrame()
        )
        del_hh = pd.concat(_init_del) if len(_init_del) > 0 else pd.DataFrame()
        # combine_df_hh_main, del_hh = get_combine_df(hh_df, dict_pool_sample["Main"].copy().value_counts().reset_index())
        # Process the HH and main to have the HH with IDs and People in HH
        logger.info("SOME EXTRA PROCESS PP")
        _, main_pp_df_all = process_combine_df(combine_df_hh_main)
        main_pp_df_all[all_rela_exist] = (
            main_pp_df_all[all_rela_exist].astype(float).astype(int)
        )

        del_hh.to_csv(
            os.path.join(processed_data, "keep_check", f"new_rela_del4_first_{i}.csv"),
            index=False,
        )
        main_pp_df_all.to_csv(
            os.path.join(processed_data, "keep_check", f"new_rela_main4_pp_df_{i}.csv"),
            index=False,
        )

        store_pp_df = extra_pp_df(main_pp_df_all)
        ls_df_pp = [store_pp_df]

        main_pp_df_all[all_rela_exist] = main_pp_df_all[all_rela_exist].astype(int)
        dict_hhid = dict(zip(hh_df["hhid"], hh_df[geo_lev]))

        del_df = []
        for rela in all_rela_exist:
            logger.info(f"Doing {rela} now lah~")
            to_del_df, pop_rela = process_rela_fast(
                main_pp_df_all, rela, dict_pool_sample[rela].copy()
            )  # fix this

            if len(pop_rela) > 0:
                pop_rela[geo_lev] = pop_rela.apply(
                    lambda r: dict_hhid[str(r["hhid"])], axis=1
                )
                cols_main = [
                    f"{x}_main"
                    for x in PP_ATTS
                    if x not in ["relationship", "persid", "hhid", geo_lev]
                ]
                rename_cols = {
                    f"{name}_{rela}": name
                    for name in PP_ATTS
                    if name not in ["relationship", "persid", "hhid", geo_lev]
                }
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

        if len(del_hh) == 0:
            ls_del_init = []
        else:
            ls_del_init = list(del_hh["hhid"].astype(str))

        del_df_final = pd.concat(del_df)
        ls_del_id = list(del_df_final["hhid"].astype(str)) + ls_del_init
        hh_df["hhid"] = hh_df["hhid"].astype(str)

        hh_df_keep = hh_df[~hh_df["hhid"].isin(ls_del_id)]
        hh_df_got_rm = hh_df[hh_df["hhid"].isin(ls_del_id)]

        logger.info("Updating the pool")
        temp_pool = pool.set_index(cols_pool)
        temp_keep = hh_df_got_rm.set_index(cols_pool)
        pool = temp_pool[~temp_pool.index.isin(temp_keep.index)]
        pool = pool.reset_index()
        logger.info("Done updating the pool")
        # Get the new marg to handle the new df

        # We do not convert the deleted to new marg anymore
        # We created the new ones based on kept and update the kept
        ls_cols_not_marg = [
            x for x in hh_df_keep if x not in processed_already
        ]  # not used yet, will think should I make it aim for only the needed cols for faster converge

        marg_from_kept_hh = convert_full_to_marg_count(
            hh_df_keep, geo_lev, AVAILABLE_RELATIONSHIPS + ["POA", "hhid"]
        )
        diff_marg = get_diff_marg(marg_hh, marg_from_kept_hh)
        new_kept_hh = adjust_kept_hh_match_census(hh_df_keep, diff_marg, geo_lev)

        # checking
        new_kept_marg = convert_full_to_marg_count(
            new_kept_hh, geo_lev, AVAILABLE_RELATIONSHIPS + ["POA", "hhid"]
        )
        new_diff_marg = get_diff_marg(marg_hh, new_kept_marg)
        checking_not_neg = new_diff_marg < 0
        assert checking_not_neg.any(axis=None) == False

        # now get the new marg
        marg_hh = new_diff_marg

        # update and append hh and pp
        new_fin_pp = all_df_pp[all_df_pp["hhid"].isin(new_kept_hh["hhid"])]

        ls_final_hh.append(new_kept_hh)
        ls_final_pp.append(new_fin_pp)

        hh_df_got_rm.to_csv(
            os.path.join(processed_data, "keep_check", f"new_rela_del4_df_hh_{i}.csv"),
            index=False,
        )
        new_kept_hh.to_csv(
            os.path.join(processed_data, "keep_check", f"new_rela_hh4_keep_df_{i}.csv"),
            index=False,
        )
        new_fin_pp.to_csv(
            os.path.join(processed_data, "keep_check", f"new_rela_pp4_df_{i}.csv"),
            index=False,
        )

        check = len(hh_df) - len(new_kept_hh)
        re_check_to_show.append(check)

        pool.to_csv(
            os.path.join(
                processed_data, "keep_check", f"new_rela_updated4_pool_{i}.csv"
            ),
            index=False,
        )
        marg_hh.to_csv(
            os.path.join(
                processed_data, "keep_check", f"new_rela_updated4_marg_{i}.csv"
            ),
            index=False,
        )

        i += 1

    # Process to combine final results of hh and df, mainly change id
    logger.info(f"DOING processing hhid after {i} ite")
    new_ls_hh = []
    new_ls_pp = []
    max_id = 1
    for hh, pp in zip(ls_final_hh, ls_final_pp):
        hh["hhid"] = hh["hhid"].astype(int)
        pp["hhid"] = pp["hhid"].astype(int)
        hh["hhid"] = hh["hhid"] + max_id
        pp["hhid"] = pp["hhid"] + max_id
        max_id = int(max(hh["hhid"])) + 1
        new_ls_hh.append(hh)
        new_ls_pp.append(pp)

    final_hh = pd.concat(new_ls_hh)
    final_pp = pd.concat(new_ls_pp)

    # # Outputing
    print(final_hh)
    print(final_pp)
    final_pp.to_csv(
        os.path.join(output_dir, f"new_syn4_pp_final_{geo_lev}_ad4.csv"), index=False
    )
    final_hh.to_csv(
        os.path.join(output_dir, f"new_syn4_hh_final_{geo_lev}_ad4.csv"), index=False
    )

    print(re_check_to_show)


if __name__ == "__main__":
    main()
