"""
New way to do rejection sample, especially for the main to other
Sample alot (like 10 mil) and select the needed sample from it, we can combine with sample method to randomly draw
"""
import pandas as pd
import pickle
import os

from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator

from PopSynthesis.Methods.connect_HH_PP.paras_dir import processed_data, geo_lev
from PopSynthesis.DataProcessor.utils.seed.pp.process_relationships import (
    AVAILABLE_RELATIONSHIPS,
)
from PopSynthesis.Methods.connect_HH_PP.scripts.const import PP_ATTS, HH_ATTS


init_n_pool = int(1e7)  # 10 Mils


def process_combine_df(combine_df):
    hh_df = combine_df[HH_ATTS + [geo_lev]]
    all_rela_exist = AVAILABLE_RELATIONSHIPS.copy()
    all_rela_exist.remove("Main")
    pp_cols = PP_ATTS + all_rela_exist + [geo_lev]
    pp_cols.remove("relationship")
    pp_cols.remove("persid")
    pp_df = combine_df[pp_cols]
    return hh_df, pp_df


def extra_pp_df(pp_df):
    to_drop_cols = [x for x in pp_df.columns if x in AVAILABLE_RELATIONSHIPS]
    pp_df = pp_df.drop(columns=to_drop_cols)
    pp_df["relationship"] = "Main"
    return pp_df


def learn_para_BN_diric(model, data_df, state_names):
    para_learn = BayesianEstimator(model=model, data=data_df, state_names=state_names)
    ls_CPDs = para_learn.get_parameters(prior_type="K2")
    model.add_cpds(*ls_CPDs)
    return model


def get_2_pp_connect_state_names(state_names_base, rela):
    new_dict_name = {}
    for name in state_names_base:
        new_dict_name[f"{name}_main"] = state_names_base[name]
        new_dict_name[f"{name}_{rela}"] = state_names_base[name]
    return new_dict_name


def inference_model_get(ls_rela, state_names_base):
    re_dict = {}
    for rela in ls_rela:
        df = pd.read_csv(os.path.join(processed_data, f"connect_main_{rela}.csv"))
        id_cols = [x for x in df.columns if "hhid" in x or "persid" in x]
        df = df.drop(columns=id_cols)
        print(f"Learn BN {rela}")
        rela_state_names = get_2_pp_connect_state_names(state_names_base, rela)
        model = learn_struct_BN_score(
            df, show_struct=False, state_names=rela_state_names
        )
        model = learn_para_BN_diric(model, df, state_names=rela_state_names)
        re_dict[rela] = BayesianModelSampling(model)
    return re_dict


def pools_get(ls_rela, dict_model_inference, pool_size):
    re_dict = {}
    for rela in ls_rela:
        infer_model = dict_model_inference[rela]
        pool = infer_model.forward_sample(size=int(pool_size), show_progress=True)
        re_dict[rela] = pool
    return re_dict


def process_rela_using_count(main_pp_df, rela, pool_count, geo_lev):
    dict_hhid_geo = dict(zip(main_pp_df["hhid"], main_pp_df[geo_lev]))

    cols_main = [x for x in pool_count if "_main" in x]
    cols_rela = [x for x in pool_count if f"_{rela}" in x]
    rename_to_main = {x.replace("_main", ""): x for x in cols_main}

    sub_pp_df = main_pp_df[main_pp_df[rela] > 0]
    pp_cols = [x for x in sub_pp_df.columns if x not in AVAILABLE_RELATIONSHIPS]
    sub_pp_df = sub_pp_df[pp_cols + [rela]]
    sub_pp_df = sub_pp_df.rename(columns=rename_to_main)

    pool_count["comb_rela"] = pool_count.apply(
        lambda r: ([r[c] for c in cols_rela], r["count"]), axis=1
    )
    gb_pool = (
        pool_count.groupby(cols_main)["comb_rela"]
        .apply(lambda x: list(x))
        .reset_index()
    )
    gb_main_id = sub_pp_df.groupby(cols_main)["hhid"].apply(lambda x: list(x))
    gb_main_re = sub_pp_df.groupby(cols_main)[rela].apply(lambda x: list(x))
    gb_main = pd.concat([gb_main_id, gb_main_re], axis=1).reset_index()
    merge_df = gb_main.merge(gb_pool, on=cols_main, how="left")

    # Process keep df, this is the rela
    keep_df = merge_df[~merge_df["comb_rela"].isna()]

    def select_ran_sam(r):
        ls_num = r[rela]  # this match with ls hhid
        ls_pos_choose = r["comb_rela"]
        temp_df = pd.DataFrame(ls_pos_choose, columns=["comb", "weight"])
        temp_df["weight"] = temp_df["weight"] / temp_df["weight"].sum()
        final_list = []
        for num in ls_num:
            selected_df = temp_df.sample(n=num, replace=True, weights="weight")
            selected_combs = list(selected_df["comb"])
            final_list.append(selected_combs)
        return final_list

    keep_df["selected"] = keep_df.apply(select_ran_sam, axis=1)
    rela_df = keep_df[["hhid", "selected"]].explode(["hhid", "selected"])
    rela_df = rela_df.explode(["selected"])
    rela_df[cols_rela] = pd.DataFrame(rela_df["selected"].tolist())
    rela_df = rela_df.drop(columns=["selected"])
    if len(rela_df) != 0:
        rela_df[geo_lev] = rela_df.apply(lambda r: dict_hhid_geo[r["hhid"]], axis=1)
    else:
        rela_df[geo_lev] = None
    rela_df["relationship"] = rela
    rename_rela_to = {x: x.replace(f"_{rela}", "") for x in cols_rela}
    rela_df = rela_df.rename(columns=rename_rela_to)

    # Process del df, this is Main
    del_df_main = merge_df[merge_df["comb_rela"].isna()]
    del_df_main = del_df_main.drop(columns=["comb_rela"])
    del_df_main = del_df_main.explode(["hhid"])
    del_df_main["relationship"] = "Self"
    if len(del_df_main) != 0:
        del_df_main[geo_lev] = del_df_main.apply(
            lambda r: dict_hhid_geo[r["hhid"]], axis=1
        )
    else:
        del_df_main[geo_lev] = None

    return del_df_main, rela_df


def process_rela_fast(main_pp_df, rela, pool):
    all_cols = [
        x
        for x in main_pp_df.columns
        if x not in AVAILABLE_RELATIONSHIPS and x != geo_lev
    ]
    all_cols.remove("hhid")
    all_cols_main = [f"{x}_main" for x in all_cols]
    all_cols_rela_rename = {f"{x}_{rela}": x for x in all_cols}

    all_val_in_pool = pool[all_cols_main].value_counts()
    sub_pp_df = main_pp_df[main_pp_df[rela] > 0]
    all_val_in_rela_sub = sub_pp_df[all_cols].value_counts()

    dict_to_sample = {}
    to_delete = []

    ls_val_rela = list(all_val_in_rela_sub.index)
    ls_val_pool = list(all_val_in_pool.index)
    ls_val_both = list(set(ls_val_rela) & set(ls_val_pool))
    ls_val_not_in_pool = list(set(ls_val_rela) - set(ls_val_pool))

    for val in ls_val_both:
        print(f"PROCESS to keep {val}")
        q = ""
        for i, col in enumerate(all_cols_main):
            if i != 0:
                q += " & "
            q += f"{col} == '{val[i]}'"
        sub_pool_sample = pool.query(q)
        dict_to_sample[val] = sub_pool_sample
    for val in ls_val_not_in_pool:
        print(f"PROCESS to del {val}")
        q = ""
        for i, col in enumerate(all_cols):
            if i != 0:
                q += " & "
            q += f"{col} == '{val[i]}'"
        sub_rela_de = sub_pp_df.query(q)
        to_delete.append(sub_rela_de)

    print("GETTING the hhid that is hard to sample, maybe wrong")
    if len(to_delete) > 0:
        to_del_df = pd.concat(to_delete)
        check_df = sub_pp_df[~sub_pp_df["hhid"].isin(to_del_df["hhid"])]
    else:
        to_del_df = None
        check_df = sub_pp_df
    print(f"Final DF atm for {rela}")

    print("DOING sampling by rela now")
    gb_df_hhid = check_df.groupby(all_cols)["hhid"].apply(lambda x: list(x))
    gb_df_num_rela = check_df.groupby(all_cols)[rela].apply(lambda x: list(x))
    comb_df = pd.merge(gb_df_hhid, gb_df_num_rela, left_index=True, right_index=True)
    ls_to_com_df = []
    hold_ids = []
    for check_val, ls_hhid, ls_rela in zip(
        comb_df.index, comb_df["hhid"], comb_df[rela]
    ):
        to_sample_df = dict_to_sample[check_val]
        tot = 0
        for hhid, n in zip(ls_hhid, ls_rela):
            hold_id = [hhid] * n
            hold_ids += hold_id
            tot += n
        re_df = to_sample_df.sample(n=tot, replace=True)
        ls_to_com_df.append(re_df)
    if len(ls_to_com_df) == 0:
        print("Weird this has nothing")
        final_rela_df = []
    else:
        final_rela_df = pd.concat(ls_to_com_df)
        final_rela_df["hhid"] = hold_ids
        final_rela_df["relationship"] = rela

    return to_del_df, final_rela_df


def main():
    # Import the synthetic with main and households
    # combine_df = pd.read_csv(os.path.join(processed_data, f"SynPop_hh_main_{geo_lev}.csv"))
    combine_df = pd.read_csv(os.path.join(processed_data, "1e5_hh_main.csv"))
    combine_df = combine_df[combine_df["age"].notna()]
    # Process the HH and main to have the HH with IDs and People in HH
    hh_df, main_pp_df_all = process_combine_df(combine_df)
    # Store the HH in df, Store the main in a list to handle later
    store_pp_df = extra_pp_df(main_pp_df_all)
    ls_df_pp = [store_pp_df]

    state_names_pp = None
    with open(os.path.join(processed_data, "dict_pp_states.pickle"), "rb") as handle:
        state_names_pp = pickle.load(handle)

    all_rela_exist = AVAILABLE_RELATIONSHIPS.copy()
    all_rela_exist.remove("Main")

    dict_model_inference = inference_model_get(all_rela_exist, state_names_pp)
    dict_pool_sample = pools_get(all_rela_exist, dict_model_inference, 1e6)

    for rela in all_rela_exist:
        to_del_df, pop_rela = process_rela_fast(
            main_pp_df_all, rela, dict_pool_sample[rela]
        )
        pop_rela.to_csv(
            os.path.join(processed_data, f"1e5_test_pp_{rela}.csv"), index=False
        )
        to_del_df.to_csv(
            os.path.join(processed_data, f"1e5_test_del_main_pp_{rela}.csv"),
            index=False,
        )
        # ls_df_pp.append(pop_rela)
    # sample to have the pool
    # Group the available HH into diffrent group and get total number number


if __name__ == "__main__":
    main()
