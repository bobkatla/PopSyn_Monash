"""
New way to do rejection sample, especially for the main to other
Sample alot (like 10 mil) and select the needed sample from it, we can combine with sample method to randomly draw
"""
import pandas as pd
import pickle
import os

from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score, learn_para_BN
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BayesianEstimator

from PopSynthesis.Methods.connect_HH_PP.paras_dir import processed_data
from PopSynthesis.Methods.connect_HH_PP.scripts.const import *


init_n_pool = 20000


def process_combine_df(combine_df):
    combine_df["hhid"] = combine_df.index
    hh_df = combine_df[HH_ATTS]
    all_rela_exist = ALL_RELA.copy()
    all_rela_exist.remove("Self")
    hh_df["hhsize"] = combine_df[all_rela_exist].sum(axis=1)
    pp_cols = PP_ATTS + all_rela_exist
    pp_cols.remove("relationship")
    pp_cols.remove("persid")
    pp_df = combine_df[pp_cols]
    return hh_df, pp_df


def extra_pp_df(pp_df):
    to_drop_cols = [x  for x in pp_df.columns if x in ALL_RELA]
    pp_df = pp_df.drop(columns=to_drop_cols)
    pp_df["relationsip"] = "Self"
    return pp_df


def learn_para_BN_diric(model, data_df, state_names):
    para_learn = BayesianEstimator(
            model=model,
            data=data_df,
            state_names=state_names
        )
    ls_CPDs = para_learn.get_parameters(
        prior_type='K2'
    )
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
        df = pd.read_csv(f"../data/connect_main_{rela}.csv")
        id_cols = [x for x in df.columns if "hhid" in x or "persid" in x]
        df = df.drop(columns=id_cols)
        print(f"Learn BN {rela}")
        rela_state_names = get_2_pp_connect_state_names(state_names_base, rela)
        model = learn_struct_BN_score(df, show_struct=False, state_names=rela_state_names)
        model = learn_para_BN_diric(model, df, state_names=rela_state_names)
        re_dict[rela] = BayesianModelSampling(model)
    return re_dict


def process_rela_fast(main_pp_df, infer_model, rela):
    pool = infer_model.forward_sample(size=init_n_pool, show_progress=True)

    all_cols = [x for x in main_pp_df.columns if x not in ALL_RELA]
    all_cols.remove("hhid")
    all_cols_main = [f"{x}_main" for x in all_cols]
    all_cols_rela_rename = {f"{x}_{rela}": x for x in all_cols}

    all_val_in_pool = pool[all_cols_main].value_counts()
    sub_pp_df = main_pp_df[main_pp_df[rela] > 0]
    all_val_in_rela_sub = sub_pp_df[all_cols].value_counts()

    dict_to_sample = {}
    to_delete = []

    for val in all_val_in_rela_sub.index:
        print(f"PROCESS main {val}")
        if val in all_val_in_pool.index:
            query = True
            for i, col in enumerate(all_cols_main):
                query &= pool[col] == val[i]
            sub_pool_sample = pool[query]
            dict_to_sample[val] = sub_pool_sample
        else:
            query = True
            for i, col in enumerate(all_cols):
                query &= sub_pp_df[col] == val[i]
            sub_rela_de = sub_pp_df[query]
            to_delete.append(sub_rela_de)

    print("GETTING the hhid that is hard to sample, maybe wrong")
    to_del_df = pd.concat(to_delete)
    check_df = sub_pp_df[~sub_pp_df["hhid"].isin(to_del_df["hhid"])]
    print(f"Final DF atm for {rela}")

    print("DOING sampling by rela now")
    gb_df_hhid = check_df.groupby(all_cols)["hhid"].apply(lambda x: list(x))
    gb_df_num_rela = check_df.groupby(all_cols)[rela].apply(lambda x: list(x))
    comb_df = pd.merge(gb_df_hhid, gb_df_num_rela, left_index=True, right_index=True)

    ls_to_com_df = []
    hold_ids = []
    for check_val, ls_hhid, ls_rela in zip(comb_df.index, comb_df["hhid"], comb_df[rela]):
        to_sample_df = dict_to_sample[check_val]
        tot = 0
        for hhid, n in zip (ls_hhid, ls_rela):
            hold_id = [hhid]*n
            hold_ids += hold_id
            tot += n
        re_df = to_sample_df.sample(n=tot, replace=True)
        ls_to_com_df.append(re_df)
    final_rela_df = pd.concat(ls_to_com_df)
    final_rela_df["hhid"] = hold_ids
    final_rela_df["relationship"] = rela
    
    return to_del_df, final_rela_df


def main():
    # Import the synthetic with main and households
    combine_df = pd.read_csv(r"..\output\SynPop_hh_main_POA.csv")
    # Process the HH and main to have the HH with IDs and People in HH
    hh_df, main_pp_df_all = process_combine_df(combine_df)
    # Store the HH in df, Store the main in a list to handle later
    store_pp_df = extra_pp_df(main_pp_df_all)
    ls_df_pp = [store_pp_df]

    state_names_pp = None
    with open(os.path.join(processed_data, 'dict_pp_states.pickle'), 'rb') as handle:
        state_names_pp = pickle.load(handle)

    all_rela_exist = ALL_RELA.copy()
    all_rela_exist.remove("Self")

    dict_model_inference = inference_model_get(all_rela_exist, state_names_pp)

    for rela in all_rela_exist:
        infer_model = dict_model_inference[rela]
        to_del_df, pop_rela = process_rela_fast(main_pp_df_all, infer_model, rela)
        pop_rela.to_csv(os.path.join(processed_data, f"pp_{rela}.csv"), index=False)
        to_del_df.to_csv(os.path.join(processed_data, f"del_main_pp_{rela}.csv"), index=False)
        # ls_df_pp.append(pop_rela)
    # sample to have the pool
    # Group the available HH into diffrent group and get total number number



if __name__ == "__main__":
    main()