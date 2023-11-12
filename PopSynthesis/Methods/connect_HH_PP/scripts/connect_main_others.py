import pandas as pd
from process_data import HH_ATTS, PP_ATTS, ALL_RELA
from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score, learn_para_BN
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
from pgmpy.estimators import BayesianEstimator
import pickle5 as pickle


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


def process_rela_connect(main_pp_df, infer_model, rela):
    # Loop through each HH and append
    all_cols = [x for x in main_pp_df.columns if x not in ALL_RELA]
    all_cols.remove("hhid")
    sub_pp_df = main_pp_df[main_pp_df[rela] > 0]
    ls_df = []
    for i, row in sub_pp_df.iterrows():
        if i % 100 == 0:
            print(f"DOING  {rela}: {(i*100)/len(main_pp_df)}%")
        evidences = [State(f"{name}_main", row[name]) for name in all_cols]
        print(evidences, rela)
        syn = infer_model.rejection_sample(evidence=evidences, size=row[rela], show_progress=True)
        remove_cols = [x for x in syn.columns if "_main" in x]
        syn = syn.drop(columns=remove_cols)
        syn.columns = syn.columns.str.rstrip(f'_{rela}')
        syn["relationship"] = rela
        syn["hhid"] = row["hhid"]
        ls_df.append(syn)
    re_df = pd.concat(ls_df)
    return re_df


def main():
    # Import the synthetic with main and households
    combine_df = pd.read_csv(r"..\output\SynPop_hh_main_POA.csv")
    # Process the HH and main to have the HH with IDs and People in HH
    hh_df, main_pp_df_all = process_combine_df(combine_df)
    # Store the HH in df, Store the main in a list to handle later
    store_pp_df = extra_pp_df(main_pp_df_all)
    ls_df_pp = [store_pp_df]

    state_names_pp = None
    with open('../data/dict_pp_states.pickle', 'rb') as handle:
        state_names_pp = pickle.load(handle)

    all_rela_exist = ALL_RELA.copy()
    all_rela_exist.remove("Self")

    dict_model_inference = inference_model_get(all_rela_exist, state_names_pp)

    # Process the HH and with each rela
    for rela in all_rela_exist:
        infer_model = dict_model_inference[rela]
        pop_rela = process_rela_connect(main_pp_df_all, infer_model, rela)
        pop_rela.to_csv(f"pp_{rela}.csv", index=False)
        ls_df_pp.append(pop_rela)

    fin_pp_df = pd.concat(ls_df_pp)
    hh_df.to_csv("syn_hh.csv", index=False)
    fin_pp_df.to_csv("syn_pp_connected.csv", index=False)
    

if __name__ ==  "__main__":
    main()