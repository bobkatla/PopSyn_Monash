import pandas as pd
from process_data import HH_ATTS, PP_ATTS, ALL_RELA
from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score, learn_para_BN
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State


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


def inference_model_get(ls_rela):
    re_dict = {}
    for rela in ls_rela:
        df = pd.read_csv(f"../data/connect_main_{rela}.csv")
        id_cols = [x for x in df.columns if "hhid" in x or "persid" in x]
        df = df.drop(columns=id_cols)
        print(f"Learn BN {rela}")
        model = learn_struct_BN_score(df, show_struct=False)
        model = learn_para_BN(model, df)
        re_dict[rela] = BayesianModelSampling(model)
    return re_dict


def process_rela_connect(main_pp_df, infer_model, rela):
    print(f"Processing the relationship {rela}")
    # Loop through each HH and append
    all_cols = [x for x in main_pp_df.columns if x not in ALL_RELA]
    all_cols.remove("hhid")
    ls_df = []
    for i, row in main_pp_df.iterrows():
        evidences = [State(f"{name}_main", row[name]) for name in all_cols]
        syn = infer_model.rejection_sample(evidence=evidences, size=row[rela], show_progress=True)
        remove_cols = [x for x in syn.columns if "_main" in x]
        syn = syn.drop(columns=remove_cols)
        if row[rela] > 0:
            syn.columns = syn.columns.str.rstrip(f'_{rela}')
            syn["relationship"] = rela
            syn["hhid"] = row["hhid"]
            ls_df.append(syn)
    re_df = pd.concat(ls_df)
    return re_df


def main():
    # Import the synthetic with main and households
    combine_df = pd.read_csv(r"..\data\tempt\SynPop_forward_hh_main_sa1.csv")
    # Process the HH and main to have the HH with IDs and People in HH
    hh_df, main_pp_df_all = process_combine_df(combine_df)
    # Store the HH in df, Store the main in a list to handle later
    store_pp_df = extra_pp_df(main_pp_df_all)
    ls_df_pp = [store_pp_df]

    all_rela_exist = ALL_RELA.copy()
    all_rela_exist.remove("Self")

    dict_model_inference = inference_model_get(all_rela_exist)

    # Process the HH and with each rela
    for rela in all_rela_exist:
        infer_model = dict_model_inference[rela]
        pop_rela = process_rela_connect(main_pp_df_all, infer_model, rela)
        ls_df_pp.append(pop_rela)

    fin_pp_df = pd.concat(ls_df_pp)
    hh_df.to_csv("syn_hh.csv", index=False)
    fin_pp_df.to_csv("syn_pp_connected.csv", index=False)
    

if __name__ ==  "__main__":
    main()