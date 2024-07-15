import pandas as pd
from collections import defaultdict
import polars as pl
import pickle
import os
pd.options.mode.chained_assignment = None  # default='warn'

from PopSynthesis.Methods.connect_HH_PP.paras_dir import data_dir, processed_data
from PopSynthesis.Methods.connect_HH_PP.scripts.const import *



def main():
    # Import HH and PP samples (VISTA)
    # Pre-processing
    hh_df_raw = pl.read_csv(os.path.join(data_dir ,"H_VISTA_1220_SA1.csv"))
    pp_df_raw = pl.read_csv(os.path.join(data_dir, "P_VISTA_1220_SA1.csv"))
    selected_hh_for_weights = hh_df_raw.select(pl.col("hhid", "wdhhwgt_sa3", "wehhwgt_sa3"))
    selected_pp_for_weights = hh_df_raw.select(pl.col("persid", "wdperswgt_sa3", "weperswgt_sa3"))

    weights_dict = get_weights_dict(selected_hh_for_weights, selected_pp_for_weights)
    files_exist = True
    if files_exist:
        hh_df = pd.read_csv(os.path.join(processed_data,"before_process_more_hh.csv"))
        pp_df = pd.read_csv(os.path.join(processed_data,"before_process_more_pp.csv"))
    else:
        # Process pp
        pp_df = process_rela(pp_df_raw[PP_ATTS])
        pp_df = get_main_max_age(pp_df)
        pp_df = convert_pp_age_gr(pp_df=pp_df)
        
        # Process hh
        hh_df = convert_all_hh_atts(hh_df_raw[HH_ATTS], pp_df)

        hh_df.to_csv(os.path.join(processed_data,"before_process_more_hh.csv"), index=False)
        pp_df.to_csv(os.path.join(processed_data,"before_process_more_pp.csv"), index=False)
    
    # return dict statenames for hh
    dict_hh_state_names = {hh_cols: list(hh_df[hh_cols].unique()) for hh_cols in hh_df.columns if hh_cols not in ALL_RELA and hh_cols not in NOT_INCLUDED_IN_BN_LEARN}
    with open(os.path.join(processed_data, 'dict_hh_states.pickle'), 'wb') as handle:
        pickle.dump(dict_hh_state_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return dict statenames for pp
    dict_pp_state_names = {pp_cols: list(pp_df[pp_cols].unique()) for pp_cols in pp_df.columns if pp_cols not in NOT_INCLUDED_IN_BN_LEARN}
    with open(os.path.join(processed_data, 'dict_pp_states.pickle'), 'wb') as handle:
        pickle.dump(dict_pp_state_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # This part is to create the just simple converted samples from hh and pp
    pp_df = add_weights_in_df(pp_df, weights_dict, type="pp")
    hh_df = add_weights_in_df(hh_df, weights_dict, type="hh")
    pp_df.to_csv(os.path.join(processed_data, "ori_sample_pp.csv"), index=False)
    hh_df.to_csv(os.path.join(processed_data, "ori_sample_hh.csv"), index=False)
    
    # process hh_main
    main_pp_df = pp_df[pp_df["relationship"]=="Main"]
    df_hh_main = process_hh_main_person(hh_df, main_pp_df, to_csv=True, include_weights=False)

    for rela in ALL_RELA:
        if rela != "Self":
            print(f"DOING {rela}")
            sub_df = pp_df[pp_df["relationship"]==rela]
            df_main_other = process_main_other(main_pp_df, sub_df, rela=rela, to_csv=True, include_weights=False)

if __name__ == "__main__":
    main()

