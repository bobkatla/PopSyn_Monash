import pandas as pd
import math


def SRMSE(synthetic_df, tot_df, df_controls, skip_ls=[]):
    households = synthetic_df
    persons = synthetic_df

    hold = 0
    ite = 0
    for name, df, att, exp in zip(df_controls['target'], df_controls['seed_table'], df_controls['control_field'],  df_controls['expression']):
        if att in skip_ls: continue
        df = eval(df)
        filtered_df = df[eval(exp)]
        syn_num = len(filtered_df)
        expected_num = int(tot_df[att])
        hold += (expected_num - syn_num) ** 2
        ite += 1
    return math.sqrt(hold / ite)


if __name__ == "__main__":
    syn_df = pd.read_csv("synthetic_households.csv")
    tot_df = pd.read_csv("data/STATE_all_control.csv")
    con_df = pd.read_csv("hh_controls.csv")
    print(SRMSE(syn_df, tot_df, con_df))