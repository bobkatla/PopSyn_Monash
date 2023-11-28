import os
from glob import glob
import pandas as pd
import numpy as np

from PopSynthesis.Methods.connect_HH_PP.paras_dir import processed_data, output_dir, geo_lev
from PopSynthesis.Methods.connect_HH_PP.scripts.const import *

del_df = []
for file in glob(os.path.join(processed_data, "del*")):
    df = pd.read_csv(file)
    del_df.append(df)
del_df = pd.concat(del_df)

main_pp_df_hh = pd.read_csv(os.path.join(processed_data, f"SynPop_hh_main_{geo_lev}.csv"))
main_pp_df_hh["hhid"] = main_pp_df_hh.index

hh_cols = [x for x in HH_ATTS] + [geo_lev]
hh_df = main_pp_df_hh[hh_cols]


pp_cols = [x for x in PP_ATTS if x not in ["persid", "relationship"]]
pp_main_df = main_pp_df_hh[pp_cols]
pp_main_df["relationship"] = "Main"


all_df_pp = []
for file in glob(os.path.join(processed_data, "pp*")):
    df = pd.read_csv(file)
    rela = file.split("\\")[-1].split("_")[-1].split(".")[0]
    cols_main = [f"{x}_main" for x in PP_ATTS if x not in["relationship", "persid", "hhid"]]
    rename_cols = {f"{name}_{rela}": name for name in PP_ATTS if name not in["relationship", "persid", "hhid"]}
    df = df.drop(columns=cols_main)
    df = df.rename(columns=rename_cols)
    all_df_pp.append(df)
all_df_pp = pd.concat(all_df_pp)

final_pp_df = pd.concat([all_df_pp, pp_main_df])
final_pp_df = final_pp_df[~final_pp_df["hhid"].isin(del_df["hhid"])]

final_pp_df.to_csv(os.path.join(output_dir, "syn_pp_final.csv"), index=False)

hh_df_rm = hh_df[~hh_df["hhid"].isin(del_df["hhid"])]

counting_pp_hh = final_pp_df["hhid"].value_counts()
hh_df_rm["hhsize"] = hh_df_rm.apply(lambda r: counting_pp_hh[r["hhid"]], axis=1)

hh_df_rm.to_csv(os.path.join(output_dir, "syn_hh_final.csv"), index=False)
