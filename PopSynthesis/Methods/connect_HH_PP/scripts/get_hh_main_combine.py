"""
We have the HH already (with the all the adjusted atts), now we need to run this script
to have the main pp df
then we can others rela

maybe I need to group the generating results into 1 place (census, samples, IPU and this, maybe we should get the normal BN as well)

"""

import pandas as pd
import numpy as np
from PopSynthesis.Methods.connect_HH_PP.paras_dir import processed_data, geo_lev, output_dir
from PopSynthesis.Methods.connect_HH_PP.scripts.const import *
from PopSynthesis.Methods.connect_HH_PP.scripts.sample_hh_main import *
from PopSynthesis.Methods.connect_HH_PP.scripts.sample_pp import *
from PopSynthesis.Methods.connect_HH_PP.scripts.process_all_hh_pp import *
import logging
# create logger
logger = logging.getLogger('connect_hh_pp')


def get_combine_df(hh_df, pool_hh_main):
    logger.info("PROCESSING the pool for later process")
    pool_hh_main = pool_hh_main.astype(str)
    count_pool = pool_hh_main.value_counts()
    count_pool = count_pool.reset_index()
    count_pool["id_pool"] = count_pool.index

    logger.info("Process the given HH df")
    hh_df = hh_df.astype(str)
    ls_match = list(hh_df.columns)
    ls_match.remove("hhid")
    gb_df = hh_df.groupby(ls_match)["hhid"].apply(lambda x: list(x))
    ls_match.remove("POA")
    gb_df = gb_df.reset_index()

    # Concat first then explode, it will save run time
    logger.info("Start combining to connect hh and pp")
    combine = gb_df.merge(count_pool, on=ls_match, how="left")
    combine["hhid"] = combine["hhid"].astype(str)

    c_ls_id = combine.groupby("hhid")["id_pool"].apply(lambda x: list(x))
    c_ls_count = combine.groupby("hhid")["count"].apply(lambda x: list(x))
    c_ls_com = pd.concat([c_ls_id, c_ls_count], axis=1)
    df_com = c_ls_com.reset_index()
    df_com["hhid"] = df_com["hhid"].apply(lambda x: eval(x))

    logger.info("Start select based on distribution in the pool")
    def ran_sam(r):
        ids_choose_from = r["id_pool"]
        if str(ids_choose_from[0]) == "nan":
            return None
        else:
            counts = r["count"]
            p = [x/ sum(counts) for x in counts]
            sz = len(r["hhid"])
            return np.random.choice(ids_choose_from, size=sz, p=p)

    df_com["selection"] = df_com.apply(ran_sam, axis=1)
    del_df = df_com[df_com["selection"].isna()]
    keep_df = df_com[~df_com["selection"].isna()]

    logger.info("Final processing before outputting the syn hh-main connect")
    # Process keeping
    keep_df = keep_df.drop(columns=["id_pool", "count"])
    keep_df = keep_df.explode(["hhid", "selection"])
    keep_df = keep_df.rename(columns={"selection": "id_pool"})

    sub_hh_df = hh_df[hh_df["hhid"].isin(keep_df["hhid"])]
    com_df_hh = keep_df.merge(sub_hh_df, on="hhid")

    count_pool = count_pool.drop(columns=hh_df.columns, errors="ignore")
    sub_count_pool = count_pool[count_pool["id_pool"].isin(keep_df["id_pool"])]
    com_df_hh_main = com_df_hh.merge(sub_count_pool, on="id_pool")
    com_df_hh_main = com_df_hh_main.drop(columns=["id_pool", "count"])

    # process del
    del_df = del_df.explode("hhid")
    com_df_del = del_df.merge(hh_df, on="hhid")
    com_df_del = com_df_del.drop(columns=["id_pool", "count", "selection"])

    assert len(com_df_del) + len(com_df_hh_main) == len(hh_df)
    
    return com_df_hh_main, com_df_del


def main():
    with open(os.path.join(processed_data, 'dict_hh_states.pickle'), 'rb') as handle:
        hh_state_names = pickle.load(handle)
    with open(os.path.join(processed_data, 'dict_pp_states.pickle'), 'rb') as handle:
        pp_state_names = pickle.load(handle)
    state_names = hh_state_names | pp_state_names

    #learning to get the HH only with main person
    df_seed_hh_main = pd.read_csv(os.path.join(processed_data, "connect_hh_main.csv"))
    # drop all the ids as they are not needed for in BN learning
    id_cols_hh_main = [x for x in df_seed_hh_main.columns if "hhid" in x or "persid" in x]
    df_seed_hh_main = df_seed_hh_main.drop(columns=id_cols_hh_main)
    logger.info("GETTING the pool HH and Main")
    pool_hh_main = get_pool(df_seed_hh_main, state_names, int(1e5))

    hh_df = pd.read_csv(os.path.join(output_dir, "adjust", "final", "saving_hh_dwelltype.csv"), low_memory=False)
    hh_df["hhid"] = hh_df.index

    combine_df, del_df = get_combine_df(hh_df, pool_hh_main)
    print(combine_df)
    print(len(hh_df))
    print(del_df)


if __name__ == "__main__":
    main()