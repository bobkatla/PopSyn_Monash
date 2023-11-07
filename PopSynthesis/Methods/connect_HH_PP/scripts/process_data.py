import pandas as pd
import numpy as np
from collections import defaultdict 


HH_ATTS = [
    "hhid",
    "dwelltype",
    "owndwell",
    "hhinc",
    "totalvehs"
]

PP_ATTS = [
    "persid",
    "hhid",
    "age",
    "sex",
    "relationship",
    "persinc",
    "nolicence",
    "anywork"
]

LS_GR_RELA = ["Self", "Spouse", "Child", "Grandchild"] # For the rest we will make them Others
HANDLE_THE_REST_RELA = "Others"
ALL_RELA = LS_GR_RELA + [HANDLE_THE_REST_RELA]


def check_rela_gb(gb_df):
    for hhid, rela_gr in zip(gb_df.index, gb_df):
        check_dict = defaultdict(lambda: 0)
        for i in rela_gr: check_dict[i] += 1 
        if check_dict["Self"] == 0:
            # print(hhid)
            print([f"{x} - {y}" for x, y in check_dict.items() if x != "Self"])
        elif check_dict["Self"] > 1:
            print("NOOOOOOOOOO", hhid, rela_gr)


def process_rela(pp_df):
    # We will have 4 groups: spouse, child, grandchild and others
    # First we need to make sure each HH has 1 Self

    gb_df = pp_df.groupby("hhid")["relationship"].apply(lambda x: list(x))
    # check_rela_gb(gb_df)
    
    # There are cases of no Self but 2 spouses, we need to process them
    # There are various cases, requires some manual works
    # In order of replacement: 1 person, 2 spouses, 1 spouse, no spouse then pick the oldest
    # Thus we have 2 way of replacement: oldest (apply for 1 person and others) and spouse
    ls_to_replace = []
    for hhid, rela_gr in zip(gb_df.index, gb_df):
        check_dict = defaultdict(lambda: 0)
        for i in rela_gr: check_dict[i] += 1
        if check_dict["Self"] == 0:
            replace_method = "oldest" if check_dict["Spouse"] == 0 else "spouse"
            ls_to_replace.append((hhid, replace_method))

    # start to replace to fix errors
    for hhid, replace_method in ls_to_replace:
        sub_df = pp_df[pp_df["hhid"]==hhid]
        idx_to_replace = None
        if replace_method == "spouse":
            sub_sub_df = sub_df[sub_df["relationship"]=="Spouse"]
            idx_to_replace = sub_sub_df.index[0]
        elif replace_method == "oldest":
            idx_to_replace = sub_df["age"].idxmax()
        assert idx_to_replace is not None
        pp_df.at[idx_to_replace, "relationship"] = "Self"

    # check again
    gb_df_2 = pp_df.groupby("hhid")["relationship"].apply(lambda x: list(x))
    check_rela_gb(gb_df_2) # Should print nothing

    # replace values in columns
    pp_df.loc[~pp_df["relationship"].isin(LS_GR_RELA), "relationship"] = HANDLE_THE_REST_RELA
    # print(pp_df["relationship"].unique())
    return pp_df


def adding_pp_related_atts(pp_df, hh_df):
    # This adding the persons-related atts to the hh df for later sampling
    # at the moment we will use to have the number of each relationship
    # the total will make the hhsize
    ls_rela = pp_df["relationship"].unique()
    gb_df_pp = pp_df.groupby("hhid")["relationship"].apply(lambda x: list(x))
    dict_count_rela = {}
    for hhid, rela_gr in zip(gb_df_pp.index, gb_df_pp):
        check_dict = {x: 0 for x in ls_rela}
        for i in rela_gr: check_dict[i] += 1
        dict_count_rela[hhid] = check_dict

    for rela in ls_rela:
        hh_df[rela] = hh_df.apply(lambda row: dict_count_rela[row["hhid"]][rela], axis=1)

    # check Self again
    assert len(hh_df["Self"].unique()) == 1
    assert hh_df["Self"].unique()[0] == 1

    return hh_df.drop(columns=["Self"])


def process_hh_main_person(hh_df, main_pp_df, to_csv=False, name_file="connect_hh_main"):
    # they need to perfect match
    assert len(hh_df) == len(main_pp_df)
    combine_df = hh_df.merge(main_pp_df, on="hhid", how="inner")
    combine_df = combine_df.drop(columns=["relationship"])
    if to_csv:
        combine_df.to_csv(f"../data/{name_file}.csv", index=False)
    return combine_df


def process_main_other(main_pp_df, sub_df, rela, to_csv=True):
    assert len(main_pp_df["relationship"].unique()) == 1 # It is Self
    assert len(sub_df["relationship"].unique()) == 1 # It is the relationship we checking
    # Change the name to avoid confusion
    main_pp_df = main_pp_df.add_suffix('_main', axis=1)
    sub_df = sub_df.add_suffix(f'_{rela}', axis=1)
    main_pp_df = main_pp_df.rename(columns={"hhid_main": "hhid"})
    sub_df = sub_df.rename(columns={f"hhid_{rela}": "hhid"})

    combine_df = main_pp_df.merge(sub_df, on="hhid", how="right")
    combine_df = combine_df.drop(columns=[f"relationship_{rela}", "relationship_main"])
    
    if to_csv:
        combine_df.to_csv(f"../data/connect_main_{rela}.csv", index=False)
    
    return combine_df


def main():
    # Import HH and PP samples (VISTA)
    hh_df = pd.read_csv("..\..\..\Generator_data\data\source2\VISTA\SA\H_VISTA_1220_SA1.csv")[HH_ATTS]
    pp_df = pd.read_csv("..\..\..\Generator_data\data\source2\VISTA\SA\P_VISTA_1220_SA1.csv")[PP_ATTS]

    pp_df = process_rela(pp_df)
    hh_df = adding_pp_related_atts(pp_df, hh_df)

    #Tempo saving
    # pp_df.to_csv("../data/first_processed_all_P.csv", index=False)
    # hh_df.to_csv("../data/first_processed_all_H.csv", index=False)
    
    main_pp_df = pp_df[pp_df["relationship"]=="Self"]
    
    # process hh_main
    df_hh_main = process_hh_main_person(hh_df, main_pp_df, to_csv=True)

    for rela in ALL_RELA:
        if rela != "Self":
            sub_df = pp_df[pp_df["relationship"]==rela]
            df_main_other = process_main_other(main_pp_df, sub_df, rela=rela, to_csv=True)


if __name__ == "__main__":
    main()