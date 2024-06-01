import pandas as pd
from collections import defaultdict 
import pickle
import os
pd.options.mode.chained_assignment = None  # default='warn'

from PopSynthesis.Methods.connect_HH_PP.paras_dir import data_dir, processed_data
from PopSynthesis.Methods.connect_HH_PP.scripts.const import *


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


def adding_pp_related_atts(hh_df, pp_df):
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
    assert len(hh_df["Main"].unique()) == 1
    assert hh_df["Main"].unique()[0] == 1

    return hh_df.drop(columns=["Main"])


def process_hh_main_person(hh_df, main_pp_df, to_csv=False, name_file="connect_hh_main", include_weights=True):
    # they need to perfect match
    assert len(hh_df) == len(main_pp_df)
    combine_df = hh_df.merge(main_pp_df, on="hhid", how="inner")
    combine_df = combine_df.drop(columns=["relationship"])
    # For this we use the weights of the hh, we can change to main if we want to
    if "_weight_x" in combine_df.columns:
        combine_df = combine_df.rename(columns={"_weight_x": "_weight"})
        combine_df = combine_df.drop(columns=["_weight_y"])

    if not include_weights:
        combine_df = combine_df.drop(columns="_weight")
    
    if to_csv:
        combine_df.to_csv(os.path.join(processed_data ,f"{name_file}.csv"), index=False)
    return combine_df


def process_main_other(main_pp_df, sub_df, rela, to_csv=True, include_weights=True):
    assert len(main_pp_df["relationship"].unique()) == 1 # It is Main
    assert len(sub_df["relationship"].unique()) == 1 # It is the relationship we checking
    # Change the name to avoid confusion
    main_pp_df = main_pp_df.add_suffix('_main', axis=1)
    sub_df = sub_df.add_suffix(f'_{rela}', axis=1)
    main_pp_df = main_pp_df.rename(columns={"hhid_main": "hhid"})
    sub_df = sub_df.rename(columns={f"hhid_{rela}": "hhid"})

    combine_df = main_pp_df.merge(sub_df, on="hhid", how="right")
    combine_df = combine_df.drop(columns=[f"relationship_{rela}", "relationship_main"])

    if "_weight_main" in combine_df.columns:
        combine_df = combine_df.rename(columns={"_weight_main": "_weight"})
        combine_df = combine_df.drop(columns=[f"_weight_{rela}"])

    if not include_weights:
        combine_df = combine_df.drop(columns="_weight")
    
    if to_csv:
        combine_df.to_csv(os.path.join(processed_data, f"connect_main_{rela}.csv"), index=False)
    
    return combine_df

def get_weights_dict(hh_df_w, pp_df_w):
    re_dict = {}
    # Process HH weights
    hh_df_w["_weight"] = hh_df_w["wdhhwgt_sa3"].fillna(0) + hh_df_w["wehhwgt_sa3"].fillna(0)
    pp_df_w["_weight"] = pp_df_w["wdperswgt_sa3"].fillna(0) + pp_df_w["weperswgt_sa3"].fillna(0)
    re_dict["hh"] = dict(zip(hh_df_w["hhid"], hh_df_w["_weight"]))
    re_dict["pp"] = dict(zip(pp_df_w["persid"], pp_df_w["_weight"]))
    return re_dict


def add_weights_in_df(df, weights_dict, type="hh"):
    select_col = None
    dict_check = weights_dict[type]
    if type == "hh":
        check_cols = [x for x in df.columns if "hhid" in x]
        if len(check_cols) == 0:
            raise ValueError("No HHID to match with the weights")
        else:
            select_col = check_cols[0] # Don't know there will be mutiple but just incase, will select the first col
        
    elif type == "pp":
        check_cols = [x for x in df.columns if "persid" in x]
        if len(check_cols) == 0:
            raise ValueError("No persid to match with the weights")
        elif len(check_cols) == 1:
            select_col = check_cols[0]
        else:
            pref_val = "persid_main" # We will now use the weights of the main person
            select_col = pref_val if pref_val in check_cols else check_cols[0]
    else:
        raise ValueError("You pick wrong type for dict check")
    
    assert select_col is not None
    df["_weight"] = df.apply(lambda row: dict_check[row[select_col]], axis=1)
    return df


def add_converted_inc(pp_df):
    def process_inc(row):
        r_check = row["persinc"]
        val = None
        if "p.w." in r_check:
            r_check = r_check.replace("p.w.", "").replace(" ", "").replace("$", "")
            if "+" in r_check:
                r_check = r_check.replace("+", "")
            elif "-" in r_check:
                r_check = r_check.split("-")[0]
            else:
                raise ValueError(f"Dunno I never seen this lol {r_check}")
            val = int(r_check)
        elif "Zero" in r_check:
            val = 0
        elif "Negative" in r_check:
            val = -1
        elif "Missing" in r_check:
            val = -2
        else:
            raise ValueError(f"Dunno I never seen this lol {r_check}")
        return val
    
    pp_df["inc_dummy"] = pp_df.apply(process_inc, axis=1)
    return pp_df


def get_main_max_age(pp_df):
    # add the dummy inc to rank
    ls_hh_id = pp_df["hhid"].unique()
    for hh_id in ls_hh_id:
        print(hh_id)
        sub_df = pp_df[pp_df["hhid"]==hh_id]
        idx_max_age = sub_df["age"].idxmax()
        rela_max_age = sub_df.loc[idx_max_age]["relationship"]
        # CONFIRMED this will be Spouse or Others only
        pp_df.at[idx_max_age, "relationship"] = "Main"
        if rela_max_age != "Self":
            sub_sub_df = sub_df[sub_df["relationship"]=="Self"]
            idx_self = sub_sub_df.index[0]
            pp_df.at[idx_self, "relationship"] = rela_max_age
    return pp_df


def convert_pp_age_gr(pp_df, range_age=10, age_limit=100):
    check_dict = {}
    hold_min = None
    new_name = None
    for i in range(age_limit):
        if i % range_age == 0:
            hold_min = i
            new_name = f"{hold_min}-{hold_min+range_age-1}"
        check_dict[i] = new_name
    check_dict["others"] = f"{age_limit}+"
    
    def convert_age(row):
        if row["age"] in check_dict:
            return check_dict[row["age"]]
        else:
            return check_dict["others"]
        
    pp_df["age"] = pp_df.apply(convert_age, axis=1)
    return pp_df


def convert_hh_totvehs(hh_df, veh_limit=4):
    def convert_veh(row):
        if row["totalvehs"] < veh_limit:
            return str(row["totalvehs"])
        else:
            return f"{veh_limit}+"
    hh_df["totalvehs"] = hh_df.apply(convert_veh, axis=1)
    return hh_df


def convert_hh_inc(hh_df, check_states):
    def con_inc(row):
        hh_inc = row["hhinc"]
        # Confime hhinc always exist, it's float
        if hh_inc < 0:
            return "Negative income" #NOTE: None like this but exist in census, need to check whether this can be an issue
        elif hh_inc > 0:
            for state in check_states:
                bool_val = None
                if "$" in state:
                    state = state.replace(",", "").replace("$", "")
                    if "more" in state:
                        val = state.split(" ")[0]
                        bool_val = hh_inc >= int(val)
                        state = val + "+"
                    elif "-" in state:
                        state = state.split(" ")[0]
                        a, b = state.split("-")
                        bool_val = hh_inc >= int(a) and hh_inc <= int(b)
                    else:
                        raise ValueError(f"Dunno I never seen this lol {state}")
                if bool_val:
                    return state
        else:
            return "Nil income"
    hh_df["hhinc"] = hh_df.apply(con_inc, axis=1)
    return hh_df

def convert_hh_dwell(hh_df): # Removing the occupied rent free
    hh_df["owndwell"] = hh_df.apply(lambda r: "Something Else" if r["owndwell"] == "Occupied Rent-Free" else r["owndwell"], axis=1)
    return hh_df


def convert_hh_size(hh_df):
    hh_df["hhsize"] = hh_df.apply(lambda r: "8+" if r["hhsize"] >= 8 else str(r["hhsize"]), axis=1)
    return hh_df


def convert_all_hh_atts(hh_df, pp_df):
    hh_df = adding_pp_related_atts(hh_df, pp_df)
    hh_df = convert_hh_totvehs(hh_df)
    hh_df = convert_hh_inc(hh_df, check_states=LS_HH_INC)
    hh_df = convert_hh_dwell(hh_df)
    hh_df = convert_hh_size(hh_df)
    return hh_df


def main():
    # Import HH and PP samples (VISTA)
    # Pre-processing
    hh_df_raw = pd.read_csv(os.path.join(data_dir ,"H_VISTA_1220_SA1.csv"))
    pp_df_raw = pd.read_csv(os.path.join(data_dir, "P_VISTA_1220_SA1.csv"))
    weights_dict = get_weights_dict(hh_df_raw[["hhid", "wdhhwgt_sa3", "wehhwgt_sa3"]], pp_df_raw[["persid", "wdperswgt_sa3", "weperswgt_sa3"]])

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

