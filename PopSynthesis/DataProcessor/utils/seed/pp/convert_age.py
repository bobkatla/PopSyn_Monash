
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

