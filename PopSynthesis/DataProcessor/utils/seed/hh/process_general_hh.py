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
            return "Negative income"  # NOTE: None like this but exist in census, need to check whether this can be an issue
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


def convert_hh_dwell(hh_df):  # Removing the occupied rent free
    hh_df["owndwell"] = hh_df.apply(
        lambda r: "Something Else"
        if r["owndwell"] == "Occupied Rent-Free"
        else r["owndwell"],
        axis=1,
    )
    return hh_df


def convert_hh_size(hh_df):
    hh_df["hhsize"] = hh_df.apply(
        lambda r: "8+" if r["hhsize"] >= 8 else str(r["hhsize"]), axis=1
    )
    return hh_df
