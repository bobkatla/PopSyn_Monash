import polars as pl


def get_weights_dict(hh_df_w: pl.DataFrame, pp_df_w: pl.DataFrame):
    re_dict = {}
    # Process HH weights
    hh_df_w["_weight"] = hh_df_w["wdhhwgt_sa3"].fillna(0) + hh_df_w[
        "wehhwgt_sa3"
    ].fillna(0)
    pp_df_w["_weight"] = pp_df_w["wdperswgt_sa3"].fillna(0) + pp_df_w[
        "weperswgt_sa3"
    ].fillna(0)
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
            select_col = check_cols[
                0
            ]  # Don't know there will be mutiple but just incase, will select the first col

    elif type == "pp":
        check_cols = [x for x in df.columns if "persid" in x]
        if len(check_cols) == 0:
            raise ValueError("No persid to match with the weights")
        elif len(check_cols) == 1:
            select_col = check_cols[0]
        else:
            pref_val = "persid_main"  # We will now use the weights of the main person
            select_col = pref_val if pref_val in check_cols else check_cols[0]
    else:
        raise ValueError("You pick wrong type for dict check")

    assert select_col is not None
    df["_weight"] = df.apply(lambda row: dict_check[row[select_col]], axis=1)
    return df
