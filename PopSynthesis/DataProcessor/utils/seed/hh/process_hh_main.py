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
