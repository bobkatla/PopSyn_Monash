"""This is specific to the SAA method"""


import pandas as pd


def process_main_other(main_pp_df, sub_df, rela, include_weights=True):
    assert len(main_pp_df["relationship"].unique()) == 1  # It is Main
    assert (
        len(sub_df["relationship"].unique()) == 1
    )  # It is the relationship we checking
    # Change the name to avoid confusion
    main_pp_df = main_pp_df.add_suffix("_main", axis=1)
    sub_df = sub_df.add_suffix(f"_{rela}", axis=1)
    main_pp_df = main_pp_df.rename(columns={"hhid_main": "hhid"})
    sub_df = sub_df.rename(columns={f"hhid_{rela}": "hhid"})

    combine_df = main_pp_df.merge(sub_df, on="hhid", how="right")
    combine_df = combine_df.drop(columns=[f"relationship_{rela}", "relationship_main"])

    if "_weight_main" in combine_df.columns:
        combine_df = combine_df.rename(columns={"_weight_main": "_weight"})
        combine_df = combine_df.drop(columns=[f"_weight_{rela}"])

    if not include_weights:
        combine_df = combine_df.drop(columns="_weight")

    return combine_df
