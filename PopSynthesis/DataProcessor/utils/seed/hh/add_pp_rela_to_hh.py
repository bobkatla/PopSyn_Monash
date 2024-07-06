
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

