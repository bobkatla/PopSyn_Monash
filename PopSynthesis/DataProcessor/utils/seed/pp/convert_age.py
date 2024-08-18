import polars as pl


def convert_pp_age_gr(pp_df, range_age=10, age_limit=100):
    pp_df = pp_df.to_pandas()
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
    return pl.from_pandas(pp_df)
