from collections import defaultdict
from PopSynthesis.Methods.connect_HH_PP.scripts.const import LS_GR_RELA, HANDLE_THE_REST_RELA
import polars as pl

def check_rela_gb(gb_df):
    for hhid, rela_gr in zip(gb_df.index, gb_df):
        check_dict = defaultdict(lambda: 0)
        for i in rela_gr:
            check_dict[i] += 1
        if check_dict["Self"] == 0:
            # print(hhid)
            print([f"{x} - {y}" for x, y in check_dict.items() if x != "Self"])
        elif check_dict["Self"] > 1:
            print("NOOOOOOOOOO", hhid, rela_gr)

    
def process_not_accept_values(pp_df):
    # Remove not accept value
    # At the moment we remove Null and Missing for income
    pp_df = pp_df.drop_nulls()
    pp_df_missing = pp_df.filter(pl.col("persinc")=="Missing/Refused")
    to_rm_hhid = list(pp_df_missing["hhid"].unique())
    pp_df = pp_df.filter(~pl.col("hhid").is_in(to_rm_hhid))
    return pp_df


def convert_simple_income(income_str):
    if "Negative" in income_str:
        return -1
    elif "Missing" in income_str:
        # This should not happen as we will filter no income
        return -2
    elif "Zero" in income_str:
        return 0
    elif "-" in income_str:
        return int(income_str.split("-")[0].replace("$", ""))
    elif "+" in income_str:
        return 2000
    else:
        raise ValueError("Weird")


def process_rela(pp_df: pl.DataFrame):
    # To handle relationship, generally we based on income, age and gender
    # First we need to make sure each HH has 1 Self
    income_col = pl.col("persinc")
    pp_df.with_columns(pl.when)
    gb_df_rela_list = pp_df.groupby("hhid").agg(pl.col("relationship"))
    # First replace the first person to be Main, there should be no Self left

    # return pp_df
