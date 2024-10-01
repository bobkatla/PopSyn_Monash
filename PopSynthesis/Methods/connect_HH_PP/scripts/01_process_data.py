import pandas as pd
from pathlib import Path
import pickle
import os

from PopSynthesis.Methods.connect_HH_PP.paras_dir import processed_data
from PopSynthesis.Methods.connect_HH_PP.scripts.const import NOT_INCLUDED_IN_BN_LEARN
from PopSynthesis.DataProcessor.DataProcessor import get_generic_generator
from PopSynthesis.DataProcessor.utils.seed.pp.process_relationships import (
    AVAILABLE_RELATIONSHIPS,
)


output_dir = Path(__file__).parent.parent.resolve() / "output"
assert output_dir.exists()


def process_hh_main_person(
    hh_df: pd.DataFrame, main_pp_df: pd.DataFrame, include_weights: bool = True
):
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
    return combine_df


def process_main_other(
    main_pp_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    rela: str,
    include_weights: bool = True,
):
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


def main():
    # Import HH and PP samples (VISTA)
    data_processer = get_generic_generator(output_dir)
    hh_df = data_processer.hh_seed_data
    pp_df = data_processer.pp_seed_data

    # return dict statenames for hh
    dict_hh_state_names = {
        hh_cols: list(hh_df[hh_cols].unique())
        for hh_cols in hh_df.columns
        if hh_cols not in AVAILABLE_RELATIONSHIPS
        and hh_cols not in NOT_INCLUDED_IN_BN_LEARN
    }
    with open(os.path.join(processed_data, "dict_hh_states.pickle"), "wb") as handle:
        pickle.dump(dict_hh_state_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return dict statenames for pp
    dict_pp_state_names = {
        pp_cols: list(pp_df[pp_cols].unique())
        for pp_cols in pp_df.columns
        if pp_cols not in NOT_INCLUDED_IN_BN_LEARN
    }
    with open(os.path.join(processed_data, "dict_pp_states.pickle"), "wb") as handle:
        pickle.dump(dict_pp_state_names, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # This part is to create the just simple converted samples from hh and pp
    # We need to add the count of each rela here or at DATA processor
    # data_processer.output_seed("ori_sample_pp", "ori_sample_hh")
    # hh_df = pd.read_csv(os.path.join(processed_data, "ori_sample_hh.csv"))

    # process hh_main
    main_pp_df = pp_df[pp_df["relationship"] == "Main"]
    df_hh_main = process_hh_main_person(hh_df, main_pp_df, include_weights=False)
    df_hh_main.to_csv(os.path.join(processed_data, "connect_hh_main.csv"), index=False)

    for rela in AVAILABLE_RELATIONSHIPS:
        if rela != "Main":
            print(f"DOING {rela}")
            sub_df = pp_df[pp_df["relationship"] == rela]
            df_main_other = process_main_other(
                main_pp_df, sub_df, rela=rela, include_weights=False
            )
            df_main_other.to_csv(
                os.path.join(processed_data, f"connect_main_{rela}.csv"), index=False
            )


if __name__ == "__main__":
    main()
