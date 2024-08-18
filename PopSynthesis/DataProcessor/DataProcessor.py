"""
Contain the base class with basic feature, there will be children classes
"""
from pathlib import Path
from os import PathLike
from typing import Union
from PopSynthesis.DataProcessor.utils.const_files import (
    hh_seed_file,
    pp_seed_file,
    raw_data_dir,
    processed_data_dir,
    output_dir,
)
from PopSynthesis.DataProcessor.utils.const_process import (
    HH_ATTS,
    PP_ATTS,
    LS_HH_INC,
)
from PopSynthesis.DataProcessor.utils.general_utils import find_file
from PopSynthesis.DataProcessor.utils.seed.hh.process_general_hh import (
    convert_hh_totvehs,
    convert_hh_size,
    convert_hh_dwell,
    convert_hh_inc,
)
from PopSynthesis.DataProcessor.utils.seed.pp.process_relationships import (
    process_rela,
    process_not_accept_values,
)
from PopSynthesis.DataProcessor.utils.seed.pp.process_main_others import (
    process_main_other,
)
from PopSynthesis.DataProcessor.utils.seed.pp.convert_age import convert_pp_age_gr
import polars as pl
import pandas as pd


# Future will convert to polars, stick with pandas mainly for now


class DataProcessorGeneric:
    def __init__(
        self,
        raw_data_src: PathLike[Union[Path, str]],
        mid_processed_src: PathLike[Union[Path, str]],
        output_data_src: PathLike[Union[Path, str]],
    ) -> None:
        self.raw_data_path = Path(raw_data_src)
        self.mid_process_path = Path(mid_processed_src)
        self.output_data_path = Path(output_data_src)

    def process_all_seed(self):
        hh_df = self.process_households_seed()
        pp_df = self.process_persons_seed()
        # Steps to make sure all persons belongs to households, if not remove
        hhid_in_pp = list(pp_df["hhid"].unique())
        filtered_hh = hh_df[hh_df["hhid"].isin(hhid_in_pp)]
        print(f"Removed {len(hh_df) - len(filtered_hh)} households total")

        # households size equal number of persons
        sub_check_hhsz = filtered_hh[["hhid", "hhsize"]].set_index("hhid")
        sub_check_pp = pp_df.groupby("hhid")["persid"].apply(lambda x: list(x))
        check_combine = pd.concat([sub_check_hhsz, sub_check_pp], axis=1)

        def check_match_hhsz(r):
            hhsz_from_pp = len(r["persid"])
            hhsz_from_hh = r["hhsize"]
            if hhsz_from_hh == "8+":
                return hhsz_from_pp >= 8
            else:
                return hhsz_from_pp == int(hhsz_from_hh)

        check_combine.loc[:, ["cross_check"]] = check_combine.apply(
            check_match_hhsz, axis=1
        )
        assert check_combine["cross_check"].all()

        # Next we add weights

    def process_households_seed(self) -> pd.DataFrame:
        # Import the hh seed data
        hh_file = find_file(base_path=self.raw_data_path, filename=hh_seed_file)
        raw_hh_seed = pl.read_csv(hh_file)
        hh_df = raw_hh_seed[HH_ATTS]
        hh_df = convert_hh_totvehs(hh_df)
        hh_df = convert_hh_inc(hh_df, check_states=LS_HH_INC)
        hh_df = convert_hh_dwell(hh_df)
        hh_df = convert_hh_size(hh_df)
        return hh_df.to_pandas()

    def process_persons_seed(self) -> pd.DataFrame:
        pp_file = find_file(base_path=self.raw_data_path, filename=pp_seed_file)
        raw_hh_seed = pl.read_csv(pp_file)
        pp_df = raw_hh_seed[PP_ATTS]
        pp_df = process_not_accept_values(pp_df)
        pp_df = process_rela(pp_df)
        pp_df = convert_pp_age_gr(pp_df)
        return pp_df.to_pandas()

    def process_all_census(self):
        NotImplemented

    def process_households_census(self):
        NotImplemented

    def process_persons_census(self):
        NotImplemented


if __name__ == "__main__":
    a = DataProcessorGeneric(raw_data_dir, processed_data_dir, output_dir)
    a.process_all_seed()
