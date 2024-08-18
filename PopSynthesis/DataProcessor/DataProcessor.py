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
    LS_GR_RELA,
    LS_HH_INC,
)
from PopSynthesis.DataProcessor.utils.general_utils import find_file
from PopSynthesis.DataProcessor.utils.seed.hh.process_general_hh import (
    convert_hh_totvehs,
    convert_hh_size,
    convert_hh_dwell,
    convert_hh_inc,
)
from PopSynthesis.DataProcessor.utils.seed.pp.process_relationships import process_rela, process_not_accept_values
from PopSynthesis.DataProcessor.utils.seed.pp.process_main_others import process_main_other
from PopSynthesis.DataProcessor.utils.seed.pp.convert_age import convert_pp_age_gr, get_main_max_age
from PopSynthesis.DataProcessor.utils.seed.pp.convert_inc import add_converted_inc
import polars as pl


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
        # Steps to make sure all persons belongs to households, and households size equal number of persons

    def process_households_seed(self):
        # Import the hh seed data
        hh_file = find_file(base_path=self.raw_data_path, filename=hh_seed_file)
        raw_hh_seed = pl.read_csv(hh_file)
        hh_df = raw_hh_seed[HH_ATTS]
        hh_df = convert_hh_totvehs(hh_df)
        hh_df = convert_hh_inc(hh_df, check_states=LS_HH_INC)
        hh_df = convert_hh_dwell(hh_df)
        hh_df = convert_hh_size(hh_df)
        return hh_df

    def process_persons_seed(self):
        pp_file = find_file(base_path=self.raw_data_path, filename=pp_seed_file)
        raw_hh_seed = pl.read_csv(pp_file)
        pp_df = raw_hh_seed[PP_ATTS]
        pp_df = process_not_accept_values(pp_df)
        pp_df = process_rela(pp_df)
        print(pp_df)
        # pp_df = get_main_max_age(pp_df)
        # pp_df = convert_pp_age_gr(pp_df)
        # pp_df = add_converted_inc(pp_df)


    def process_all_census(self):
        NotImplemented

    def process_households_census(self):
        NotImplemented

    def process_persons_census(self):
        NotImplemented


if __name__ == "__main__":
    a = DataProcessorGeneric(raw_data_dir, processed_data_dir, output_dir)
    a.process_persons_seed()
