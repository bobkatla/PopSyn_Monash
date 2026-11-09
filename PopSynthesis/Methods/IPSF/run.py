"""Main place to run the SAA method"""

from PopSynthesis.Methods.IPSF.const import (
    SAA_ODERED_ATTS_HH,
    CONSIDERED_ATTS_HH,
)
from PopSynthesis.Methods.IPSF.SAA.operations.wrapper_saa_run import saa_run
from PopSynthesis.Methods.IPSF.utils.condensed import condense_df
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Literal, Optional


def process_data_from_files(marg_file: Path, pool_file: Path, zone_field: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    marg = pd.read_csv(marg_file, header=[0, 1])
    sample_geog_columns = marg.columns[
        marg.columns.get_level_values(0) == "sample_geog"
    ]
    if len(sample_geog_columns) > 0:
        marg = marg.drop(columns=sample_geog_columns)

    zone_columns = marg.columns[marg.columns.get_level_values(0) == zone_field]
    if len(zone_columns) != 1:
        raise ValueError(
            f"Expected exactly one zone column named {zone_field!r}; "
            f"found {len(zone_columns)}."
        )
    marg = marg.set_index(zone_columns[0])
    pool = pd.read_csv(pool_file)
    pool = pool.drop(columns=["serialno", "sample_geog"], errors="ignore")
    return marg, pool


def run_saa(
        marg_file: Path,
        pool_file: Path,
        zone_field: str,
        output_file: Path,
        considered_atts: Optional[List[str]] = None,
        ordered_to_adjust_atts: Optional[List[str]] = None,
        max_run_time: int = 15,
        extra_rm_frac: float = 0.3,
        last_adjustment_order: Optional[List[str]] = None,
        output_each_step: bool = False,
        add_name_for_step_output: str = "",
        include_zero_cell_values: bool = False,
        randomly_add_last: Optional[List[str]] = None,
        meta_output_dir: Path = None,
        stochastic_mode: Literal["count_native", "legacy_exact"] = "count_native",
        ) -> None:
    considered_atts = list(
        CONSIDERED_ATTS_HH if considered_atts is None else considered_atts
    )
    ordered_to_adjust_atts = list(
        SAA_ODERED_ATTS_HH
        if ordered_to_adjust_atts is None
        else ordered_to_adjust_atts
    )
    marg, pool = process_data_from_files(marg_file, pool_file, zone_field)
    condensed_pool = condense_df(pool.astype(str), maintain_order=True)

    # saa run
    final_syn_hh, _ = saa_run(
        marg,
        condensed_pool,
        considered_atts=considered_atts,
        ordered_to_adjust_atts=ordered_to_adjust_atts,
        max_run_time=max_run_time,
        extra_rm_frac=extra_rm_frac,
        last_adjustment_order=last_adjustment_order or [],
        output_each_step=output_each_step,
        add_name_for_step_output=add_name_for_step_output,
        include_zero_cell_values=include_zero_cell_values,
        randomly_add_last=randomly_add_last or [],
        meta_output_dir=meta_output_dir if meta_output_dir is not None else output_file.parent / "meta",
        stochastic_mode=stochastic_mode,
    )

    final_syn_hh.write_csv(output_file)

