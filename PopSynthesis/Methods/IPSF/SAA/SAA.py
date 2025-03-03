""" 
This will be the main for running SAA

SAA will take in the marginals and the seed data and output the final synthetic population
Will thinking of doing in Polars
"""

import pandas as pd
import polars as pl

from PopSynthesis.Methods.IPSF.const import output_dir
from PopSynthesis.Methods.IPSF.SAA.operations.general import (
    process_raw_ipu_marg,
    adjust_atts_state_match_census,
)
from typing import List


class SAA:
    def __init__(
        self,
        marginal_raw: pd.DataFrame,
        considered_atts: List[str],
        ordered_to_adjust_atts: List[str],
        count_pool: pl.DataFrame,
    ) -> None:
        self.ordered_atts_to_adjust = ordered_to_adjust_atts
        self.considered_atts = considered_atts
        self.pool = count_pool
        self.init_required_inputs(marginal_raw)

    def init_required_inputs(self, marginal_raw: pd.DataFrame):
        converted_segment_marg = process_raw_ipu_marg(
            marginal_raw, atts=self.considered_atts
        )
        self.segmented_marg = converted_segment_marg

    def run(self, output_each_step: bool = False, extra_name: str = "") -> pl.DataFrame:
        # Output the synthetic population, the main point
        curr_syn_pop = None
        adjusted_atts = []
        for att in self.ordered_atts_to_adjust:
            sub_census = self.segmented_marg[att].reset_index()
            sub_census = pl.from_pandas(sub_census)
            curr_syn_pop = adjust_atts_state_match_census(
                att, curr_syn_pop, sub_census, adjusted_atts, self.pool
            )
            adjusted_atts.append(att)
            if output_each_step:
                curr_syn_pop.write_csv(
                    output_dir / f"syn_pop_adjusted_{att}{extra_name}.csv"
                )
        return curr_syn_pop
