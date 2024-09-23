""" 
This will be the main for running SAA

SAA will take in the marginals and the seed data and output the final synthetic population
Will thinking of doing in Polars
"""

import pandas as pd

from PopSynthesis.Methods.IPSF.const import POOL_SIZE
from PopSynthesis.Methods.IPSF.SAA.operations.general import process_raw_ipu_init, adjust_atts_state_match_census
from typing import List, Dict

class SAA:
    def __init__(self, marginal_raw: pd.DataFrame, seed_raw: pd.DataFrame, ordered_to_adjust_atts:List[str], att_states: Dict[str, List[str]], pool_sz: int = POOL_SIZE) -> None:
        self.ordered_atts = ordered_to_adjust_atts
        self.known_att_states = att_states
        self.init_required_inputs(marginal_raw, seed_raw)

    def init_required_inputs(self, marginal_raw: pd.DataFrame, seed_raw: pd.DataFrame):
        converted_segment_marg, converted_seed = process_raw_ipu_init(marginal_raw, seed_raw)
        self.seed = converted_seed
        self.segmented_marg = converted_segment_marg

    def run(self) -> pd.DataFrame:
        # Output the synthetic population, the main point
        curr_syn_pop = None
        adjusted_atts = []
        pool = self.seed # change later
        for att in self.ordered_atts:
            sub_census = self.segmented_marg[att].reset_index()
            curr_syn_pop = adjust_atts_state_match_census(att, curr_syn_pop, sub_census, adjusted_atts, pool)
            adjusted_atts.append(att)
        return curr_syn_pop
