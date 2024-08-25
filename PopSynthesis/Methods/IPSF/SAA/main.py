""" 
This will be the main for running SAA

SAA will take in the marginals and the seed data and output the final synthetic population
Will thinking of doing in Polars
"""

import pandas as pd

from PopSynthesis.Methods.IPSF.const import POOL_SIZE
from typing import List, Dict, Union

class SAA:
    def __init__(self, seed: pd.DataFrame, marginal: pd.DataFrame, pool_sz: int = POOL_SIZE, ordered_to_adjust_atts=List[str]) -> None:
        self.seed = seed
        self.marginal = marginal
        self.ordered_atts = ordered_to_adjust_atts

    def create_pool(self, known_atts_states: Union[None, Dict[str, List[str]]]) -> pd.DataFrame:
        # this creates the pool based on the seed data and also the dict if given
        self.pool = None
        pass

    def adjust_the_atts(self):
        NotImplemented

    def run(self) -> pd.DataFrame:
        # Output the synthetic population, the main point
        NotImplemented