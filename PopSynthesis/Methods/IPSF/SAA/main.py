""" 
This will be the main for running SAA

SAA will take in the marginals and the seed data and output the final synthetic population
Will thinking of doing in Polars
"""

import polars as pl

from PopSynthesis.Methods.IPSF.const import POOL_SIZE
from typing import List

class SAA:
    def __init__(self, seed: pl.DataFrame, marginal: pl.DataFrame, pool_sz: int = POOL_SIZE, ordered_to_adjust_atts=List[str]) -> None:
        self.seed = seed
        self.marginal = marginal
        self.ordered_atts = ordered_to_adjust_atts


    def create_pool(self) -> pl.DataFrame:
        # this creates the pool based on the seed data, generally we will want the count
        self.pool = None
        pass

    def adjust_the_atts(self)

    def run(self) -> pl.DataFrame:
        # Output the synthetic population, the main point
        