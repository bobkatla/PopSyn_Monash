"""
Now for each zone we will do adjustment

For a given zone (a vector), we can identify which att is pos and which is neg so we can replace
Inputs:
curr_syn - curr_syn of that zone (matrix) with counts
diff_census - diff from census with neg and pos (vector)
pool - data of all possible values (matrix) with counts
adjusted_atts - adjusted atts (list) so we need to maintain those combinations

Outputs: New syn_pop for that zone

We need to think about how to better this process, not with pairing and simple finding, maybe matrix opt
"""
import pandas as pd
import numpy as np
from PopSynthesis.Methods.IPSF.const import zone_field
from typing import List

def zone_adjustment(curr_syn_count: pd.DataFrame, diff_census: pd.Series, pool: pd.DataFrame, adjusted_atts: List[str]) -> pd.DataFrame:
    assert zone_field in curr_syn_count.columns
    assert zone_field in pool.columns

    print(diff_census)