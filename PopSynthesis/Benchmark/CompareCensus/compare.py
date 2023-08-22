"""
Compare with the census
"""

import pandas as pd
import math


def compare_RMS_census(marg_census:pd.Series, marg_syn:pd.Series):
    assert marg_census.index.names == marg_syn.index.names
    hold = 0
    for k in marg_census.index:
        census_val = marg_census[k]
        syn_val = marg_syn[k] if k in marg_syn.index else 0
        hold += (census_val - syn_val)**2
    return math.sqrt(hold)
