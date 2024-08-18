"""
Compare with the census
"""

import pandas as pd
import math


def compare_RMS_census(marg_census: pd.Series, marg_syn: pd.Series):
    ls_census_atts = list(marg_census.index.get_level_values(0).unique())
    ls_syn_atts = list(marg_syn.index.get_level_values(0).unique())
    assert ls_census_atts == ls_syn_atts
    n = marg_census[ls_census_atts[0]].sum()
    hold = 0
    for k in marg_census.index:
        census_val = marg_census[k]
        syn_val = marg_syn[k] if k in marg_syn.index else 0
        hold += ((census_val - syn_val) / n) ** 2
    return math.sqrt(hold)
