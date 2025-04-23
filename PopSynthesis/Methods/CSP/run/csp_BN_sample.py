"""Using BN to sample should quick for each step, but the conditional need to think abit"""

# need the possible states for each atts, also including the min and max for each count (n_rela)
# process the given dict of conditionals (seed matching with count) to do the fitting for BN
# dertermine the n_rela for each hh using BN (merge later), need conditional sampling and then merge
# 
import pandas as pd

from typing import Dict, Union, List

def sample_rela_BN(hh_df: pd.DataFrame, final_conditonals: pd.DataFrame, hhsz: str, relationship: str, possible_states:Dict[str, List[str]]=None) -> pd.DataFrame:
    print("geee")
    raise NotImplementedError("This function is not implemented yet")