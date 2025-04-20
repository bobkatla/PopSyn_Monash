"""Main placeholder to run_csp"""
from PopSynthesis.Methods.CSP.run.csp_sample import csp_sample_by_hh
from PopSynthesis.Methods.CSP.run.create_pool_pairs import create_pool_pairs
from PopSynthesis.Methods.CSP.run.process_pools_by_needs import process_original_pools
import pandas as pd
from typing import Dict,Union

def run_csp(hh_df: pd.DataFrame, configs: Dict[str, Union[str, pd.DataFrame]]) -> pd.DataFrame:
    """Run CSP with the given hh df and configs"""
    # From config we can have the seed hh, seed pp, we constraint by hh_size
    hh_seed = configs["hh_seed"]
    pp_seed = configs["pp_seed"]
    hhid = configs["hhid"]
    relationship = configs["relationship"]
    hhsz = configs["hh_size"]
    ori_pools = create_pool_pairs(hh_seed, pp_seed, hhid, relationship)
    # If we use IPF we can just use the original pool pairs (as all samples exist)
    final_conditonals = process_original_pools(ori_pools, method="original")
    syn_pp = csp_sample_by_hh(hh_df, final_conditonals, hhsz, relationship)
    return syn_pp

