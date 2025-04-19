"""Main placeholder to run_csp"""
from PopSynthesis.Methods.CSP.run.csp_sample import csp_sample_by_hh
from PopSynthesis.Methods.CSP.run.create_pool_pairs import create_pool_pairs
from PopSynthesis.Methods.CSP.run.process_pools_by_needs import process_original_pools


def run_csp(hh_df, configs):
    """Run CSP with the given hh df and configs"""
    # From config we can have the seed hh, seed pp, we constraint by hh_size
    sub_hh_seed = None
    sub_pp_seed = None
    hhid = None
    relationship = None
    hhsz = None
    ori_pools = create_pool_pairs(sub_hh_seed, sub_pp_seed, hhid, relationship)
    # If we use IPF we can just use the original pool pairs (as all samples exist)
    final_conditonals = process_original_pools(ori_pools, method="original")
    syn_pp = csp_sample_by_hh(hh_df, final_conditonals, hhsz, relationship)
    return syn_pp

