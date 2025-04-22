"""Main placeholder to run_csp"""
from PopSynthesis.Methods.CSP.run.csp_sample import csp_sample_by_hh
from PopSynthesis.Methods.CSP.run.create_pool_pairs import create_pool_pairs
from PopSynthesis.Methods.CSP.run.process_pools_by_needs import process_original_pools
from PopSynthesis.Methods.CSP.const import ZONE_ID, HHID
import pandas as pd
from typing import Dict,Union


def inflate_based_on_total(df, target_col: str) -> pd.DataFrame:
    assert target_col in df.columns, "The dataframe must contain the target column"
    # Repeat rows based on column values
    df_repeated = df.loc[df.index.repeat(df[target_col])].reset_index(drop=True)

    # Drop the column
    df_repeated = df_repeated.drop(columns=target_col)
    return df_repeated


def run_csp(hh_df: pd.DataFrame, configs: Dict[str, Union[str, pd.DataFrame]]) -> pd.DataFrame:
    """Run CSP with the given hh df and configs"""
    # From config we can have the seed hh, seed pp, we constraint by hh_size
    hh_df = inflate_based_on_total(hh_df, "total")
    # add hhid
    hh_df[HHID] = hh_df.reset_index(drop=True).index + 1
    hh_seed = configs["hh_seed"]
    pp_seed = configs["pp_seed"]
    hhid = configs["hhid"]
    relationship = configs["relationship"]
    hhsz = configs["hh_size"]
    ori_pools = create_pool_pairs(hh_seed, pp_seed, hhid, relationship)
    # If we use IPF we can just use the original pool pairs (as all samples exist)
    final_conditonals = process_original_pools(ori_pools, method="original")
    # csp_sample_by_hh(hh_df.drop(columns=[ZONE_ID]), final_conditonals, hhsz, relationship)
    final_syn_pp = []
    for zid in hh_df[ZONE_ID].unique():
        print(f"Processing zone {zid}")
        syn_pp = csp_sample_by_hh(hh_df[hh_df[ZONE_ID]==zid].drop(columns=[ZONE_ID]), final_conditonals, hhsz, relationship)
        syn_pp[ZONE_ID] = zid
        final_syn_pp.append(syn_pp)
    return pd.concat(final_syn_pp, ignore_index=True)

