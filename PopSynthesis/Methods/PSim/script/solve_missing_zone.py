"""
To generate SA1 we need to do with zero-cell issue,
one way (here) is to do BN and generate the seed for those area.
We can start with just random sample, but in the future,
each zone should be calibrated to fit with the census (or some other data)
"""

import pandas as pd
import bnlearn as bn
from PopSynthesis.Methods.BN.oldBN.BN_old import sampling


def _learn_BN(df_seed):
    # Should return the BN learnt from the data
    black_ls = None
    return None


def _sampling_BN(BN, ls_zone, n):
    # Should return a pd.DataFrame
    ls_re = []
    

def _extract_missing_zones(ls_available:list[str], ls_zones:list[str], zone_lev:str):
    # Assuming they are the same type
    assert ls_available.dtype == ls_zones.dtype
    ls_available = list(ls_available)
    ls_zones = list(ls_zones)
    # Should return a list of missing zones
    ls_unavai = []
    for zone in ls_available:
        if zone not in ls_zones:
            ls_unavai.append(zone)
    assert len(ls_available) == len(ls_unavai) + len(ls_zones)
    return ls_unavai


def main():
    # Import data
    df_seed_H = pd.read_csv("../data/H_sample.csv")
    # df_seed_P = pd.read_csv("../data/p_sample.csv")

    name_zone_lev = "SA1"
    df_census = pd.read_csv(f"../data/census_{name_zone_lev}.csv")
    ls_all_zones = df_census[name_zone_lev].astype("Int64").unique()
    # Extract missing zones

    df_seed = df_seed_H
    ls_zones = df_seed[name_zone_lev].astype("Int64").unique()
    ls_missing_zones = _extract_missing_zones(ls_available=ls_all_zones, ls_zones=ls_zones ,zone_lev=name_zone_lev)
    # Learn BN
    BN = _learn_BN(df_seed=df_seed)
    dummy_seed = _sampling_BN(BN, ls_zone=ls_missing_zones, ratio_dummy=0.2)
    # Combine dummy seed and original seed
    # Output the new seed, this will be the new input
    

if __name__ == "__main__":
    main()