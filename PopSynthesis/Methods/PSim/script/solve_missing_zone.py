"""
To generate SA1 we need to do with zero-cell issue,
one way (here) is to do BN and generate the seed for those area.
We can start with just random sample, but in the future,
each zone should be calibrated to fit with the census (or some other data)
"""

def _learn_BN(df_seed):
    # Should return the BN learnt from the data
    NotImplemented

def _sampling_BN(BN, ls_zone, ratio_dummy):
    # Should return a pd.DataFrame
    NotImplemented

def _extract_missing_zones(df_seed, df_census, zone_lev="SA1"):
    # Should return a list of missing zones
    NotImplemented

def main():
    print("Hey")
    # Import data
    df_seed = None
    df_census_SA1 = None
    # Extract missing zones
    ls_missing_zones = _extract_missing_zones(df_seed=df_seed, df_census=df_census_SA1, zone_lev="SA1")
    # Learn BN
    BN = _learn_BN(df_seed=df_seed)
    dummy_seed = _sampling_BN(BN, ls_zone=ls_missing_zones, ratio_dummy=0.2)
    # Combine dummy seed and original seed
    # Output the new seed, this will be the new input
    

if __name__ == "__main__":
    main()