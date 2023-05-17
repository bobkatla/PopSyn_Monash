"""
Rejection loop to match the sampling with census
Termination criteria to end early if needed
"""

import pandas as pd


def check_sample(df_sample, sa_level, zone):
    # Check zone exist in sample
    if zone in df_sample[sa_level]:
        NotImplemented
    else: return []


def reject_sample(BN_model, n=0, evidence=None):
    if n == 0: return None
    else:
        NotImplemented


def err_check(final_df, ls_atts_census, control_df):
    ls_err_att = [("name in sample", err_point, [("state", different_census, )])]
    return ls_err_att


def reject_loop(BN_model, df_sample, census, sa_level, control_df):
    # Loop through each zone in census
    ls_df = []
    for zone in census[sa_level]: # change this later
        tot = census["Tot"] # need to find index of that zone as well

        # Search whether it exist in sample
        ini_df = check_sample(df_sample, sa_level, zone)
        if ini_df != []: ls_df.append(ini_df)
        extra_n = tot - len(ini_df)
        assert extra_n >= 0
        # Loop is here
        att_done = [] # this is to store atts we have optimised so we can use them as evidence again
        while (extra_n > 0):
            extra_df = reject_sample(BN_model, n=extra_n)
            if extra_df: ls_df.append(extra_df)
            final_df = pd.concat(ls_df)
            err_list = err_check(final_df, census[zone], control_df) # THIS IS WRONG BUT JUST LEFT IDEA HERE


def test():
    # testing
    NotImplemented


if __name__ == "__main__":
    test()
