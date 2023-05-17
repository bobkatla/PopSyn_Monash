"""
File to double check after we got simple data
Mostly will the case of zero-cell (data exists in the census but not in sample)
--> we can aggregate or remove --> aggregate, maybe moveup 
Also we have the issue that in the sample, some code (sa1 and sa2) are old codes, just maybe has new code for 2021 now

Something todo, defo:
1. Taking only the Metro Mel (Greater Mel and Greater Geelong), they do exist all in VISTA
2. Check what SA don't exist in sample, alot sa1 yes, but maybe not sa2
3. Deal with zero cell (aggre how, maybe just combine with nearest SA)
"""

import pandas as pd


def main():
    df_census = pd.read_csv("./data/census_SA2.csv")
    df_H = pd.read_csv("./data/H_sample.csv")
    count=0
    for z in df_census["SA2"]:
        if z not in df_H["SA2"]:
            print(z, "not")
            count += 1
    print(count)


if __name__ == "__main__":
    main()