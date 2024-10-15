"""Run the CSP from a given syn HH (also SAA to adjust again)"""


import pandas as pd
from PopSynthesis.Methods.IPSF.const import data_dir
from PopSynthesis.Methods.IPSF.CSP.operations.convert_seeds import convert_seeds_to_pairs

def main():
    # TODO: is there anyway to use pp marg
    # get the syn hh
    hh_marg = pd.read_csv(data_dir / "hh_marginals_ipu.csv", header=[0, 1])
    hh_marg = hh_marg.drop(columns=hh_marg.columns[hh_marg.columns.get_level_values(0)=="sample_geog"][0])
    hh_seed = pd.read_csv(data_dir / "hh_sample_ipu.csv").drop(columns=["sample_geog"])
    pp_seed = pd.read_csv(data_dir / "pp_sample_ipu.csv").drop(columns=["sample_geog"])
    # process seed
    seed_pairs = convert_seeds_to_pairs(hh_seed, pp_seed, "serialno", "relationship", "Main")

    # create pools
    print(seed_pairs["Main-Grandparent"])

if __name__ == "__main__":
    main()