"""Run the CSP from a given syn HH (also SAA to adjust again)"""


import pandas as pd
import pickle
from PopSynthesis.Methods.IPSF.const import data_dir, POOL_SIZE, processed_dir
from PopSynthesis.Methods.IPSF.CSP.operations.convert_seeds import (
    convert_seeds_to_pairs,
    pair_states_dict,
)
from PopSynthesis.Methods.IPSF.utils.pool_utils import create_pool


def main():
    # TODO: is there anyway to use pp marg
    # get the data
    with open(processed_dir / "dict_pool_pairs.pickle", "rb") as handle:
        pools_ref = pickle.load(handle)
    print(pools_ref)


if __name__ == "__main__":
    main()
