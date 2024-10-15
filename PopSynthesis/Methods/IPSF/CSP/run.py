"""Run the CSP from a given syn HH (also SAA to adjust again)"""


import pickle
from PopSynthesis.Methods.IPSF.const import processed_dir, PP_ATTS, NOT_INCLUDED_IN_BN_LEARN


def main():
    # TODO: is there anyway to use pp marg
    # get the data
    with open(processed_dir / "dict_pool_pairs.pickle", "rb") as handle:
        pools_ref = pickle.load(handle)
    pp_atts = list(set(PP_ATTS) - set(NOT_INCLUDED_IN_BN_LEARN))
    print(pp_atts)
    # rename the HH-Main so the so Main match the rest


if __name__ == "__main__":
    main()
