"""Run the CSP from a given syn HH (also SAA to adjust again)"""


import pickle
from PopSynthesis.Methods.IPSF.const import processed_dir, PP_ATTS, NOT_INCLUDED_IN_BN_LEARN
from PopSynthesis.Methods.IPSF.CSP.operations.extra_filters import filter_mismatch_hhsz


def main():
    # TODO: is there anyway to use pp marg
    # get the data
    with open(processed_dir / "dict_pool_pairs.pickle", "rb") as handle:
        pools_ref = pickle.load(handle)
    pp_atts = list(set(PP_ATTS) - set(NOT_INCLUDED_IN_BN_LEARN))
    all_rela = [x.split("-")[-1] for x in pools_ref.keys()]
    # rename the HH-Main so the so Main match the rest
    rename_main = {x: f"{x}_Main" for x in pp_atts}
    pools_ref["HH-Main"] = pools_ref["HH-Main"].rename(columns=rename_main)
    pools_ref["HH-Main"] = filter_mismatch_hhsz(pools_ref["HH-Main"], "hhsize", all_rela)
    print(pools_ref["HH-Main"])


if __name__ == "__main__":
    main()
