"""Run the CSP from a given syn HH (also SAA to adjust again)"""


import pickle
import pandas as pd
from PopSynthesis.Methods.IPSF.const import processed_dir, output_dir, PP_ATTS, NOT_INCLUDED_IN_BN_LEARN, zone_field
from PopSynthesis.Methods.IPSF.CSP.operations.extra_filters import filter_mismatch_hhsz
from PopSynthesis.Methods.IPSF.CSP.operations.sample_from_pairs import sample_matching_from_pairs


def main():
    # TODO: is there anyway to use pp marg
    # get the data
    HHID = "hhid"
    syn_hh = pd.read_csv(output_dir / "SAA_output_HH_again.csv", index_col=0).reset_index(drop=True)
    syn_hh["hhid"] = syn_hh.index
    with open(processed_dir / "dict_pool_pairs.pickle", "rb") as handle:
        pools_ref = pickle.load(handle)
    pp_atts = list(set(PP_ATTS) - set(NOT_INCLUDED_IN_BN_LEARN))
    hh_atts = [x for x in syn_hh.columns if x not in [zone_field, HHID]]
    all_rela = [x.split("-")[-1] for x in pools_ref.keys()]
    # rename the HH-Main so the so Main match the rest
    rename_main = {x: f"{x}_Main" for x in pp_atts}
    pools_ref["HH-Main"] = pools_ref["HH-Main"].rename(columns=rename_main)
    pools_ref["HH-Main"] = filter_mismatch_hhsz(pools_ref["HH-Main"], "hhsize", all_rela)
    
    main_pp, removed_syn, kept_syn = sample_matching_from_pairs(syn_hh, HHID, pools_ref["HH-Main"], hh_atts, list(rename_main.values()) + all_rela)
    # Test for the HH and
    print(main_pp)
    print(removed_syn)
    print(kept_syn)


if __name__ == "__main__":
    main()
