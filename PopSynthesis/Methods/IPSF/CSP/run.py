"""Run the CSP from a given syn HH (also SAA to adjust again)"""


import pandas as pd
import pickle
from PopSynthesis.Methods.IPSF.const import data_dir, POOL_SIZE, processed_dir
from PopSynthesis.Methods.IPSF.CSP.operations.convert_seeds import convert_seeds_to_pairs, pair_states_dict
from PopSynthesis.Methods.IPSF.utils.pool_utils import create_pool


def main():
    # TODO: is there anyway to use pp marg
    # get the data
    hh_marg = pd.read_csv(data_dir / "hh_marginals_ipu.csv", header=[0, 1])
    hh_seed = pd.read_csv(data_dir / "hh_sample_ipu.csv").drop(columns=["sample_geog"])
    pp_seed = pd.read_csv(data_dir / "pp_sample_ipu.csv").drop(columns=["sample_geog"])
    with open(processed_dir / "dict_hh_states.pickle", "rb") as handle:
        hh_att_state = pickle.load(handle)
    with open(processed_dir / "dict_pp_states.pickle", "rb") as handle:
        pp_att_state = pickle.load(handle)

    hh_marg = hh_marg.drop(columns=hh_marg.columns[hh_marg.columns.get_level_values(0)=="sample_geog"][0])
    # vars
    rela_col = "relationship"
    id_col = "serialno"
    main_rela = "Main"

    # process seed
    seed_pairs = convert_seeds_to_pairs(hh_seed, pp_seed, id_col, rela_col, main_rela)

    # create pools
    pools_ref = {}
    for pair_name, pair_seed in seed_pairs.items():
        name1, name2 = pair_name.split("-")
        ori_states_1 = hh_att_state if name1 == "HH" else pp_att_state
        ori_states_2 = pp_att_state # because the second one always people
        processed_states_ref = pair_states_dict(ori_states_1, ori_states_2, name1, name2)
        # we only need matching columns that we wish to process for BN
        # this excludes relationship and ids
        assert set(processed_states_ref.keys()) <= set(pair_seed.columns)
        filtered_seed = pair_seed[list(processed_states_ref.keys())]
        pools_ref[pair_name] = create_pool(filtered_seed, state_names=processed_states_ref, pool_sz=POOL_SIZE)
        
    print(pools_ref)
    

if __name__ == "__main__":
    main()