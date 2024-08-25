"""Main place to run SAA for households synthesis"""


import pandas as pd
import pickle
from PopSynthesis.Methods.IPSF.const import data_dir, processed_dir, output_dir
from PopSynthesis.Methods.IPSF.SAA.main import SAA


def run_main() -> None:
    hh_marg = pd.read_csv(data_dir / "hh_marginals_ipu.csv", header=[0,1])
    hh_seed = pd.read_csv(data_dir / "hh_sample_ipu.csv")
    with open(processed_dir / "dict_hh_states.pickle", "rb") as handle:
        hh_att_state = pickle.load(handle)
    order_adjustment = ["hhsize", "hhinc", "totalvehs", "dwelltype", "owndwell"]
    saa = SAA(hh_marg, hh_seed, order_adjustment, hh_att_state)
    final_syn_pop = saa.run()


if __name__ == "__main__":
    run_main()