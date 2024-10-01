"""Main place to run SAA for households synthesis"""


import pandas as pd
import pickle
from PopSynthesis.Methods.IPSF.const import data_dir, processed_dir, POOL_SIZE
from PopSynthesis.Methods.IPSF.utils.pool_utils import create_pool
from PopSynthesis.Methods.IPSF.SAA.main import SAA


def run_main() -> None:
    hh_marg = pd.read_csv(data_dir / "hh_marginals_ipu.csv", header=[0, 1])
    hh_seed = pd.read_csv(data_dir / "hh_sample_ipu.csv")
    with open(processed_dir / "dict_hh_states.pickle", "rb") as handle:
        hh_att_state = pickle.load(handle)
    order_adjustment = [
        "hhsize",
        "hhinc",
        "totalvehs",
        "dwelltype",
        "owndwell",
    ]
    pool = create_pool(seed=hh_seed, state_names=hh_att_state, pool_sz=int(POOL_SIZE))
    hh_marg = hh_marg.head(5)
    saa = SAA(hh_marg, hh_seed, order_adjustment, hh_att_state, pool)
    final_syn_pop = saa.run()
    print(final_syn_pop)


if __name__ == "__main__":
    run_main()
