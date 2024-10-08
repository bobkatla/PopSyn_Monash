"""Main place to run SAA for households synthesis"""


import pandas as pd
import pickle
from PopSynthesis.Methods.IPSF.const import data_dir, processed_dir, output_dir, POOL_SIZE
from PopSynthesis.Methods.IPSF.utils.pool_utils import create_pool
from PopSynthesis.Methods.IPSF.SAA.main import SAA
import time


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
    hh_marg = hh_marg.head(3)
    hh_seed = hh_seed[order_adjustment]
    pool = create_pool(seed=hh_seed, state_names=hh_att_state, pool_sz=POOL_SIZE)
    saa = SAA(hh_marg, hh_seed, order_adjustment, hh_att_state, pool)

    start_time = time.time()
    ###
    final_syn_pop = saa.run()
    ###
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(rem, 60)       # 60 seconds in a minute
    print(f"Processing took {int(hours)}h-{int(minutes)}m-{seconds:.2f}s")
    
    print(final_syn_pop)
    final_syn_pop.to_csv(output_dir / "SAA_output_HH.csv")


if __name__ == "__main__":
    run_main()
