"""
Tools for creating the pool

BN we may need to move the code else where later
"""
import pandas as pd

from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score, learn_para_BN
from pgmpy.sampling import BayesianModelSampling

from typing import Dict, List, Union


def create_pool(seed: pd.DataFrame, state_names: Union[None, Dict[str, List[str]]], pool_sz: int, special:bool=True):
    print("Learn BN")
    model = learn_struct_BN_score(seed, show_struct=False, state_names=state_names)
    model = learn_para_BN(model, seed, state_names=state_names)
    print("Doing the sampling")
    inference = BayesianModelSampling(model)
    pool = inference.forward_sample(size=pool_sz, show_progress=True)

    # Special case, I will fix this to be dynamic later
    if special:
        while "Negative income" not in list(pool["hhinc"].unique()):
            print(list(pool["hhinc"].unique()))
            print("Not yet have it, gotta sample negative inc again")
            pool = inference.forward_sample(size=pool_sz, show_progress=True)
    return pool

