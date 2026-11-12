"""
Tools for creating the pool

BN we may need to move the code else where later
"""
import warnings

import pandas as pd

from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score, learn_para_BN
from pgmpy.sampling import BayesianModelSampling

from typing import Dict, List, Union


def create_pool(
    seed: pd.DataFrame,
    state_names: Union[None, Dict[str, List[str]]],
    pool_sz: int,
    special: bool = True,
    seed_value: int | None = None,
    required_states: Union[None, Dict[str, List[str]]] = None,
):
    """Legacy one-shot pool builder.

    New code should use the census-informed helpers in
    ``PopSynthesis.Methods.BN.utils.bn_pool``. The former unbounded retry for
    Negative income has intentionally been replaced by post-sample validation.
    """
    print("Learn BN")
    model = learn_struct_BN_score(seed, show_struct=False, state_names=state_names)
    model = learn_para_BN(model, seed, state_names=state_names)
    print("Doing the sampling")
    inference = BayesianModelSampling(model)
    pool = inference.forward_sample(
        size=pool_sz,
        seed=seed_value,
        show_progress=True,
    )

    required_states = dict(required_states or {})
    if special and "hhinc" in pool.columns:
        warnings.warn(
            "create_pool(special=True) is deprecated; use required_states or "
            "the census-informed BN pool builder instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        required_states.setdefault("hhinc", []).append("Negative income")
    for attribute, states in required_states.items():
        if attribute not in pool.columns:
            raise ValueError(f"Required-state attribute {attribute!r} is not in the pool")
        missing = set(states) - set(pool[attribute].astype(str).unique())
        if missing:
            raise RuntimeError(
                f"Sampled pool is missing required {attribute!r} states: "
                f"{sorted(missing)}. Use an explicit prior or a larger pool."
            )
    return pool
