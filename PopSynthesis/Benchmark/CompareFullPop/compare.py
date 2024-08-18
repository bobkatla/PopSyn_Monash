"""
This is used when we have the full population (not possible in real life) and compare with it.
This should be more correct but not realistic/ realiable
Often we just use the seed data as the full pop
"""
from PopSynthesis.Benchmark.legacy.checker import update_SRMSE
import pandas as pd
import time
import math


def full_pop_SRMSE(actual: pd.DataFrame, pred: pd.DataFrame) -> float:
    return update_SRMSE(actual=actual, pred=pred)


def SRMSE_based_on_counts(actual_cou: pd.Series, pred_cou: pd.Series) -> float:
    assert pred_cou.index.names == actual_cou.index.names
    start_time = time.time()
    # key step: we need to organize the cols for 2 df to match so the val count later is correct
    actual_sum = actual_cou.sum()
    pred_sum = pred_cou.sum()

    hold = 0
    for com in actual_cou.index:
        actual_freq = actual_cou[com] / actual_sum
        pred_val = pred_cou[com] if com in pred_cou else 0
        pred_freq = pred_val / pred_sum
        hold += (actual_freq - pred_freq) ** 2

    for com in pred_cou.index:
        if com not in actual_cou.index:
            hold += (pred_cou[com] / pred_sum) ** 2

    result = math.sqrt(hold * (len(actual_cou)))
    duration = time.time() - start_time
    print(f"Calculated the SRMSE in {duration} seconds")
    return result


def main():
    NotImplemented


if __name__ == "__main__":
    main()
