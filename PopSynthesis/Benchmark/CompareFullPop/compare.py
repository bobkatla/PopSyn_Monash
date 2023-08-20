"""
This is used when we have the full population (not possible in real life) and compare with it.
This should be more correct but not realistic/ realiable
Often we just use the seed data as the full pop
"""
from PopSynthesis.Benchmark.legacy.checker import update_SRMSE
import pandas as pd


def full_pop_SRMSE(actual: pd.DataFrame, pred: pd.DataFrame) -> float:
    return update_SRMSE(actual=actual, pred=pred)


def main():
    NotImplemented


if __name__ == "__main__":
    main()