"""
Some basic tools to process data
"""

import pandas as pd


def sample_from_full_pop(full_pop: pd.DataFrame, sample_rate:int) -> pd.DataFrame:
    N = len(full_pop)
    one_percent = int(N/100)
    # It is noted that with small samples, cannot ebtablish the edges
    seed_df = full_pop.sample(n = sample_rate * one_percent).copy()
    return seed_df


def main():
    NotImplemented


if __name__ == "__main__":
    main()