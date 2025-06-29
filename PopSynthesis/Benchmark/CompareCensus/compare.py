"""
Compare with the census
"""
import pandas as pd
import numpy as np


def powered2_diff(census_data: pd.DataFrame, syn_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the powered 2 difference between census and synthetic data.
    """
    if not set(census_data.columns) == set(syn_data.columns):
        raise ValueError("Census data columns must match synthetic data columns.")
    if not set(census_data.index) == set(syn_data.index):
        raise ValueError("Census data index must match synthetic data index.")

    diff = (census_data - syn_data).pow(2)
    return diff


def get_RMSE(census_data: pd.DataFrame, syn_data: pd.DataFrame, return_type: str = "zonal") -> pd.Series:
    """
    Calculate the Root Mean Square Error (RMSE) between census and synthetic data.
    """
    diff = powered2_diff(census_data, syn_data)
    if return_type == "zonal":
        return diff.mean(axis=1).apply(np.sqrt)
    elif return_type == "attribute":
        return diff.mean(axis=0).apply(np.sqrt)