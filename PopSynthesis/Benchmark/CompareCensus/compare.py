"""
Compare with the census
"""
import numpy as np
from typing import Tuple, List, Union


def powered2_diff(census_matrix: np.ndarray, syn_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the powered 2 difference between census and synthetic data.
    Both inputs are numpy arrays with the same shape.
    """
    if census_matrix.shape != syn_matrix.shape:
        raise ValueError(f"Census matrix shape {census_matrix.shape} must match synthetic matrix shape {syn_matrix.shape}.")
    
    # Calculate powered 2 difference using numpy
    diff_matrix = np.power(census_matrix - syn_matrix, 2)
    return diff_matrix


def get_RMSE(census_matrix: np.ndarray, syn_matrix: np.ndarray, return_type: str = "zonal") -> np.ndarray:
    """
    Calculate the Root Mean Square Error (RMSE) between census and synthetic data.
    Both inputs are numpy arrays with the same shape.
    
    Args:
        census_matrix: numpy array of census data
        syn_matrix: numpy array of synthetic data  
        return_type: "zonal" (mean across columns) or "attribute" (mean across rows)
    
    Returns:
        numpy array of RMSE values
    """
    diff_matrix = powered2_diff(census_matrix, syn_matrix)
    
    if return_type == "zonal":
        # Mean across columns (axis=1), then sqrt
        rmse_values = np.sqrt(np.mean(diff_matrix, axis=1))
        return rmse_values
    elif return_type == "attribute":
        # Mean across rows (axis=0), then sqrt
        rmse_values = np.sqrt(np.mean(diff_matrix, axis=0))
        return rmse_values
    else:
        raise ValueError(f"return_type must be 'zonal' or 'attribute', got {return_type}")


def validate_matrix_inputs(census_matrix: np.ndarray, syn_matrix: np.ndarray) -> None:
    """
    Validate that the input matrices are compatible for comparison.
    
    Args:
        census_matrix: numpy array of census data
        syn_matrix: numpy array of synthetic data
    
    Raises:
        ValueError: if matrices are incompatible
    """
    if not isinstance(census_matrix, np.ndarray):
        raise ValueError("census_matrix must be a numpy array")
    if not isinstance(syn_matrix, np.ndarray):
        raise ValueError("syn_matrix must be a numpy array")
    
    if census_matrix.shape != syn_matrix.shape:
        raise ValueError(f"Census matrix shape {census_matrix.shape} must match synthetic matrix shape {syn_matrix.shape}")
    
    if census_matrix.size == 0 or syn_matrix.size == 0:
        raise ValueError("Matrices cannot be empty")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(census_matrix)) or np.any(np.isnan(syn_matrix)):
        raise ValueError("Matrices contain NaN values")
    if np.any(np.isinf(census_matrix)) or np.any(np.isinf(syn_matrix)):
        raise ValueError("Matrices contain infinite values")