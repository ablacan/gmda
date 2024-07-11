"""
Script for the Correlation Score metric.
"""

# Imports
import numpy as np
from typing import Tuple, Optional

def pearson_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the Pearson correlation coefficient matrix between each pair of variables in two matrices.
    
    Parameters:
        x (np.ndarray): Matrix 1 with shape (nb_samples, vb_vars).
        y (np.ndarray): Matrix 2 with shape (nb_samples, nb_vars).
    
    Returns:
        np.ndarray: Matrix with shape (nb_vars, nb_vars) containing the Pearson correlation coefficients.
    """

    def standardize(a: np.ndarray) -> np.ndarray:
        """
        Standardizes the input matrix by removing the mean and scaling to unit variance.

        Parameters:
            a (np.ndarray): Input matrix to standardize.
        
        Returns:
            np.ndarray: Standardized matrix.
        """
        mean = np.mean(a, axis=0)
        std = np.std(a, axis=0)
        standardized = (a - mean) / std
        standardized[np.isnan(standardized)] = 0  # Replace NaNs resulting from zero std with zeros
        return standardized

    assert x.shape[0] == y.shape[0], "The number of samples (rows) must be the same in both matrices."

    # Standardize the input matrices
    x_standardized = standardize(x)
    y_standardized = standardize(y)

    # Compute Pearson correlation coefficient matrix
    correlation_matrix = np.dot(x_standardized.T, y_standardized) / x.shape[0]
    
    return correlation_matrix

def get_corr_error(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Computes the mean difference of the correlation matrices between two given matrices.
    
    Parameters:
        x (np.ndarray): Matrix 1 with shape (nb_samples, nb_vars).
        y (np.ndarray): Matrix 2 with shape (nb_samples, nb_vars).
    
    Returns:
        Tuple[float, np.ndarray]: Mean difference score and absolute difference matrix.
    """
    # Compute the correlation matrices for both inputs
    corr_x = pearson_correlation(x, x)
    corr_y = pearson_correlation(y, y)

    # Compute the absolute difference matrix
    diff_matrix = np.abs(corr_x - corr_y)

    # Calculate the mean difference score
    mean_diff_score = diff_matrix.mean() / 2

    return mean_diff_score, diff_matrix