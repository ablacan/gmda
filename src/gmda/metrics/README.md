# Metrics

This module provides functions to compute the Pearson correlation matrix, the mean difference of correlation matrices between two given datasets, as well as the precision-recall metric based on k-NN computations.

# Correlation Function

## Functions

### `pearson_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray`

Computes the Pearson correlation coefficient matrix between each pair of variables in two matrices.

#### Parameters:
- `x` (np.ndarray): Matrix 1 with shape (nb_samples, nb_vars).
- `y` (np.ndarray): Matrix 2 with shape (nb_samples, nb_vars).

#### Returns:
- `np.ndarray`: Matrix with shape (nb_vars, nb_vars) containing the Pearson correlation coefficients.

### `mean_diff_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]`

Computes the mean difference of the correlation matrices between two given matrices. The lower the mean difference score, the more similar the correlation structures are between the two matrices.

#### Parameters:
- `x` (np.ndarray): Matrix 1 with shape (nb_samples, nb_vars).
- `y` (np.ndarray): Matrix 2 with shape (nb_samples, nb_vars).

#### Returns:
- `Tuple[float, np.ndarray]`: Mean difference score and absolute difference matrix.

## Example Usage

Here is an example of how to use the functions:

```python
import numpy as np
from your_module import pearson_correlation, mean_diff_corr

# Generate random matrices for demonstration
np.random.seed(0)
x = np.random.rand(100, 10)
y = np.random.rand(100, 10)

# Calculate Pearson correlation matrix
corr_matrix = pearson_correlation(x, y)
print("Pearson Correlation Matrix:")
print(corr_matrix)

# Calculate mean difference correlation
mean_diff_score, diff_matrix = mean_diff_corr(x, y)
print("\nMean Difference Score:", mean_diff_score)
print("Absolute Difference Matrix:")
print(diff_matrix)
```

# Precision and recall (PR)

Compute precision and recall between two datasets using k-Nearest Neighbors (k-NN).

## Parameters

- `real_data` (`np.ndarray`): The first dataset to compare (usually the real or reference dataset).
- `fake_data` (`np.ndarray`): The second dataset to compare (usually the generated or fake dataset).
- `nb_nn` (`list`, optional): A list specifying the number of neighbors used to estimate the data manifold. Default is `[3]`.

## Returns

- `tuple`: A tuple containing the precision and recall values.

## Example Usage

```python
import numpy as np

# Example data
real_data = np.random.rand(100, 10)  # 100 samples, 10 features each
fake_data = np.random.rand(100, 10)  # 100 samples, 10 features each

# Compute precision and recall
precision, recall = get_precision_recall(real_data, fake_data, nb_nn=[3])

print(f"Precision: {precision}")
print(f"Recall: {recall}")
```
