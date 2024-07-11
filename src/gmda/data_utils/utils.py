import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Callable, Optional, Tuple
import random
import time as t
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, QuantileTransformer

# SEED
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Dataloader reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() # should be 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

# Functions
def standardize_train(data, as_tensors=True):
    """
    Standardizes the training data using StandardScaler.

    Parameters:
        data (numpy.ndarray): Input data to be standardized.
        as_tensors (bool, optional): Whether to return the standardized data as a PyTorch tensor. Default is True.

    Returns:
        tuple: 
            standardized_data (numpy.ndarray or torch.Tensor): Standardized training data.
            scaler (StandardScaler): Fitted StandardScaler object.
    """
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Fit the scaler to the data and transform
    standardized_data = scaler.fit_transform(data)
    
    # Flatten the data if it contains only one feature
    if standardized_data.shape[1] == 1:
        standardized_data = standardized_data.flatten()
    
    # Convert to PyTorch tensor if required
    if as_tensors:
        standardized_data = torch.from_numpy(standardized_data).float()

    return standardized_data, scaler

def standardize_test(data, scaler, as_tensors=True):
    """
    Standardizes the test data using a pre-fitted StandardScaler.

    Parameters:
        data (numpy.ndarray): Input data to be standardized.
        scaler (StandardScaler): Pre-fitted StandardScaler object.
        as_tensors (bool, optional): Whether to return the standardized data as a PyTorch tensor. Default is True.

    Returns:
        tuple: 
            standardized_data (numpy.ndarray or torch.Tensor): Standardized test data.
            scaler (StandardScaler): The same StandardScaler object.
    """
    # Transform the data using the pre-fitted scaler
    standardized_data = scaler.transform(data)
    
    # Flatten the data if it contains only one feature
    if standardized_data.shape[1] == 1:
        standardized_data = standardized_data.flatten()
    
    # Convert to PyTorch tensor if required
    if as_tensors:
        standardized_data = torch.from_numpy(standardized_data).float()

    return standardized_data, scaler


class RandTransform():
    def __init__(self):
        self._help_ = "This object randomly transforms the input data as a linear combination of all features."

    def fit(self, X):
        self.WEIGHTS_ = np.random.randn(X.shape[1], X.shape[1])
        self.WEIGHTS_ = gram_schmidt(self.WEIGHTS_)

        if not np.allclose(np.dot(self.WEIGHTS_.T, self.WEIGHTS_), np.eye(self.WEIGHTS_.shape[1])):
            print('Random transformation matrix is not orthonormal despite Gram-Schmidt processing.')
        
        # Inverse weight matrix
        # self.WEIGHTS_INV_ = np.linalg.inv(self.WEIGHTS_)
        self.WEIGHTS_INV_ = self.WEIGHTS_.T
    
    def transform(self, X):
        # NEW_X = X@self.WEIGHTS_
        NEW_X = self.WEIGHTS_@X.T
        # return NEW_X
        return NEW_X.T
    
    def inverse_transform(self, X):
        # return X@self.WEIGHTS_INV_
        return (self.WEIGHTS_INV_@X.T).T
    
def gram_schmidt(X):
    A = np.copy(X)
    (n, m) = A.shape
   
    for i in range(m):
       
        q = A[:, i] # i-th column of A
       
        for j in range(i):
            q = q - np.dot(A[:, j], A[:, i]) * A[:, j]
       
        if np.array_equal(q, np.zeros(q.shape)):
            raise np.linalg.LinAlgError("The column vectors are not linearly independent")
       
        # normalize q
        q = q / np.sqrt(np.dot(q, q))
       
        # write the vector back in the matrix
        A[:, i] = q
   
    return A

def define_density_based_hrs(real: torch.Tensor, labels_real: torch.Tensor, density: float = None):
    """
    Computes lower and upper bounds for density-based hyper-rectangles (HRs) based on a real dataset.
    Explanation: for each dimension of each sample, the returned matrices give the upper and lower bounds 
                of a HR built upon this given sample.

    Parameters:
        real (torch.tensor): True data points with shape (N_true, N_vars).
        labels_real (torch.tensor): Labels for true data points with shape (N_true, 1) or (N_true, N_class).
        density (float): Density value to build a HR's width.

    Returns:
        tuple: 
            FINAL_LOW_BOUNDS (torch.tensor): Lower bounds for all data points and dimensions.
            FINAL_UPPER_BOUNDS (torch.tensor): Upper bounds for all data points and dimensions.
    """

    if not isinstance(density, float) or not (0 <= density <= 1):
        raise ValueError("Parameter `density` must be a float between 0 and 1.")

    # If labels are one-hot encoded, convert to single class labels
    if len(labels_real.shape) > 1:
        if labels_real.shape[1] > 1:
            labels_real = labels_real.detach().clone().argmax(dim=1)

    labels_real = labels_real.detach().clone().reshape(-1,1)
    
    device = real.device
    n_samples, n_dims = real.shape

    # Initialize tensors for storing final bounds
    FINAL_LOW_BOUNDS = torch.zeros_like(real)
    FINAL_UPPER_BOUNDS = torch.zeros_like(real)

    # Process all labels at once
    unique_labels = labels_real.unique()
    label_masks = (labels_real.unsqueeze(1) == unique_labels.unsqueeze(0))  # shape: (n_samples, n_unique_labels)

    for mask in label_masks.permute(2,1,0):
        real_label = real[mask.flatten()]
        
        # 1. Sort data and generate CDF for the current label
        real_sorted, real_sorted_idx = real_label.sort(dim=0)
        cdf_real_label = torch.linspace(0, 1, real_sorted.shape[0], device=device).unsqueeze(1).expand(-1, n_dims)

        # 2. Find original CDF for given elements
        cdf_sampled = torch.gather(cdf_real_label, 0, real_sorted_idx.argsort(dim=0))

        # 3. Set quantile bounds
        lower_q = torch.clamp(cdf_sampled - density, min=0)
        upper_q = torch.clamp(cdf_sampled + density, max=1)

        # 4. Find corresponding quantiles
        # Use torch.quantile() for each dimension separately
        new_lower_bounds = torch.stack([
            torch.quantile(real_sorted[:, i], lower_q[:, i])
            for i in range(n_dims)
        ]).T

        new_upper_bounds = torch.stack([
            torch.quantile(real_sorted[:, i], upper_q[:, i])
            for i in range(n_dims)
        ]).T

        # 5. Add margin to lower and upper borders at the edge of the distribution
        window = (real_label.max(dim=0).values - real_label.min(dim=0).values) * density
        new_lower_bounds = torch.where(cdf_sampled - density < 0, new_lower_bounds - window, new_lower_bounds)
        new_upper_bounds = torch.where(cdf_sampled + density > 1, new_upper_bounds + window, new_upper_bounds)

        # Store bounds
        FINAL_LOW_BOUNDS[mask.flatten()] = new_lower_bounds
        FINAL_UPPER_BOUNDS[mask.flatten()] = new_upper_bounds

    return FINAL_LOW_BOUNDS, FINAL_UPPER_BOUNDS

class DataProcessor:
    def __init__(self, custom_data_loader: Callable, custom_data_processor: Callable):
        """
        DataProcessor class to handle the loading and processing of custom datasets.

        Parameters:
            custom_data_loader (Callable): User-provided function to load data.
            custom_data_processor (Callable): User-provided function to process data.
        """
        self.custom_data_loader = custom_data_loader
        self.custom_data_processor = custom_data_processor

    def prepare_data(self, train: bool = True, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data using custom loader and processor functions.

        Parameters:
            train (bool): If True, prepare training data. Otherwise, prepare test/validation data.
            **kwargs: Additional arguments to pass to custom functions.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        data = self.custom_data_loader(train=train, **kwargs)
        X, y = self.custom_data_processor(data, train=train, **kwargs)
        return X, y

    def create_dataloaders(self, batch_size: int = 64, density: Optional[float] = None, 
                           **kwargs) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
        """
        Create DataLoader objects for training and validation datasets.

        Parameters:
            batch_size (int): Batch size for the DataLoader.
            density (Optional[float]): Density value for hyper-rectangles width.
            **kwargs: Additional arguments to pass to data preparation functions.

        Returns:
            Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]: 
            Train and validation DataLoaders and feature tensors.
        """
        X, y = self.prepare_data(train=True, **kwargs)
        X_val, y_val = self.prepare_data(train=False, **kwargs)

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        X_val = torch.from_numpy(X_val).float()
        y_val = torch.from_numpy(y_val).float()

        x_idx = torch.arange(len(X))
        x_idx_val = torch.arange(len(X_val))

        if density is not None:
            lower_bounds, upper_bounds = define_density_based_hrs(X, y, density)
            lower_bounds_val, upper_bounds_val = define_density_based_hrs(X_val, y_val, density)

            train_loader = DataLoader(
                TensorDataset(X, y, lower_bounds, upper_bounds, x_idx), 
                batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(X_val, y_val, lower_bounds_val, upper_bounds_val, x_idx_val), 
                batch_size=batch_size, shuffle=False
            )
        else:
            train_loader = DataLoader(
                TensorDataset(X, y, x_idx), 
                batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(X_val, y_val, x_idx_val), 
                batch_size=batch_size, shuffle=False
            )

        return train_loader, val_loader, X, y
    
class ToyLoader:
    def __init__(self):
        self.datasets = [
            'circles', 'moons', 'gauss', 'eight_gaussians', 'blobs', 
            'swiss_roll', 'blobs3d', 'swiss_roll3d', 'blobs_multidim'
        ]

    def train_val_loaders(self, dataset: str, batch_size: int, density: float = None, n_dim: int = 4, nb_centers: int = 10):
        """ 
        Generates PyTorch dataloaders for training and validation datasets.
        
        Parameters:
            dataset (str): Dataset name among available datasets: 
                           'circles', 'moons', 'gauss', 'eight_gaussians', 'blobs', 
                           'swiss_roll', 'blobs3d', 'swiss_roll3d', 'blobs_multidim'.
            batch_size (int): Batch size of the dataloader.
            density (float): Density value to build a HR's width. Default is None.
            n_dim (int, optional): Number of dimensions for multidimensional datasets. Default is 4.
            nb_centers (int, optional): Number of centers for the blobs_multidim dataset. Default is 10.

        Returns:
            tuple: train_dataloader, val_dataloader (DataLoader), X (torch.Tensor), y (torch.Tensor)
        """
        if dataset not in self.datasets:
            raise ValueError(f"Dataset {dataset} not recognized. Available datasets: {self.datasets}")

        # Generate and standardize datasets
        X, y = self._generate_dataset(dataset, 6000, n_dim, nb_centers)
        X_val, y_val = self._generate_dataset(dataset, 1500, n_dim, nb_centers, is_validation=True)

        # Standardize the data
        X, self.scaler_x = standardize_train(X, as_tensors=False)
        X_val, _ = standardize_test(X_val, self.scaler_x, as_tensors=False)

        # Encode labels if necessary
        if dataset in ['eight_gaussians', 'swiss_roll', 'swiss_roll3d', 'blobs3d', 'blobs_multidim']:
            y, y_val = self._encode_labels(y, y_val)

        # Convert data to tensors
        X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
        X_val, y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()

        # Create indices
        x_idx = torch.arange(len(X))
        x_idx_val = torch.arange(len(X_val))

        # Build dataloaders
        if density is not None:
            train_dataloader, val_dataloader = self._build_dataloaders_with_hr(
                X, y, X_val, y_val, x_idx, x_idx_val, batch_size, density
            )
        else:
            train_dataloader = DataLoader(TensorDataset(X, y, x_idx), batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(TensorDataset(X_val, y_val, x_idx_val), batch_size=batch_size, shuffle=False)

        return train_dataloader, val_dataloader, X, y

    def _generate_dataset(self, dataset: str, n_samples: int, n_dim: int, nb_centers: int, is_validation: bool = False):
        """Generate a specified toy dataset."""
        if dataset == 'circles':
            return datasets.make_circles(n_samples=n_samples, factor=0.4, noise=0.08, random_state=0 if not is_validation else 42)
        elif dataset == 'moons':
            return datasets.make_moons(n_samples=n_samples, noise=0.08, random_state=0 if not is_validation else 42)
        elif dataset == 'gauss':
            X = np.random.normal(0, 1, (n_samples, 2))
            y = np.zeros((n_samples,))
            return X, y
        elif dataset == 'eight_gaussians':
            label = torch.randint(0, 8, (n_samples,))
            theta = (np.pi / 4) * label.float()
            centers = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
            noise = torch.randn_like(centers) * 0.1
            X = (centers + noise).numpy()
            y = label.numpy()
            return X, y
        elif dataset == 'blobs':
            return datasets.make_blobs(n_samples=n_samples, random_state=42 if not is_validation else 41)
        elif dataset == 'swiss_roll':
            X, y = datasets.make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42 if not is_validation else 40)
            X = X[:, [0, 2]]
            # Discrete labels for simplicity purposes
            mask1 = X[:,0]>=-1
            y[mask1] = 0
            mask2 = X[:,0]<-1
            y[mask2] = 1
            mask3 = (X[:,0]>=-1)*(X[:,0]<7)
            mask4 = (X[:,1]>=-6)*(X[:,1]<9)
            y[mask3*mask4] = 2
            return X, y
        elif dataset == 'blobs3d':
            return datasets.make_blobs(n_samples=n_samples, n_features=3, centers=5, random_state=41 if not is_validation else 42)
        elif dataset == 'blobs_multidim':
            return datasets.make_blobs(n_samples=n_samples, n_features=n_dim, centers=nb_centers, random_state=42 if not is_validation else 41)
        elif dataset== 'swiss_roll3d':
            X, y = datasets.make_swiss_roll(n_samples=n_samples, noise=0.1, random_state=42 if not is_validation else 40)
            # Discrete labels for simplicity purposes
            x = X[:,[0,2]]
            mask1 = x[:,0]>=-1
            y[mask1] = 0
            mask2 = x[:,0]<-1
            y[mask2] = 1
            mask3 = (x[:,0]>=-1)*(x[:,0]<7)
            mask4 = (x[:,1]>=-6)*(x[:,1]<9)
            y[mask3*mask4] = 2
            return X, y
    def _encode_labels(self, y, y_val):
        """One-hot encode labels."""
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(np.unique(y).reshape(-1, 1))
        y = encoder.transform(y.reshape(-1, 1)).toarray()
        y_val = encoder.transform(y_val.reshape(-1, 1)).toarray()
        return y, y_val

    def _build_dataloaders_with_hr(self, X, y, X_val, y_val, x_idx, x_idx_val, batch_size, density):
        """Build dataloaders with additional processing for density probes in GMDA."""
        lower_bounds, upper_bounds = define_density_based_hrs(X, y, density)
        lower_bounds_val, upper_bounds_val = define_density_based_hrs(X_val, y_val, density)
        train_dataloader = DataLoader(TensorDataset(X, y, lower_bounds, upper_bounds, x_idx), batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(TensorDataset(X_val, y_val, lower_bounds_val, upper_bounds_val, x_idx_val), batch_size=batch_size, shuffle=False)
        return train_dataloader, val_dataloader