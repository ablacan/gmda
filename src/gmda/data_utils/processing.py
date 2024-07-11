# Imports
import sys
import os
import random
import time as t
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
from .utils import define_density_based_hrs, standardize_test, standardize_train, RandTransform, gram_schmidt

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

class GestureLoader:
    def __init__(self):
        """
        GestureLoader class to handle the loading and processing of the Gesture Phase Segmentation dataset.
        Dataset URL: https://archive.ics.uci.edu/dataset/302/gesture+phase+segmentation
        """
        self.path_train = './data/gesture/train_gesture.csv'
        self.path_test = './data/gesture/test_gesture.csv'

    def init_path(self, path_train: str, path_test: str):
        self.path_train = path_train
        self.path_test = path_test

    def load_data(self, test: bool = False) -> pd.DataFrame:
        """
        Load data from CSV files.
        
        Parameters:
            test (bool): If True, load the test dataset. Otherwise, load the training dataset.
        
        Returns:
            pd.DataFrame: Loaded dataset.
        """
        file_path = self.path_test if test else self.path_train
        df = pd.read_csv(file_path, sep=',')
        return df

    def process_data(self, df: pd.DataFrame, test: bool = False, random_transform: bool = False, quantile_transform: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the data, including encoding the target and applying transformations.
        
        Parameters:
            df (pd.DataFrame): DataFrame to process.
            test (bool): If True, processing test data.
            random_transform (bool): If True, apply random transformation.
            quantile_transform (bool): If True, apply quantile transformation.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        target_col = 'Phase'
        
        if not test:
            self.Encoder_target = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.Encoder_target.fit(df[target_col].unique().reshape(-1, 1))
        
        y = self.Encoder_target.transform(df[target_col].to_numpy().reshape(-1, 1)).astype(float)
        df = df.drop('Phase', axis=1)
        X = df.to_numpy().astype(float)
        
        if random_transform:
            if not test:
                self.scaler_transfo = RandTransform()
                self.scaler_transfo.fit(X)
            X = self.scaler_transfo.transform(X)
        elif quantile_transform:
            if not test:
                self.scaler_transfo = QuantileTransformer(n_quantiles=10000, output_distribution='normal', random_state=42)
                self.scaler_transfo.fit(X)
            X = self.scaler_transfo.transform(X)

        return X, y

    def train_val_loaders(self, dataset: str=None, batch_size: int = 64, density: Optional[float] = None, random_transform: bool = False, quantile_transform: bool = False) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
        """
        Prepare DataLoader objects for training and validation datasets.
        
        Parameters:
            batch_size (int): Batch size for the DataLoader.
            density (Optional[float]): density value for hyper-rectangles width.
            random_transform (bool): If True, apply random transformation.
            quantile_transform (bool): If True, apply quantile transformation.
        
        Returns:
            Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]: Train and validation DataLoaders and feature tensors.
        """
        df = self.load_data(test=False)
        df_val = self.load_data(test=True)
        
        X, y = self.process_data(df, test=False, random_transform=random_transform, quantile_transform=quantile_transform)
        X_val, y_val = self.process_data(df_val, test=True, random_transform=random_transform, quantile_transform=quantile_transform)
        
        X, self.scaler_x = standardize_train(X, as_tensors=False)
        X_val, _ = standardize_test(X_val, self.scaler_x, as_tensors=False)
        
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(y).float()
        X_val = torch.from_numpy(X_val).float()
        Y_val = torch.from_numpy(y_val).float()
        
        x_idx = torch.arange(len(X))
        x_idx_val = torch.arange(len(X_val))

        if density is not None:
            lower_bounds, upper_bounds = define_density_based_hrs(X, Y, density)
            lower_bounds_val, upper_bounds_val = define_density_based_hrs(X_val, Y_val, density)
            
            train_loader = DataLoader(TensorDataset(X, Y, lower_bounds, upper_bounds, x_idx), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(TensorDataset(X_val, Y_val, lower_bounds_val, upper_bounds_val, x_idx_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        else:
            train_loader = DataLoader(TensorDataset(X, Y, x_idx), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(TensorDataset(X_val, Y_val, x_idx_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

        return train_loader, val_loader, X, Y

    def train_test_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load and process train and test datasets.
        
        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Processed train and test datasets.
        """
        df = self.load_data(test=False)
        df_val = self.load_data(test=True)
        
        X, y = self.process_data(df, test=False)
        X_val, y_val = self.process_data(df_val, test=True)
        
        X, self.scaler_x = standardize_train(X, as_tensors=False)
        X_val, _ = standardize_test(X_val, self.scaler_x, as_tensors=False)

        return (X, y), (X_val, y_val)
    
    def process_fake_ctgan(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process fake data generated by CTGAN.
        
        Parameters:
            x (np.ndarray): Fake data to process.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        y = x[:, -1]
        x = x[:, :-1]
        
        y = self.Encoder_target.transform(y.reshape(-1, 1)).astype(float)
        x, _ = standardize_test(x, self.scaler_x, as_tensors=False)

        return np.float64(x), y
    
    def process_fake_tabddpm(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process fake data generated by TabDDPM.
        
        Parameters:
            x (np.ndarray): Fake feature data.
            y (np.ndarray): Fake target data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        x, _ = standardize_test(x, self.scaler_x, as_tensors=False)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        elif len(y.shape) == 2 and y.shape[1] == 1:
            y = y.reshape(-1, 1)

        return np.float64(x), np.float64(y)


class DiabetesLoader:
    def __init__(self):
        """
        DiabetesLoader class to handle the loading and processing of the Diabetes dataset.
        Dataset URL: https://www.openml.org/search?type=data&sort=runs&id=37&status=active
        """
        self.path_train = './data/diabetes/train_diabetes.csv'
        self.path_test = './data/diabetes/test_diabetes.csv'

    def init_path(self, path_train: str, path_test: str):
        self.path_train = path_train
        self.path_test = path_test

    def load_data(self, test: bool = False) -> pd.DataFrame:
        """
        Load data from CSV files.
        
        Parameters:
            test (bool): If True, load the test dataset. Otherwise, load the training dataset.
        
        Returns:
            pd.DataFrame: Loaded dataset.
        """
        file_path = self.path_test if test else self.path_train
        df = pd.read_csv(file_path, sep=',')
        return df
    
    def process_data(self, df: pd.DataFrame, test: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the data, including encoding the target.
        
        Parameters:
            df (pd.DataFrame): DataFrame to process.
            test (bool): If True, processing test data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        target_col = 'class'
        df.loc[df[target_col] == "tested_positive", target_col] = 1
        df.loc[df[target_col] == "tested_negative", target_col] = 0
        y = df[target_col].to_numpy().reshape(-1, 1).astype(float)
        df = df.drop(target_col, axis=1)
        X = df.to_numpy().astype(float)

        return X, y

    def train_val_loaders(self, batch_size: int = 64, density: Optional[float] = None) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
        """
        Prepare DataLoader objects for training and validation datasets.
        
        Parameters:
            batch_size (int): Batch size for the DataLoader.
            density (Optional[float]): density value for hyper-rectangles width.
        
        Returns:
            Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]: Train and validation DataLoaders and feature tensors.
        """
        df = self.load_data(test=False)
        df_val = self.load_data(test=True)
        
        X, y = self.process_data(df, test=False)
        X_val, y_val = self.process_data(df_val, test=True)
        
        X, self.scaler_x = standardize_train(X, as_tensors=False)
        X_val, _ = standardize_test(X_val, self.scaler_x, as_tensors=False)
        
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(y).float()
        X_val = torch.from_numpy(X_val).float()
        Y_val = torch.from_numpy(y_val).float()
        
        x_idx = torch.arange(len(X))
        x_idx_val = torch.arange(len(X_val))

        if density is not None:
            lower_bounds, upper_bounds = define_density_based_hrs(X, Y, density)
            lower_bounds_val, upper_bounds_val = define_density_based_hrs(X_val, Y_val, density)
            
            train_loader = DataLoader(TensorDataset(X, Y, lower_bounds, upper_bounds, x_idx), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(TensorDataset(X_val, Y_val, lower_bounds_val, upper_bounds_val, x_idx_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        else:
            train_loader = DataLoader(TensorDataset(X, Y, x_idx), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(TensorDataset(X_val, Y_val, x_idx_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

        return train_loader, val_loader, X, Y

    def train_test_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load and process train and test datasets.
        
        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Processed train and test datasets.
        """
        df = self.load_data(test=False)
        df_val = self.load_data(test=True)
        
        X, y = self.process_data(df, test=False)
        X_val, y_val = self.process_data(df_val, test=True)
        
        X, self.scaler_x = standardize_train(X, as_tensors=False)
        X_val, _ = standardize_test(X_val, self.scaler_x, as_tensors=False)

        return (X, y), (X_val, y_val)
    
    def process_fake_ctgan(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process fake data generated by CTGAN.
        
        Parameters:
            x (np.ndarray): Fake data to process.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        y = x[:, -1]
        x = x[:, :-1]
        
        y[y == 'tested_negative'] = 0
        y[y == 'tested_positive'] = 1
        x, _ = standardize_test(x, self.scaler_x, as_tensors=False)

        return np.float64(x), y.reshape(-1, 1).astype(float)
    
    def process_fake_tabddpm(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process fake data generated by TabDDPM.
        
        Parameters:
            x (np.ndarray): Fake feature data.
            y (np.ndarray): Fake target data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        x, _ = standardize_test(x, self.scaler_x, as_tensors=False)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        elif len(y.shape) == 2 and y.shape[1] == 1:
            y = y.reshape(-1, 1)

        return np.float64(x), np.float64(y)

class WILTLoader:
    def __init__(self):
        """
        WILTLoader class to handle the loading and processing of the WILT dataset.
        Dataset URL: https://www.openml.org/search?type=data&sort=runs&id=40983&status=active
        Cite:
        Johnson, B., Tateishi, R., Hoan, N., 2013.
        A hybrid pansharpening approach and multiscale object-based image analysis for mapping diseased pine and oak trees.
        International Journal of Remote Sensing, 34 (20), 6969-6982.
        """
        self.path_train = './data/wilt/train_wilt.csv'
        self.path_test = './data/wilt/test_wilt.csv'

    def init_path(self, path_train: str, path_test: str):
        self.path_train = path_train
        self.path_test = path_test

    def load_data(self, test: bool = False) -> pd.DataFrame:
        """
        Load data from CSV files.
        
        Parameters:
            test (bool): If True, load the test dataset. Otherwise, load the training dataset.
        
        Returns:
            pd.DataFrame: Loaded dataset.
        """
        file_path = self.path_test if test else self.path_train
        df = pd.read_csv(file_path, sep=',')
        return df
    
    def process_data(self, df: pd.DataFrame, test: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the data, including encoding the target.
        
        Parameters:
            df (pd.DataFrame): DataFrame to process.
            test (bool): If True, processing test data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        target_col = 'class'
        df.loc[df[target_col] == 1, target_col] = 0
        df.loc[df[target_col] == 2, target_col] = 1
        y = df[target_col].to_numpy().reshape(-1, 1).astype(float)
        df = df.drop(target_col, axis=1)
        X = df.to_numpy().astype(float)

        return X, y

    def train_val_loaders(self, batch_size: int = 64, density: Optional[float] = None) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
        """
        Prepare DataLoader objects for training and validation datasets.
        
        Parameters:
            batch_size (int): Batch size for the DataLoader.
            density (Optional[float]): density value for hyper-rectangles width.
        
        Returns:
            Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]: Train and validation DataLoaders and feature tensors.
        """
        df = self.load_data(test=False)
        df_val = self.load_data(test=True)
        
        X, y = self.process_data(df, test=False)
        X_val, y_val = self.process_data(df_val, test=True)
        
        X, self.scaler_x = standardize_train(X, as_tensors=False)
        X_val, _ = standardize_test(X_val, self.scaler_x, as_tensors=False)
        
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(y).float()
        X_val = torch.from_numpy(X_val).float()
        Y_val = torch.from_numpy(y_val).float()
        
        x_idx = torch.arange(len(X))
        x_idx_val = torch.arange(len(X_val))

        if density is not None:
            lower_bounds, upper_bounds = define_density_based_hrs(X, Y, density)
            lower_bounds_val, upper_bounds_val = define_density_based_hrs(X_val, Y_val, density)
            
            train_loader = DataLoader(TensorDataset(X, Y, lower_bounds, upper_bounds, x_idx), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(TensorDataset(X_val, Y_val, lower_bounds_val, upper_bounds_val, x_idx_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        else:
            train_loader = DataLoader(TensorDataset(X, Y, x_idx), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(TensorDataset(X_val, Y_val, x_idx_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

        return train_loader, val_loader, X, Y

    def train_test_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load and process train and test datasets.
        
        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Processed train and test datasets.
        """
        df = self.load_data(test=False)
        df_val = self.load_data(test=True)
        
        X, y = self.process_data(df, test=False)
        X_val, y_val = self.process_data(df_val, test=True)
        
        X, self.scaler_x = standardize_train(X, as_tensors=False)
        X_val, _ = standardize_test(X_val, self.scaler_x, as_tensors=False)

        return (X, y), (X_val, y_val)
    
    def process_fake_ctgan(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process fake data generated by CTGAN.
        
        Parameters:
            x (np.ndarray): Fake data to process.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        y = x[:, -1]
        x = x[:, :-1]
        
        y[y == 1] = 0
        y[y == 2] = 1
        x, _ = standardize_test(x, self.scaler_x, as_tensors=False)

        return np.float64(x), y.reshape(-1, 1).astype(float)
    
    def process_fake_tabddpm(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process fake data generated by TabDDPM.
        
        Parameters:
            x (np.ndarray): Fake feature data.
            y (np.ndarray): Fake target data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        x, _ = standardize_test(x, self.scaler_x, as_tensors=False)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        elif len(y.shape) == 2 and y.shape[1] == 1:
            y = y.reshape(-1, 1)

        return np.float64(x), np.float64(y)

class MagicLoader:
    def __init__(self):
        """
        MagicLoader class to handle the loading and processing of the MAGIC Gamma Telescope dataset.
        Dataset URL: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
        Cite:
        Bock, R. (2007). MAGIC Gamma Telescope. UCI Machine Learning Repository. https://doi.org/10.24432/C52C8B.
        """
        self.path_train = './data/magic/train_magic.csv'
        self.path_test = './data/magic/test_magic.csv'

    def init_path(self, path_train: str, path_test: str):
        self.path_train = path_train
        self.path_test = path_test

    def load_data(self, test: bool = False) -> pd.DataFrame:
        """
        Load data from CSV files.
        
        Parameters:
            test (bool): If True, load the test dataset. Otherwise, load the training dataset.
        
        Returns:
            pd.DataFrame: Loaded dataset.
        """
        file_path = self.path_test if test else self.path_train
        df = pd.read_csv(file_path, sep=',')
        return df
    
    def process_data(self, df: pd.DataFrame, test: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the data, including encoding the target.
        
        Parameters:
            df (pd.DataFrame): DataFrame to process.
            test (bool): If True, processing test data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        target_col = 'class'
        df.loc[df[target_col] == 'g', target_col] = 0
        df.loc[df[target_col] == 'h', target_col] = 1
        y = df[target_col].to_numpy().reshape(-1, 1).astype(float)
        df = df.drop(target_col, axis=1)
        X = df.to_numpy().astype(float)

        return X, y

    def train_val_loaders(self, batch_size: int = 64, density: Optional[float] = None) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
        """
        Prepare DataLoader objects for training and validation datasets.
        
        Parameters:
            batch_size (int): Batch size for the DataLoader.
            density (Optional[float]): density value for hyper-rectangles width.
        
        Returns:
            Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]: Train and validation DataLoaders and feature tensors.
        """
        df = self.load_data(test=False)
        df_val = self.load_data(test=True)
        
        X, y = self.process_data(df, test=False)
        X_val, y_val = self.process_data(df_val, test=True)
        
        X, self.scaler_x = standardize_train(X, as_tensors=False)
        X_val, _ = standardize_test(X_val, self.scaler_x, as_tensors=False)
        
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(y).float()
        X_val = torch.from_numpy(X_val).float()
        Y_val = torch.from_numpy(y_val).float()
        
        x_idx = torch.arange(len(X))
        x_idx_val = torch.arange(len(X_val))

        if density is not None:
            lower_bounds, upper_bounds = define_density_based_hrs(X, Y, density)
            lower_bounds_val, upper_bounds_val = define_density_based_hrs(X_val, Y_val, density)
            
            train_loader = DataLoader(TensorDataset(X, Y, lower_bounds, upper_bounds, x_idx), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(TensorDataset(X_val, Y_val, lower_bounds_val, upper_bounds_val, x_idx_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        else:
            train_loader = DataLoader(TensorDataset(X, Y, x_idx), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(TensorDataset(X_val, Y_val, x_idx_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

        return train_loader, val_loader, X, Y

    def train_test_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load and process train and test datasets.
        
        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Processed train and test datasets.
        """
        df = self.load_data(test=False)
        df_val = self.load_data(test=True)
        
        X, y = self.process_data(df, test=False)
        X_val, y_val = self.process_data(df_val, test=True)
        
        X, self.scaler_x = standardize_train(X, as_tensors=False)
        X_val, _ = standardize_test(X_val, self.scaler_x, as_tensors=False)

        return (X, y), (X_val, y_val)
    
    def process_fake_ctgan(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process fake data generated by CTGAN.
        
        Parameters:
            x (np.ndarray): Fake data to process.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        y = x[:, -1]
        x = x[:, :-1]
        
        y[y == 'g'] = 0
        y[y == 'h'] = 1
        x, _ = standardize_test(x, self.scaler_x, as_tensors=False)

        return np.float64(x), y.reshape(-1, 1).astype(float)
    
    def process_fake_tabddpm(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process fake data generated by TabDDPM.
        
        Parameters:
            x (np.ndarray): Fake feature data.
            y (np.ndarray): Fake target data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        x, _ = standardize_test(x, self.scaler_x, as_tensors=False)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        elif len(y.shape) == 2 and y.shape[1] == 1:
            y = y.reshape(-1, 1)

        return np.float64(x), np.float64(y)

class GTExLoader:
    def __init__(self):
        """
        GTExLoader class to handle the loading and processing of the GTEx dataset.
        Cite:
        #TODO
        ...
        """
        self.path_train = "./data/gtex/df_train_gtex_L974.csv"
        self.path_test = "./data/gtex/df_test_gtex_L974.csv"

        self._tissues = ['Adipose Tissue', 'Adrenal Gland', 'Blood', 'Blood Vessel',
                   'Brain', 'Breast', 'Colon', 'Esophagus', 'Heart', 'Liver', 'Lung',
                   'Muscle', 'Nerve', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate',
                   'Salivary Gland', 'Skin', 'Small Intestine', 'Spleen', 'Stomach',
                   'Testis', 'Thyroid', 'Uterus', 'Vagina']
    
    def init_path(self, path_train: str, path_test: str):
        self.path_train = path_train
        self.path_test = path_test

    def load_data(self, test: bool = False) -> pd.DataFrame:
        """
        Load data from CSV files.
        
        Parameters:
            test (bool): If True, load the test dataset. Otherwise, load the training dataset.
        
        Returns:
            pd.DataFrame: Loaded dataset.
        """
        file_path = self.path_test if test else self.path_train
        df = pd.read_csv(file_path, sep=',')
        return df
    
    def process_data(self, df: pd.DataFrame, test: bool = False, random_transform: bool = False, quantile_transform: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the data, including encoding the target and handling transformations.
        
        Parameters:
            df (pd.DataFrame): DataFrame to process.
            test (bool): If True, processing test data.
            random_transform (bool): If True, apply random transformation.
            quantile_transform (bool): If True, apply quantile transformation.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        df = df.dropna()
        
        col_names_not_dna = [col for col in df.columns if not col.isdigit()]
        for col in ['age', 'gender', 'tissue_type']:
            col_names_not_dna.remove(col)
        df = df.drop(columns=col_names_not_dna)
        
        numerical_covs = df[['age', 'gender']]
        numerical_covs.loc[numerical_covs['gender'] == "male", 'gender'] = 0
        numerical_covs.loc[numerical_covs['gender'] == "female", 'gender'] = 1
        numerical_covs = numerical_covs.values.astype(np.float32)

        target_col = 'tissue_type'
        if not test:
            self.Encoder_target = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.Encoder_target.fit(df[target_col].unique().reshape(-1, 1))
        y = self.Encoder_target.transform(df[target_col].to_numpy().reshape(-1, 1)).astype(float)
        
        df = df.drop(columns=['age', 'gender', 'tissue_type'])
        X = df.to_numpy().astype(float)
        
        if random_transform:
            if not test:
                self.scaler_rand = RandTransform()
                self.scaler_rand.fit(X)
            X = self.scaler_rand.transform(X)
        elif quantile_transform:
            if not test:
                self.scaler_rand = QuantileTransformer(n_quantiles=10000, output_distribution='normal', random_state=42)
                self.scaler_rand.fit(X)
            X = self.scaler_rand.transform(X)

        return X, y

    def train_val_loaders(self, batch_size: int = 64, density: Optional[float] = None, random_transform: bool = False, quantile_transform: bool = False, path_processing_files: str = None) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
        """
        Prepare DataLoader objects for training and validation datasets.
        
        Parameters:
            batch_size (int): Batch size for the DataLoader.
            density (Optional[float]): density value for hyper-rectangles width.
            random_transform (bool): If True, apply random transformation.
            quantile_transform (bool): If True, apply quantile transformation.
            path_processing_files (str): Path to processed files if any. 
        
        Returns:
            Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]: Train and validation DataLoaders and feature tensors.
        """
        df = self.load_data(test=False)
        df_val = self.load_data(test=True)
        
        X, y = self.process_data(df, test=False, random_transform=random_transform, quantile_transform=quantile_transform)
        X_val, y_val = self.process_data(df_val, test=True, random_transform=random_transform, quantile_transform=quantile_transform)
        
        X, self.scaler_x = standardize_train(X, as_tensors=False)
        X_val, _ = standardize_test(X_val, self.scaler_x, as_tensors=False)
        
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(y).float()
        X_val = torch.from_numpy(X_val).float()
        Y_val = torch.from_numpy(y_val).float()
        
        x_idx = torch.arange(len(X))
        x_idx_val = torch.arange(len(X_val))

        if density is not None:
            # Because of time consumption, it is best to save the computed CDF for a given density
            if not os.path.exists(f'{path_processing_files}/lower_bounds_{density}.pt'):
                lower_bounds, upper_bounds = define_density_based_hrs(X, Y, density)
                torch.save(lower_bounds, f'{path_processing_files}/lower_bounds_{density}.pt')
                torch.save(upper_bounds, f'{path_processing_files}/upper_bounds_{density}.pt')
                lower_bounds_val, upper_bounds_val = define_density_based_hrs(X_val, Y_val, density)
                torch.save(lower_bounds_val, f'{path_processing_files}/lower_bounds_val_{density}.pt')
                torch.save(upper_bounds_val, f'{path_processing_files}/upper_bounds_val_{density}.pt')
            else:
                lower_bounds = torch.load(f'{path_processing_files}/lower_bounds_{density}.pt')
                upper_bounds = torch.load(f'{path_processing_files}/upper_bounds_{density}.pt')
                lower_bounds_val = torch.load(f'{path_processing_files}/lower_bounds_val_{density}.pt')
                upper_bounds_val = torch.load(f'{path_processing_files}/upper_bounds_val_{density}.pt')
                
            train_loader = DataLoader(TensorDataset(X, Y, lower_bounds, upper_bounds, x_idx), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(TensorDataset(X_val, Y_val, lower_bounds_val, upper_bounds_val, x_idx_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        else:
            train_loader = DataLoader(TensorDataset(X, Y, x_idx), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(TensorDataset(X_val, Y_val, x_idx_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

        return train_loader, val_loader, X, Y
    
    def train_test_data(self, standardize: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load and process train and test datasets.
        
        Parameters:
            standardize (bool): If True, standardize the features.
        
        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Processed train and test datasets.
        """
        df = self.load_data(test=False)
        df_val = self.load_data(test=True)
        
        X, y = self.process_data(df, test=False)
        X_val, y_val = self.process_data(df_val, test=True)
        
        if standardize:
            X, self.scaler_x = standardize_train(X, as_tensors=False)
            X_val, _ = standardize_test(X_val, self.scaler_x, as_tensors=False)

        return (X, y), (X_val, y_val)
    
    
    def process_fake_ctgan(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process fake data generated by CTGAN.
        
        Parameters:
            x (np.ndarray): Fake data to process.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        y = x[:, -1]
        x = x[:, :-1]
        
        y = self.Encoder_target.transform(y.reshape(-1, 1)).astype(float)
        x, _ = standardize_test(x, self.scaler_x, as_tensors=False)

        return np.float64(x), y
    
    def process_fake_tabddpm(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process fake data generated by TabDDPM.
        
        Parameters:
            x (np.ndarray): Fake feature data.
            y (np.ndarray): Fake target data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        x, _ = standardize_test(x, self.scaler_x, as_tensors=False)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        elif len(y.shape) == 2 and y.shape[1] == 1:
            y = y.reshape(-1, 1)

        return np.float64(x), np.float64(y)
    

class TCGALoader:
    
    def __init__(self):
        """
        Initializes the TCGALoader with the paths to the train and test datasets.
        Cite:
        #TODO
        """
        self.path_train = "./data/tcga/train_df.csv"
        self.path_test = "./data/tcga/test_df.csv"

        self._tissues = ['adrenal', 'bladder', 'breast', 'cervical', 'liver', 
                   'colon', 'blood', 'esophagus', 'brain', 'head', 'kidney',
                     'kidney', 'kidney', 'blood', 'brain', 'liver', 'lung', 
                     'lung', 'lung', 'ovary', 'pancreas', 'kidney', 'prostate',
                     'rectum', 'soft-tissues', 'skin', 'stomach', 'stomach',
                       'testes', 'thyroid', 'thymus', 'uterus', 'uterus', 'eye']
        
    def init_path(self, path_train: str, path_test: str):
        self.path_train = path_train
        self.path_test = path_test

    def load_data(self, test: bool = False) -> pd.DataFrame:
        """
        Loads the train or test dataset.
        
        Parameters:
            test (bool): If True, loads the test dataset. Otherwise, loads the train dataset. Default is False.
        
        Returns:
            pd.DataFrame: Loaded dataset.
        """
        path = self.path_test if test else self.path_train
        df = pd.read_csv(path, sep=',')
        return df

    def process_data(self, df: pd.DataFrame, test: bool = False, random_transform: bool = False, quantile_transform: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes the dataset by handling missing values, encoding the target column, and applying optional transformations.
        
        Parameters:
            df (pd.DataFrame): The dataset to process.
            test (bool): If True, processes the test dataset. Default is False.
            random_transform (bool): If True, applies a random transformation to the data. Default is False.
            quantile_transform (bool): If True, applies a quantile transformation to the data. Default is False.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed features and target arrays.
        """
        # Remove rows with NaN values
        df = df.dropna()

        # Define the target column and encode it
        target_col = 'tissue_type'
        if not test:
            self.Encoder_target = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.Encoder_target.fit(df[target_col].unique().reshape(-1, 1))
        y = self.Encoder_target.transform(df[target_col].to_numpy().reshape(-1, 1)).astype(float)

        # Drop non-feature columns
        df = df.drop(columns=['age', 'gender', 'cancer', 'tissue_type'])

        # Keep only landmark genes
        landmark_genes_path = './data/tcga_union_978landmark_genes.csv'
        df_landmark = pd.read_csv(landmark_genes_path)
        df_landmark['genes_ids'] = df_landmark['genes_ids'].astype(str)
        df = df[df_landmark['genes_ids']]

        # Convert features to numpy array
        X = df.to_numpy().astype(float)

        # Apply transformations if specified
        if random_transform:
            if not test:
                self.scaler_rand = RandTransform()
                self.scaler_rand.fit(X)
            X = self.scaler_rand.transform(X)
        elif quantile_transform:
            if not test:
                self.scaler_rand = QuantileTransformer(n_quantiles=10000, output_distribution='normal', random_state=42)
                self.scaler_rand.fit(X)
            X = self.scaler_rand.transform(X)

        return X, y

    #TODO
    def train_val_loaders(self, batch_size: int = 64, density: Optional[float] = None, random_transform: bool = False, quantile_transform: bool = False) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
        """
        Creates DataLoader objects for training and validation datasets.
        
        Parameters:
            batch_size (int): Batch size for DataLoader. Default is 64.
            density (Optional[float]): density value for hyper-rectangles width.
            random_transform (bool): If True, applies a random transformation to the data. Default is False.
            quantile_transform (bool): If True, applies a quantile transformation to the data. Default is False.
        
        Returns:
            Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]: Train and validation DataLoaders, and the processed training features and targets.
        """
        # Load datasets
        df = self.load_data(test=False)
        df_val = self.load_data(test=True)
        # Process data
        X, y = self.process_data(df, test=False, random_transform=random_transform, quantile_transform=quantile_transform)
        X_val, y_val = self.process_data(df_val, test=True, random_transform=random_transform, quantile_transform=quantile_transform)

        # Standardize (only numerical data)
        X, self.scaler_x = standardize_train(X, as_tensors=False)
        X_val, _ = standardize_test(X_val, self.scaler_x, as_tensors=False)

        # Convert to tensors
        X, Y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
        X_val, Y_val = torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()

        # Indices
        x_idx, x_idx_val = torch.arange(len(X)), torch.arange(len(X_val))
        
        if density is not None:
            if not os.path.exists(f'./data/tcga/lower_bounds_{density}.pt'):
                lower_bounds, upper_bounds = define_density_based_hrs(X, Y, density)
                torch.save(lower_bounds, f'./data/tcga/lower_bounds_{density}.pt')
                torch.save(upper_bounds, f'./data/tcga/upper_bounds_{density}.pt')
                
                lower_bounds_val, upper_bounds_val = define_density_based_hrs(X_val, Y_val, density)
                torch.save(lower_bounds_val, f'./data/tcga/lower_bounds_val_{density}.pt')
                torch.save(upper_bounds_val, f'./data/tcga/upper_bounds_val_{density}.pt')
            else:
                lower_bounds = torch.load(f'./data/tcga/lower_bounds_{density}.pt')
                upper_bounds = torch.load(f'./data/tcga/upper_bounds_{density}.pt')
                lower_bounds_val = torch.load(f'./data/tcga/lower_bounds_val_{density}.pt')
                upper_bounds_val = torch.load(f'./data/tcga/upper_bounds_val_{density}.pt')
            
            # Build DataLoaders with hazard ratio bounds
            train_loader = DataLoader(TensorDataset(X, Y, lower_bounds, upper_bounds, x_idx), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(TensorDataset(X_val, Y_val, lower_bounds_val, upper_bounds_val, x_idx_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        else:
            # Build DataLoaders without hazard ratio bounds
            train_loader = DataLoader(TensorDataset(X, Y, x_idx), batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(TensorDataset(X_val, Y_val, x_idx_val), batch_size=batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

        return train_loader, val_loader, X, Y
    
    def train_test_data(self, standardize: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Loads and processes train and test datasets, optionally standardizing the features.
        
        Parameters:
            standardize (bool): If True, standardizes the features. Default is True.
        
        Returns:
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: Processed train and test datasets.
        """
        df = self.load_data(test=False)
        df_val = self.load_data(test=True)
        X, y = self.process_data(df, test=False)
        X_val, y_val = self.process_data(df_val, test=True)

        if standardize:
            X, self.scaler_x = standardize_train(X, as_tensors=False)
            X_val, _ = standardize_test(X_val, self.scaler_x, as_tensors=False)

        return (X, y), (X_val, y_val)

    def process_fake_ctgan(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process fake data generated by CTGAN.
        
        Parameters:
            x (np.ndarray): Fake data to process.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        y = x[:, -1]
        x = x[:, :-1]
        
        y = self.Encoder_target.transform(y.reshape(-1, 1)).astype(float)
        x, _ = standardize_test(x, self.scaler_x, as_tensors=False)

        return np.float64(x), y
    
    def process_fake_tabddpm(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process fake data generated by TabDDPM.
        
        Parameters:
            x (np.ndarray): Fake feature data.
            y (np.ndarray): Fake target data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed feature and target arrays.
        """
        x, _ = standardize_test(x, self.scaler_x, as_tensors=False)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        elif len(y.shape) == 2 and y.shape[1] == 1:
            y = y.reshape(-1, 1)

        return np.float64(x), np.float64(y)

def build_loader(data, config: dict = None) -> DataLoader:
    """
    Build a PyTorch DataLoader with the given configuration.

    Parameters:
        data (pytorch TensorDataset): The dataset to load.
        config (dict, optional): Configuration for the DataLoader. Defaults to None.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    if config is None:
        config = {}
    
    return DataLoader(
        data,
        batch_size=config.get('batch_size', 32),
        shuffle=config.get('shuffle', True),
        num_workers=config.get('num_workers', 2),
        pin_memory=config.get('pin_memory', True),
        prefetch_factor=config.get('prefetch_factor', 2),
        persistent_workers=config.get('persistent_workers', False)
    )

def get_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """
    Calculate class weights based on the frequency of each class in the training data.

    Parameters:
        train_loader (DataLoader): DataLoader for the training data.

    Returns:
        torch.Tensor: Tensor of class weights.
    """
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels.argmax(1))
    
    all_labels = torch.cat(all_labels)
    unique_labels = torch.unique(all_labels)
    
    # Calculate weights as inverse of class frequency
    class_weights = torch.tensor([
        len(all_labels) / (all_labels == label).sum().item() 
        for label in unique_labels
    ])
    
    return class_weights

def cls_train_val_test_dataloaders(
    X: np.ndarray, 
    y: np.ndarray, 
    X_test: np.ndarray = None, 
    y_test: np.ndarray = None, 
    standardize: bool = True, 
    config_dataloaders: dict = None
) -> tuple:
    """
    Prepare train, validation, and (optionally) test DataLoaders for classification tasks.

    Parameters:
        X (np.ndarray): Feature matrix for train and validation.
        y (np.ndarray): Labels for train and validation.
        X_test (np.ndarray, optional): Feature matrix for test set. Defaults to None.
        y_test (np.ndarray, optional): Labels for test set. Defaults to None.
        standardize (bool, optional): Whether to standardize the features. Defaults to True.
        config_dataloaders (dict, optional): Configuration for DataLoaders. Defaults to None.

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_weights)
               test_loader will be None if X_test is not provided.
    """
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Standardize features if requested
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        if X_test is not None:
            X_test = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    if X_test is not None:
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_loader = build_loader(TensorDataset(X_train, y_train), config=config_dataloaders)
    val_loader = build_loader(TensorDataset(X_val, y_val), config=config_dataloaders, shuffle=False)
    
    test_loader = None
    if X_test is not None:
        test_loader = build_loader(TensorDataset(X_test, y_test), config=config_dataloaders, shuffle=False)

    # Calculate class weights
    class_weights = get_class_weights(train_loader)

    return train_loader, val_loader, test_loader, class_weights

