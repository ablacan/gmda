"""
Script for k-NN Precision and Recall.

Reference:
-----
Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., & Aila, T. (2019).
Improved Precision and Recall Metric for Assessing Generative Models. ArXiv, abs/1904.06991.

Original Code:
------
Adaptation of TensorFlow code provided by NVIDIA CORPORATION: https://github.com/kynkaat/improved-precision-and-recall-metric.git
"""

# Imports
import numpy as np
import torch
import random
import sklearn.metrics

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

def batch_pairwise_distances(U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise distances between two batches of feature vectors.

    Parameters:
        U (torch.Tensor): First feature vector with shape (n_samples_U, n_features).
        V (torch.Tensor): Second feature vector with shape (n_samples_V, n_features).

    Returns:
        torch.Tensor: Tensor of pairwise distances with shape (n_samples_U, n_samples_V).
    """
    norm_u = torch.sum(U**2, axis=1, keepdim=True)
    norm_v = torch.sum(V**2, axis=1, keepdim=True).T
    distances = torch.maximum(norm_u - 2 * torch.matmul(U, V.T) + norm_v, torch.zeros((1,)))
    return distances

class ManifoldEstimator:
    """
    Estimates the manifold of given feature vectors.
    """

    def __init__(self, features: np.ndarray, row_batch_size=25000, col_batch_size=50000, nhood_sizes=[3], clamp_to_percentile=None, eps=1e-5):
        """
        Initializes the manifold estimator.

        Parameters:
            features (np.ndarray): Matrix of feature vectors to estimate their manifold.
            row_batch_size (int): Row batch size to compute pairwise distances.
            col_batch_size (int): Column batch size to compute pairwise distances.
            nhood_sizes (list): Number of neighbors used to estimate the manifold.
            clamp_to_percentile (float): Prune hyperspheres that have radius larger than the given percentile.
            eps (float): Small number for numerical stability.
        """
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features

        # Estimate manifold of features by calculating distances to k-NN of
        # each sample.
        batch_size = features.shape[0]
        self.D = np.zeros([batch_size, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([row_batch_size, batch_size], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, batch_size, row_batch_size):
            end1 = min(begin1 + row_batch_size, batch_size)
            row_batch = features[begin1:end1]

            for begin2 in range(0, batch_size, col_batch_size):
                end2 = min(begin2 + col_batch_size, batch_size)
                col_batch = features[begin2:end2]

                distance_batch[0:end1 - begin1, begin2:end2] = batch_pairwise_distances(torch.tensor(row_batch), torch.tensor(col_batch)).numpy()

            self.D[begin1:end1, :] = np.partition(distance_batch[0:end1 - begin1, :], seq, axis=1)[:, self.nhood_sizes]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0

    def evaluate(self, eval_features: np.ndarray, return_realism=False, return_neighbors=False):
        """
        Evaluate if new feature vectors are within the manifold.

        Parameters:
            eval_features (np.ndarray): Matrix of feature vectors to evaluate.
            return_realism (bool): Whether to return realism scores.
            return_neighbors (bool): Whether to return nearest neighbor indices.

        Returns:
            np.ndarray: Predictions indicating if features are within the manifold.
            Optional[np.ndarray]: Realism scores.
            Optional[np.ndarray]: Nearest neighbor indices.
        """
        num_eval = eval_features.shape[0]
        num_ref = self.D.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref], dtype=np.float32)
        batch_predictions = np.zeros([num_eval, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros([num_eval], dtype=np.float32)
        nearest_indices = np.zeros([num_eval], dtype=np.int32)

        for begin1 in range(0, num_eval, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0:end1 - begin1, begin2:end2] = batch_pairwise_distances(torch.tensor(feature_batch), torch.tensor(ref_batch)).numpy()

            samples_in_manifold = distance_batch[0:end1 - begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(np.int32)
            max_realism_score[begin1:end1] = np.max(self.D[:, 0] / (distance_batch[0:end1 - begin1, :] + self.eps), axis=1)
            nearest_indices[begin1:end1] = np.argmin(distance_batch[0:end1 - begin1, :], axis=1)

        results = (batch_predictions,)
        if return_realism:
            results += (max_realism_score,)
        if return_neighbors:
            results += (nearest_indices,)
        
        return results if len(results) > 1 else results[0]

def knn_precision_recall_features(ref_features: np.ndarray, eval_features: np.ndarray, nhood_sizes=[3], row_batch_size=10000, col_batch_size=50000):
    """
    Calculates k-NN precision and recall for two sets of feature vectors.

    Parameters:
        ref_features (np.ndarray): Feature vectors of reference samples.
        eval_features (np.ndarray): Feature vectors of generated samples.
        nhood_sizes (list): Number of neighbors used to estimate the manifold.
        row_batch_size (int): Row batch size to compute pairwise distances (parameter to trade-off between memory usage and performance).
        col_batch_size (int): Column batch size to compute pairwise distances.

    Returns:
        dict: Dictionary containing precision and recall.
    """
    state = {}

    # Initialize ManifoldEstimators.
    ref_manifold = ManifoldEstimator(ref_features, row_batch_size, col_batch_size, nhood_sizes)
    eval_manifold = ManifoldEstimator(eval_features, row_batch_size, col_batch_size, nhood_sizes)

    # Precision: How many points from eval_features are in ref_features manifold.
    precision = ref_manifold.evaluate(eval_features)
    state['precision'] = precision.mean(axis=0)

    # Recall: How many points from ref_features are in eval_features manifold.
    recall = eval_manifold.evaluate(ref_features)
    state['recall'] = recall.mean(axis=0)

    return state

def get_precision_recall(real_data: np.ndarray, fake_data: np.ndarray, nb_nn: list = [3]) -> tuple:
    """
    Compute precision and recall between datasets.

    Parameters:
        real_data (np.ndarray): First dataset of comparison.
        fake_data (np.ndarray): Second dataset to use for comparison.
        nb_nn (list): Number of neighbors used to estimate the data manifold.

    Returns:
        tuple: Precision and recall values.
    """
    state = knn_precision_recall_features(real_data, fake_data, nhood_sizes=nb_nn)
    precision = state['precision'][0]
    recall = state['recall'][0]
    return precision, recall

def get_realism_score(real_data: torch.Tensor, fake_data: torch.Tensor) -> np.ndarray:
    """
    Compute realism score between datasets.

    Parameters:
        real_data (torch.Tensor): First dataset of comparison.
        fake_data (torch.Tensor): Second dataset to use for comparison.

    Returns:
        np.ndarray: Maximum realism scores.
    """
    real_manifold = ManifoldEstimator(real_data.numpy(), clamp_to_percentile=50)
    _, realism_scores = real_manifold.evaluate(fake_data.numpy(), return_realism=True)
    return realism_scores
