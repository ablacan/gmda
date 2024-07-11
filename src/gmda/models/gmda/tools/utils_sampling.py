"""
GMDA (Generative Model Distribution Alignment) Algorithm Tools

This script contains the implementation of various tools and components
used in the GMDA algorithm. The GMDA algorithm is designed to align
the distributions of real and generated data in deep generative models.

Key components include:
- DBSSampler: A density-based sampling class for hyper-rectangles
- maxabs_loss_func_hr: the differentiable objective function 
- Other related functions and utilities for the GMDA process

For more information on the GMDA algorithm and its applications, please
refer to the accompanying documentation or research paper.

Copyright (c) [2024] [Alice Lacan]
All rights reserved.
"""

# Imports
import numpy as np
import torch
import random
from typing import Tuple, Optional

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

def compute_coverage(probs_real: torch.Tensor, probs_fake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the coverage level of each point by all the Hyper-Rectangles (HRs).

    This function calculates the sum of probabilities for real and fake data points
    across all HRs, providing a measure of how well each point is covered by the HRs.

    Parameters:
    -----------
    probs_real : torch.Tensor
        Probabilities of real data points for each HR.
    probs_fake : torch.Tensor
        Probabilities of fake data points for each HR.
    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor]
        - Sum of probabilities for real data points across all HRs.
        - Sum of probabilities for fake data points across all HRs.
    """
    return probs_real.sum(0), probs_fake.sum(0)

def compute_hr_coverage(probs_real: torch.Tensor, probs_fake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean coverage level of each Hyper-Rectangle (HR) on all points.

    This function calculates the average probability of real and fake data points
    for each HR, providing a measure of how well each HR covers the data points.

    Parameters:
    -----------
    probs_real : torch.Tensor
        Probabilities of real data points for each HR.
    probs_fake : torch.Tensor
        Probabilities of fake data points for each HR.

    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor]
        - Mean probabilities of real data points for each HR.
        - Mean probabilities of fake data points for each HR.
    """
    return probs_real.mean(-1), probs_fake.mean(-1)

def sigmoid(x: torch.Tensor, lamb: float) -> torch.Tensor:
    """
    Compute the sigmoid function.

    Parameters:
    -----------
    x : torch.Tensor
        Input tensor.
    lamb : float
        Scaling factor for the sigmoid function.

    Returns:
    --------
    torch.Tensor
        Sigmoid of the input tensor.
    """
    return 1 / (1 + torch.exp(-lamb * x))

def stable_sigmoid(x: torch.Tensor, lamb: float) -> torch.Tensor:
    """
    Compute a numerically stable version of the sigmoid function.

    This version is less prone to overflow for large positive values of x.

    Parameters:
    -----------
    x : torch.Tensor
        Input tensor.
    lamb : float
        Scaling factor for the sigmoid function.

    Returns:
    --------
    torch.Tensor
        Stable sigmoid of the input tensor.
    """
    return torch.exp(lamb * x) / (1 + torch.exp(lamb * x))

def exp_func(diff_to_bounds: torch.Tensor, lamb: float) -> torch.Tensor:
    """
    Compute an exponential function that combines regular and stable sigmoid.

    This function uses different sigmoid implementations based on the input values
    to ensure numerical stability across the entire input range.

    Parameters:
    -----------
    diff_to_bounds : torch.Tensor
        Difference to the bounds of the Hyper-Rectangles.
    lamb : float
        Scaling factor for the sigmoid functions.

    Returns:
    --------
    torch.Tensor
        Result of the exponential function.
    """
    # Initialize output tensor
    d_bis = torch.zeros_like(diff_to_bounds, device=diff_to_bounds.device)
    
    # Use regular sigmoid for non-negative values
    non_negative_mask = diff_to_bounds >= 0
    d_bis[non_negative_mask] = sigmoid(diff_to_bounds[non_negative_mask], lamb)
    
    # Use stable sigmoid for negative values
    negative_mask = diff_to_bounds < 0
    d_bis[negative_mask] = stable_sigmoid(diff_to_bounds[negative_mask], lamb)
    
    return d_bis

def indicator_func(x: torch.Tensor, low_bound: torch.Tensor, up_bound: torch.Tensor, lamb: float) -> torch.Tensor:
    """
    Compute indicator function with respect to the spread to bounds for batches of data and hyper-rectangles.

    This function calculates how much each data point falls within the bounds of each hyper-rectangle,
    using a smooth approximation of an indicator function.

    Parameters:
    -----------
    x : torch.Tensor
        Data points. Shape: (B, D) for 2D or 3D data, or (B, H, D) for higher dimensions.
        B: batch size, D: data dimension, H: number of hyper-rectangles.
    low_bound : torch.Tensor
        Lower bounds of hyper-rectangles. Shape: (H, 1, D).
    up_bound : torch.Tensor
        Upper bounds of hyper-rectangles. Shape: (H, 1, D).
    lamb : float
        Lambda parameter used in the exponential function to control the slope of the indicator function.

    Returns:
    --------
    torch.Tensor
        Indicator function values. Shape: (H, B, D).
    """
    return exp_func(x - low_bound, lamb) * exp_func(up_bound - x, lamb)

def maxabs_loss_func_hr(probs_real: torch.Tensor, probs_fake: torch.Tensor, 
                        labels_real: torch.Tensor, labels_fake: torch.Tensor, 
                        is_label_cat: bool = True) -> torch.Tensor:
    """
    Compute the 'MaxAbs' loss for hyper-rectangles (HRs).

    This loss function measures the difference between the mean probabilities of real and fake data points
    within each HR, normalized by a logarithmic factor.

    Loss formula: L(h) = |μ_true(h) - μ_fake(h)| / (1e-3 + log(1 + max(μ_true(h), μ_fake(h))))
    where μ_true(h) and μ_fake(h) are mean probabilities for real and fake data in HR h, respectively.

    Parameters:
    -----------
    probs_real : torch.Tensor
        Probabilities of real data points being in HRs. Shape: (H, B).
    probs_fake : torch.Tensor
        Probabilities of fake data points being in HRs. Shape: (H, B).
    labels_real : torch.Tensor
        Conditioning labels for real data points. Shape: (B, 1) or (B, N_class).
    labels_fake : torch.Tensor
        Conditioning labels for fake data points. Shape: (B, 1) or (B, N_class).
    is_label_cat : bool, optional
        Whether conditioning labels are used (default is True).

    Returns:
    --------
    torch.Tensor
        HR losses. Shape: (H,).
    """
    if is_label_cat:
        if len(labels_real.shape) >1:
            if labels_real.shape[1] > 1:
                labels_real = labels_real.argmax(1)
                labels_fake = labels_fake.argmax(1)
        
        labels_real = labels_real.flatten()
        labels_fake = labels_fake.flatten()
        unique_labels = labels_real.unique()
        
        # Pre-compute masks for all labels
        masks_real = (labels_real.unsqueeze(0) == unique_labels.unsqueeze(1))
        masks_fake = (labels_fake.unsqueeze(0) == unique_labels.unsqueeze(1))
        
        # Compute mean probabilities for all labels at once
        n_true = (probs_real.unsqueeze(1) * masks_real.unsqueeze(0)).sum(dim=2) / masks_real.sum(dim=1).unsqueeze(0).clamp(min=1)
        n_fake = (probs_fake.unsqueeze(1) * masks_fake.unsqueeze(0)).sum(dim=2) / masks_fake.sum(dim=1).unsqueeze(0).clamp(min=1)
        
        # Compute loss for all labels at once
        loss = torch.abs(n_true - n_fake) / (1e-3 + torch.log(1 + torch.maximum(n_true, n_fake)))
        
        # Sum losses for all labels
        loss = loss.sum(dim=1)
    else:
        n_true = probs_real.mean(-1)
        n_fake = probs_fake.mean(-1)
        loss = torch.abs(n_true - n_fake) / (1e-3 + torch.log(1 + torch.maximum(n_true, n_fake)))
    
    return loss

def maxabs_darkbin_func(probs_real: torch.Tensor, probs_fake: torch.Tensor, 
                        labels_real: torch.Tensor, labels_fake: torch.Tensor, 
                        is_label_cat: bool = True) -> torch.Tensor:
    """
    Compute the optimized 'MaxAbs' dark bin loss.

    This function calculates the absolute difference between the mean probabilities of real and fake data points
    not being in any of the hyper-rectangles (HRs), i.e., falling into the "dark bin".

    Dark bin probability: π(x) = Π_h (1 - σ(x | h)), where σ(x | h) is the probability of x being in HR h.

    Parameters:
    -----------
    probs_real : torch.Tensor
        Probabilities of real data points being in HRs. Shape: (H, B).
    probs_fake : torch.Tensor
        Probabilities of fake data points being in HRs. Shape: (H, B).
    labels_real : torch.Tensor
        Conditioning labels for real data points. Shape: (B, 1) or (B, N_class).
    labels_fake : torch.Tensor
        Conditioning labels for fake data points. Shape: (B, 1) or (B, N_class).
    is_label_cat : bool, optional
        Whether conditioning labels are used (default is True).

    Returns:
    --------
    torch.Tensor
        Dark bin loss. Shape: (1,).
    """
    if is_label_cat:
        if len(labels_real.shape)>1:
            if labels_real.shape[1] > 1:
                labels_real = labels_real.argmax(1)
                labels_fake = labels_fake.argmax(1)

        unique_labels = labels_real.unique()
        labels_real = labels_real.flatten()
        labels_fake = labels_fake.flatten()
        
        # Pre-compute masks for all labels
        masks_real = (labels_real.unsqueeze(0) == unique_labels.unsqueeze(1))
        masks_fake = (labels_fake.unsqueeze(0) == unique_labels.unsqueeze(1))
        
        # Compute dark bin probabilities
        dark_bin_real = torch.prod(1 - probs_real, dim=0)
        dark_bin_fake = torch.prod(1 - probs_fake, dim=0)
        
        # Compute mean dark bin probabilities for each label
        mean_dark_bin_real = (dark_bin_real.unsqueeze(0) * masks_real).sum(dim=1) / masks_real.sum(dim=1).clamp(min=1)
        mean_dark_bin_fake = (dark_bin_fake.unsqueeze(0) * masks_fake).sum(dim=1) / masks_fake.sum(dim=1).clamp(min=1)
        
        # Compute the total loss
        db = torch.abs(mean_dark_bin_real - mean_dark_bin_fake).sum()
    else:
        dark_bin_real = torch.prod(1 - probs_real, dim=0)
        dark_bin_fake = torch.prod(1 - probs_fake, dim=0)
        db = torch.abs(dark_bin_real.mean() - dark_bin_fake.mean())
    
    return db.unsqueeze(0)  # Return shape (1,) 

def define_new_hr_borders_dbs(
    prob_real: torch.Tensor,
    real_lower_bounds: torch.Tensor,
    real_upper_bounds: torch.Tensor,
    labels_real: torch.Tensor,
    probs_sample_dims: torch.Tensor,
    nb_hr: int = 100,
    hr_window: Optional[float] = None,
    prop_to_keep: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Density Based Sampling function to define new high-density region (HR) borders.

    This function samples new HR borders based on the probability distribution of real data points
    and their dimensions. It handles both binary and multi-class classification scenarios.

    Parameters:
        prob_real (torch.Tensor): Sampling probability of each true point (shape: [1, N_true]).
        real_lower_bounds (torch.Tensor): Lower bounds of true data points (shape: [N_true, N_vars]).
        real_upper_bounds (torch.Tensor): Upper bounds of true data points (shape: [N_true, N_vars]).
        labels_real (torch.Tensor): True data points conditioning labels (shape: [N_true, 1] or [N_true, N_class]).
        probs_sample_dims (torch.Tensor): Sampling probability of each dimension (shape: [1, N_vars]).
        nb_hr (int, optional): Total number of HRs to generate. Defaults to 100.
        hr_window (float, optional): Density value to build the HRs. Not used in current implementation.
        prop_to_keep (float, optional): Proportion of worst HRs at iteration t-1 to retain at iteration t. Defaults to 0.5.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - lower_bounds: Lower bounds of new HRs.
            - upper_bounds: Upper bounds of new HRs.
            - dims_idx_to_keep: Indices of dimensions to keep for each HR.
            - true_idx_to_store: Indices of true data points used to create new HRs.
    """
    # Convert multi-class labels to single column if necessary
    if len(labels_real.shape)>1:
        if labels_real.shape[1] > 1:
            labels_real = labels_real.argmax(1)
    
    nb_classes = len(labels_real.unique())
    prop_to_sample = 1.0 - prop_to_keep
    hrs_per_class = int(prop_to_sample * nb_hr // nb_classes + 1)

    device = real_lower_bounds.device
    all_idx = torch.arange(real_lower_bounds.shape[0], device=device)

    lower_bounds, upper_bounds, dims_idx_to_keep, true_idx_to_store = [], [], [], []

    for label in labels_real.unique():
        mask_real = (labels_real == label).flatten()
        if mask_real.sum() > 0:
            true_idx = prob_real[mask_real].multinomial(min(hrs_per_class, mask_real.sum().item()), replacement=False)
            true_idx_original = all_idx[mask_real][true_idx]

            if real_lower_bounds.shape[1] > 3:
                probs_ = probs_sample_dims.repeat(true_idx.shape[0], 1)
                new_dims_idx_to_keep = probs_.multinomial(3, replacement=False).sort().values
            else:
                new_dims_idx_to_keep = torch.arange(real_lower_bounds.shape[1], device=device).unsqueeze(0).repeat(true_idx.shape[0], 1)

            rows_to_keep = torch.arange(true_idx.shape[0], device=device).unsqueeze(1)

            bounds = torch.stack((real_lower_bounds[mask_real][true_idx], real_upper_bounds[mask_real][true_idx]))
            new_lower_bounds = torch.min(bounds, dim=0).values[rows_to_keep, new_dims_idx_to_keep]
            new_upper_bounds = torch.max(bounds, dim=0).values[rows_to_keep, new_dims_idx_to_keep]

            lower_bounds.append(new_lower_bounds)
            upper_bounds.append(new_upper_bounds)
            dims_idx_to_keep.append(new_dims_idx_to_keep)
            true_idx_to_store.append(true_idx_original)

    return (torch.cat(lower_bounds), torch.cat(upper_bounds), 
            torch.cat(dims_idx_to_keep), torch.cat(true_idx_to_store))

def fast_define_new_hr_borders_dbs(
    prob_real: torch.Tensor,
    real_lower_bounds: torch.Tensor,
    real_upper_bounds: torch.Tensor,
    labels_real: torch.Tensor,
    probs_sample_dims: torch.Tensor,
    nb_hr: int = 100,
    prop_to_keep: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Density Based Sampling function to define new high-density region (HR) borders.

    This function samples new HR borders based on the probability distribution of real data points
    and their dimensions. It handles both binary and multi-class classification scenarios.

    Parameters:
        prob_real (torch.Tensor): Sampling probability of each true point (shape: [1, N_true]).
        real_lower_bounds (torch.Tensor): Lower bounds of true data points (shape: [N_true, N_vars]).
        real_upper_bounds (torch.Tensor): Upper bounds of true data points (shape: [N_true, N_vars]).
        labels_real (torch.Tensor): True data points conditioning labels (shape: [N_true, 1] or [N_true, N_class]).
        probs_sample_dims (torch.Tensor): Sampling probability of each dimension (shape: [1, N_vars]).
        nb_hr (int, optional): Total number of HRs to generate. Defaults to 100.
        prop_to_keep (float, optional): Proportion of worst HRs at iteration t-1 to retain at iteration t. Defaults to 0.5.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - lower_bounds: Lower bounds of new HRs.
            - upper_bounds: Upper bounds of new HRs.
            - dims_idx_to_keep: Indices of dimensions to keep for each HR.
            - true_idx_to_store: Indices of true data points used to create new HRs.
    """
    # Convert multi-class labels to single column if 
    if len(labels_real.shape)>1:
        if labels_real.shape[1] > 1:
            labels_real = labels_real.detach().clone().argmax(1)
    labels_real = labels_real.detach().clone().reshape(-1,1) # reshape labels

    if len(prob_real.flatten().shape)==1:
        prob_real = prob_real.reshape(1,-1)
    
    device = real_lower_bounds.device
    n_vars = real_lower_bounds.shape[1]
    
    # Compute class-wise information
    unique_labels, label_counts = labels_real.unique(return_counts=True)
    nb_classes = len(unique_labels)
    prop_to_sample = 1.0 - prop_to_keep
    hrs_per_class = torch.full((nb_classes,), int(prop_to_sample * nb_hr // nb_classes + 1), device=device)
    
    # Adjust hrs_per_class based on available samples
    hrs_per_class = torch.min(hrs_per_class, label_counts).to(device)
    
    # Create masks for each label
    label_masks = (labels_real.unsqueeze(1) == unique_labels.unsqueeze(0))
    
    # Sample indices for all classes
    sampled_indices = torch.cat([
        torch.where(mask)[1][prob_real[mask].multinomial(count, replacement=False)]
        for mask, count in zip(label_masks.permute(2,1,0), hrs_per_class)
    ])
    
    # Sample dimensions
    if n_vars > 3:
        probs_ = probs_sample_dims.repeat(sampled_indices.shape[0], 1)
        dims_idx_to_keep = probs_.multinomial(3, replacement=False).sort().values
    else:
        dims_idx_to_keep = torch.arange(n_vars, device=device).unsqueeze(0).repeat(sampled_indices.shape[0], 1).to(device)
    
    # Compute new bounds
    selected_lower = real_lower_bounds[sampled_indices]
    selected_upper = real_upper_bounds[sampled_indices]
    
    lower_bounds = torch.gather(selected_lower, 1, dims_idx_to_keep)
    upper_bounds = torch.gather(selected_upper, 1, dims_idx_to_keep)
    
    return lower_bounds, upper_bounds, dims_idx_to_keep, sampled_indices

class DBSSampler:
    """
    Density-Based Sampling (DBS) of probes (hyper-rectangles).
    
    This class implements a sampling method that uses density-based hyper-rectangles
    to align real and generated distributions in deep generative models.
    """

    def __init__(self, nb_dim_to_keep: int = None, nb_hr: int = None, 
                 batch_size: int = None, probs_sample_dims=None,
                 probs_sample_real=None, 
                 eta: float = 0.5):
        """
        Initialize the DBSSampler.

        Parameters:
            nb_dim_to_keep (int): Number of dimensions to keep.
            nb_hr (int): Number of hyper-rectangles.
            batch_size (int): Batch size for sampling.
            probs_sample_dims (torch.Tensor): Probability distribution for sampling dimensions.
            probs_sample_real (torch.Tensor): Probability distribution for sampling real data points.
            eta (float): Persistence i.e. proportion of hyper-rectangles to keep at each iteration.
        """
        self.nb_dim_to_keep = nb_dim_to_keep
        self.nb_hr = nb_hr
        self.batch_size = batch_size
        self.probs_sample_dims = probs_sample_dims
        self.probs_sample_real = probs_sample_real
        self.prop_hr_to_keep = eta
        self.ite = 0

        self.device = probs_sample_real.device
        
        # Initialize memory
        self.memory = {
            'hr_bounds': [], 'hr_centers': [], 'hrs_next_ite': [],
            'idx_to_keep': torch.empty((nb_hr, nb_dim_to_keep), device=self.device, dtype=torch.long),
        }

        # Initialize hr_losses
        self.memory['hr_losses'] = torch.ones((self.nb_hr,), device=self.device)

        # Preallocate tensors for repeated use
        self.rows_to_keep = torch.arange(self.nb_hr, device=self.device).unsqueeze(1)

    def __call__(self, x: torch.tensor, l: torch.tensor, u: torch.tensor, idx: torch.tensor):
        """
        Main function to sample lower and upper bounds.

        Parameters:
            x (torch.Tensor): Input data.
            l (torch.Tensor): Lower bounds.
            u (torch.Tensor): Upper bounds.
            idx (torch.Tensor): Indices of input data.

        Returns:
            tuple: Lower and upper bounds, and dimensions to keep.
        """
        return self.sample_low_up_bounds(x, l, u, idx)

    def sample_low_up_bounds(self, batch_x: torch.tensor, batch_lower_bounds: torch.tensor, batch_upper_bounds: torch.tensor, batch_idx: torch.tensor):
        """
        Sample lower and upper bounds of HRs centered on data samples.

        Parameters:
            batch_x (torch.Tensor): Batch of input data.
            batch_lower_bounds (torch.Tensor): Batch of lower bounds.
            batch_upper_bounds (torch.Tensor): Batch of upper bounds.
            batch_idx (torch.Tensor): Batch of indices.

        Returns:
            tuple: Lower and upper bounds, and dimensions to keep.
        """
        if self.ite == 0:
            lower_bounds, upper_bounds = self._first_iteration_sampling(batch_x, batch_lower_bounds, batch_upper_bounds, batch_idx)
        else:
            lower_bounds, upper_bounds = self._subsequent_iteration_sampling()

        # Store current bounds and increment iteration counter
        self.memory['hr_bounds'] = (lower_bounds, upper_bounds)
        self.ite += 1

        return (lower_bounds, upper_bounds), self.memory['idx_to_keep']

    def _select_dimensions(self, batch_x: torch.tensor):
        """Select dimensions to keep based on the sampling strategy."""
        if self.nb_dim_to_keep != batch_x.shape[1]:
            probs_ = self.probs_sample_dims.repeat(self.nb_hr, 1)
            dims_idx_to_keep = probs_.multinomial(self.nb_dim_to_keep, replacement=False).to(batch_x.device)
            dims_idx_to_keep = dims_idx_to_keep.sort().values
        else:
            dims_idx_to_keep = torch.arange(batch_x.shape[1], device=self.device).repeat(self.nb_hr, 1)
        return dims_idx_to_keep

    def _first_iteration_sampling(self, batch_x: torch.tensor, batch_lower_bounds: torch.tensor, batch_upper_bounds: torch.tensor, batch_idx: torch.tensor):
        """Sampling strategy for the first iteration."""
        # Select dimensions to keep
        dims_idx_to_keep = self._select_dimensions(batch_x)
        # Uniform sampling of true samples
        idx = self.probs_sample_real[batch_idx].multinomial(self.nb_hr, replacement=False)
        self.memory['hr_centers'].append(batch_idx[idx])

        bounds = torch.stack((batch_lower_bounds[idx], batch_upper_bounds[idx]))
        lower_bounds = bounds[0].gather(1, dims_idx_to_keep)
        upper_bounds = bounds[1].gather(1, dims_idx_to_keep)

        self.memory['idx_to_keep'] = dims_idx_to_keep
        return lower_bounds, upper_bounds

    def _subsequent_iteration_sampling(self):
        """Sampling strategy for subsequent iterations."""
        # Persistence
        if int(self.prop_hr_to_keep * self.nb_hr) != 0:
            return self._sample_with_high_loss()
        else:
            # No persistence
            return self._sample_without_high_loss()

    def _sample_with_high_loss(self):
        """Sampling strategy when keeping high-loss hyper-rectangles."""
        num_to_keep = int(self.prop_hr_to_keep * self.nb_hr)
        idx_high_loss = self.memory['hr_losses'].topk(num_to_keep).indices
        old_lower_bounds, old_upper_bounds = self.memory['hr_bounds'][0][idx_high_loss], self.memory['hr_bounds'][1][idx_high_loss]
        old_dims_idx_to_keep = self.memory['idx_to_keep'][idx_high_loss]
        old_true_idx_to_keep = self.memory['hr_centers'][-1][idx_high_loss]
        # HRs defined at previous step (density based process)
        new_lower_bounds, new_upper_bounds, new_dims_idx_to_keep, new_true_idx_to_keep = self.memory['hrs_next_ite']

        lower_bounds = torch.cat((old_lower_bounds, new_lower_bounds), 0)
        upper_bounds = torch.cat((old_upper_bounds, new_upper_bounds), 0)
        dims_idx_to_keep = torch.cat((old_dims_idx_to_keep, new_dims_idx_to_keep))

        self.memory['hr_centers'].append(torch.cat((old_true_idx_to_keep, new_true_idx_to_keep)))
        self.memory['idx_to_keep'] = dims_idx_to_keep
        return lower_bounds, upper_bounds

    def _sample_without_high_loss(self):
        """Sampling strategy when not keeping high-loss hyper-rectangles."""
        # HRs defined at previous step (density based process)
        new_lower_bounds, new_upper_bounds, new_dims_idx_to_keep, new_true_idx_to_keep = self.memory['hrs_next_ite']
        self.memory['hr_centers'].append(new_true_idx_to_keep)
        self.memory['idx_to_keep'] = new_dims_idx_to_keep
        return new_lower_bounds, new_upper_bounds

    def compute_updated_bounds(self, x_lower_bounds: torch.tensor, x_upper_bounds: torch.tensor, labels_real: torch.tensor, probs_sample_real: torch.tensor, probs_sample_dims: torch.tensor=None, batch_idx: torch.tensor=None):
        """
        Compute updated bounds based on the current state.

        Parameters:
            x_lower_bounds (torch.Tensor): Lower bounds of input data.
            x_upper_bounds (torch.Tensor): Upper bounds of input data.
            labels_real (torch.Tensor): Real labels.
            probs_sample_real (torch.Tensor): Sampling probabilities for real data.
            probs_sample_dims (torch.Tensor, optional): Sampling probabilities for dimensions.
            batch_idx (torch.Tensor, optional): Batch indices.

        Returns:
            torch.Tensor: Updated sampling probabilities for real data.
        """
        lower_bounds, upper_bounds, dims_idx_to_keep, true_idx_to_keep = fast_define_new_hr_borders_dbs(
            probs_sample_real[batch_idx], x_lower_bounds, x_upper_bounds, labels_real, 
            probs_sample_dims, self.nb_hr, prop_to_keep=self.prop_hr_to_keep
        )
        # Reindexing in original indices
        true_idx_to_keep = batch_idx[true_idx_to_keep]
        self.memory['hrs_next_ite'] = (lower_bounds, upper_bounds, dims_idx_to_keep, true_idx_to_keep)

        return probs_sample_real

    def reset(self):
        """
        Reset sampler.
        """
        self.ite = 0
        # Initialize memory
        self.memory = {
            'hr_bounds': [], 'hr_centers': [], 'hrs_next_ite': [],
            'idx_to_keep': torch.empty((self.nb_hr, self.nb_dim_to_keep), device=self.device, dtype=torch.long),
        }

        # Initialize hr_losses
        self.memory['hr_losses'] = torch.ones((self.nb_hr,), device=self.device)

        # Preallocate tensors for repeated use
        self.rows_to_keep = torch.arange(self.nb_hr, device=self.device).unsqueeze(1)
