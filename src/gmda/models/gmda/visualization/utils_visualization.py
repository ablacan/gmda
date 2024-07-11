import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np
import torch
import umap
import random
from typing import Optional, Union

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across different libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_random_seeds()

def perform_umap(x, n_components=2, n_neighbors=100, min_dist=0.7):
    """
    Perform UMAP on input data and return 2D embeddings.

    Parameters:
        x (np.array): Input data for UMAP.
        n_components (int): Number of dimensions in the output (default: 2).
        n_neighbors (int): Number of neighbors to consider (default: 100).
        min_dist (float): Minimum distance between points in the output space (default: 0.7).

    Returns:
        np.array: 2D UMAP embeddings.
    """
    umap_model = umap.UMAP(n_neighbors=n_neighbors,
                           n_components=n_components,
                           min_dist=min_dist,
                           random_state=42)
    return umap_model.fit_transform(x)

def rebuild_bounds(bounds):
    """
    Rebuild 3D bounds of hyper-rectangles for visualization.

    Parameters:
        bounds (tuple): Lower and upper bounds.

    Returns:
        torch.Tensor: Reconstructed cube bounds.
    """
    A, B = bounds[0], torch.hstack((bounds[0][0:2], bounds[1][2]))
    C = torch.hstack((bounds[0][0], bounds[1][1], bounds[0][2]))
    D = torch.hstack((bounds[0][0], bounds[1][1], bounds[1][2]))
    E = torch.hstack((bounds[1][0], bounds[0][1], bounds[0][2]))
    F = torch.hstack((bounds[1][0], bounds[0][1], bounds[1][2]))
    G = torch.hstack((bounds[1][0], bounds[1][1], bounds[0][2]))
    H = bounds[1]
    return torch.stack((A,B,C,D,E,F,G,H))

def plot_cube_on_ax(axis, bounds):
    """
    Plot a 3D cube on the given axis.

    Parameters:
        axis (matplotlib.axes.Axes): The 3D axis to plot on.
        bounds (torch.Tensor): Cube bounds.

    Returns:
        matplotlib.axes.Axes: Updated axis with the cube plotted.
    """
    for i in range(0, len(bounds), 2):
        axis.plot(bounds[i:i+2,0], bounds[i:i+2,1], bounds[i:i+2,2], color='red', zorder=10)
    for i in range(0, len(bounds), 1):
        if i not in [2,3, 6,7]:
            axis.plot(bounds[[i,i+2]][:,0], bounds[[i,i+2]][:,1], bounds[[i,i+2]][:,2], color='red', zorder=10)
    for i in range(0, 4):
        axis.plot(bounds[[i,i+4]][:,0], bounds[[i,i+4]][:,1], bounds[[i,i+4]][:,2], color='red', zorder=50, alpha=0.5, lw=3)
    return axis

def plot_line(y, title='', x_label='', **kwargs):
    """
    Create a simple line plot.

    Parameters:
        y (array-like): Y-axis values.
        title (str): Plot title.
        x_label (str): X-axis label.
        **kwargs: Additional keyword arguments for plt.plot().

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    plt.figure(figsize=(10,5))
    plt.plot(np.arange(len(y)), y, **kwargs)
    plt.title(title, size=14)
    plt.ylabel('Loss', size=14)
    plt.xlabel(x_label, size=14)
    return plt.gcf()

def scatter_2d(data_2d, labels, colors=None, **kwargs):
    """
    Create a 2D scatter plot with labeled data.

    Parameters:
        data_2d (np.array): 2D data to plot.
        labels (array-like): Labels for coloring the points.
        colors (dict): Color mapping for labels.
        **kwargs: Additional keyword arguments for plt.scatter().

    Returns:
        matplotlib.axes.Axes: The plot axes.
    """
    for l in np.unique(labels):
        c = colors[l] if colors else None
        plt.scatter(data_2d[labels == l][:, 0], data_2d[labels == l][:, 1],
                    label=l, color=c, **kwargs)
    plt.legend(markerscale=3)
    return plt.gca()

class VisuEpoch:
    """Class for visualizing 2D data across training epochs."""

    def __init__(self, path=None):
        self.path = path
        self.path_suffix = '/umap_epoch_{}.png'

    def __call__(self, real, fake, labels, epoch=None, add_hr=False, hr_bounds=None, dataset='moons'):
        """
        Create and save visualization for the current epoch.

        Parameters:
            real (np.array): Real data points.
            fake (np.array): Generated data points.
            labels (np.array): Labels for the data points.
            epoch (int): Current epoch number.
            add_hr (bool): Whether to add hyper-rectangle visualization.
            hr_bounds (list): Bounds for hyper-rectangles.
            dataset (str): Name of the dataset.
        """
        full_data = np.concatenate((real, fake))
        full_labels = np.concatenate((labels.argmax(1), labels.argmax(1))) if dataset in ['swiss_roll_2d', 'eight_gaussians_2d'] else np.concatenate((labels, labels))
        real_fake_labels = np.array(["real"]*len(real)+["fake"]*len(fake))     
        
        plt.figure(figsize=(10,5))
        
        # Plot data points colored by class
        plt.subplot(121)
        for i, color in enumerate(["darkorange", "darkgreen", "darkblue"]):
            mask = full_labels == i
            if mask.any():
                plt.scatter(full_data[mask, 0], full_data[mask, 1], c=color, label=str(i), alpha=0.7)
        
        if add_hr:
            for HR_BOUNDS in hr_bounds:
                plt.gca().add_patch(Rectangle((HR_BOUNDS[0][0].cpu().numpy(), HR_BOUNDS[0][1].cpu().numpy()),
                                              np.abs(HR_BOUNDS[0][0].cpu().numpy()-HR_BOUNDS[1][0].cpu().numpy()),
                                              np.abs(HR_BOUNDS[0][1].cpu().numpy()-HR_BOUNDS[1][1].cpu().numpy()),
                                              edgecolor='k', facecolor='none', lw=4))
        plt.legend(fontsize=12, loc='lower left')
        self._set_plot_limits(dataset)
        plt.title(f"Epoch {epoch}", size=14, fontdict={'family':'serif'})

        # Plot data points colored by real/fake
        plt.subplot(122)
        plt.scatter(full_data[real_fake_labels == 'real', 0], full_data[real_fake_labels == 'real', 1], c="blue", label='real', alpha=0.7)
        plt.scatter(full_data[real_fake_labels == 'fake', 0], full_data[real_fake_labels == 'fake', 1], c="red", label='fake', alpha=0.7)
        
        self._set_plot_limits(dataset)
        plt.legend(fontsize=12, loc='lower left')
        plt.title(f"Epoch {epoch}", size=14, fontdict={'family':'serif'})
        
        plt.tight_layout()
        plt.savefig(self.path + self.path_suffix.format(epoch), format="png")

    def _set_plot_limits(self, dataset):
        """Set plot limits based on the dataset."""
        limits = {
            'moons_2d': (-2.5, 2.5, -2.5, 2.5),
            'circles_2d': (-2.5, 2.5, -2.5, 2.5),
            'swiss_roll_2d': (-2.5, 2.5, -2.5, 2.5),
            'blobs_2d': (-2.5, 2.5, -2., 2.),
            'eight_gaussians_2d': (-2.5, 2.5, -2., 2.)
        }
        if dataset in limits:
            plt.xlim(limits[dataset][0], limits[dataset][1])
            plt.ylim(limits[dataset][2], limits[dataset][3])

class VisuEpoch3D:
    """Class for visualizing 3D data across training epochs."""

    def __init__(self, path=None):
        self.path = path
        self.path_suffix = '/umap_epoch_{}.png'

    def __call__(self, real, fake, labels, epoch=None, add_hr=False, hr_bounds=None, dataset='blobs3d'):
        """
        Create and save 3D visualization for the current epoch.

        Parameters:
            real (np.array): Real data points.
            fake (np.array): Generated data points.
            labels (np.array): Labels for the data points.
            epoch (int): Current epoch number.
            add_hr (bool): Whether to add hyper-rectangle visualization.
            hr_bounds (list): Bounds for hyper-rectangles.
            dataset (str): Name of the dataset.
        """
        if epoch == 'End_only_fake':
            full_data = fake
            full_labels = labels.argmax(1) if dataset == 'blobs3d' else labels
            real_fake_labels = np.array(["fake"]*len(fake))
        else:
            full_data = np.concatenate((real, fake))
            full_labels = np.concatenate((labels.argmax(1), labels.argmax(1))) if dataset == 'blobs3d' else np.concatenate((labels, labels))
            real_fake_labels = np.array(["real"]*len(real)+["fake"]*len(fake))

        fig = plt.figure(figsize=(14,5))

        # Plot data points colored by class
        ax = fig.add_subplot(131, projection="3d")
        scatter = ax.scatter(full_data[:, 0], full_data[:, 1], full_data[:, 2], c=full_labels, alpha=0.7)
        if add_hr:
            for HR_BOUNDS in hr_bounds:
                HR_BOUNDS = (HR_BOUNDS[0].cpu(), HR_BOUNDS[1].cpu())
                CUBE_BOUNDS = rebuild_bounds(HR_BOUNDS)
                ax = plot_cube_on_ax(ax, CUBE_BOUNDS)
        ax.set_title(f"Epoch {epoch}", size=14, fontdict={'family':'serif'})
        ax.view_init(azim=-66, elev=20)
        self._set_3d_plot_limits(ax, dataset)

        # Plot data points colored by real/fake
        ax2 = fig.add_subplot(132, projection="3d")
        colors = np.where(real_fake_labels == 'real', 'blue', 'red')
        ax2.scatter(full_data[:, 0], full_data[:, 1], full_data[:, 2], c=colors, alpha=0.7)
        if add_hr:
            for HR_BOUNDS in hr_bounds:
                HR_BOUNDS = (HR_BOUNDS[0].cpu(), HR_BOUNDS[1].cpu())
                CUBE_BOUNDS = rebuild_bounds(HR_BOUNDS)
                ax2 = plot_cube_on_ax(ax2, CUBE_BOUNDS)
        ax2.set_title(f"Epoch {epoch}", size=14, fontdict={'family':'serif'})
        ax2.view_init(azim=-66, elev=20)
        self._set_3d_plot_limits(ax2, dataset)

        # Additional view
        ax3 = fig.add_subplot(133, projection="3d")
        ax3.scatter(full_data[:, 0], full_data[:, 1], full_data[:, 2], c=colors, alpha=0.7)
        ax3.set_title(f"Epoch {epoch}", size=14, fontdict={'family':'serif'})
        ax3.view_init(azim=-90 if dataset == 'swiss_roll3d' else -6, elev=20)
        self._set_3d_plot_limits(ax3, dataset)

        plt.tight_layout()
        plt.savefig(self.path + self.path_suffix.format(epoch), format="png")

    def _set_3d_plot_limits(self, ax, dataset):
        """Set 3D plot limits based on the dataset."""
        if dataset in ['swiss_roll3d', 'blobs3d']:
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_zlim(-2.5, 2.5)

class VisuEpochMultiDim:
    """
    A class for visualizing multi-dimensional data using UMAP projections.
    """

    def __init__(self, path: Optional[str] = None):
        """
        Initialize the VisuEpochMultiDim object.

        Parameters:
            path (str, optional): Base path for saving visualization files.
        """
        self.path = path
        self.path_suffix = '/umap_epoch_{}.png'

    def __call__(self, real: np.ndarray, fake: np.ndarray, labels: np.ndarray, 
                 epoch: Optional[int] = None, add_hr: bool = False, 
                 hr_bounds: Optional[np.ndarray] = None, dataset: str = 'tcga_pca'):
        """
        Generate and save UMAP visualizations for real and fake data.

        Parameters:
            real (np.ndarray): Array of real data points.
            fake (np.ndarray): Array of fake (generated) data points.
            labels (np.ndarray): Array of labels for the data points.
            epoch (int, optional): Current epoch number for labeling.
            add_hr (bool): Flag for adding hyper-rectangle bounds (not implemented).
            hr_bounds (np.ndarray, optional): Hyper-rectangle bounds (not used).
            dataset (str): Name of the dataset (not used in current implementation).

        Note:
            The add_hr and hr_bounds parameters are included for compatibility
            but are not currently used in the visualization.
        """
        # Combine real and fake data
        full_data = np.concatenate((real, fake))
        
        # Process labels
        if labels.shape[1] > 1:
            full_labels = np.concatenate((labels.argmax(1), labels.argmax(1)))  # For one-hot encoded labels
        else:
            full_labels = np.concatenate((labels, labels))  # For non-one-hot encoded labels
        
        real_fake_labels = np.array(["real"] * len(real) + ["fake"] * len(fake))

        # Perform UMAP projection
        umap_proj = perform_umap(full_data, n_neighbors=int(0.1 * len(full_data)) + 1)

        # Create the visualization
        fig = plt.figure(figsize=(10, 5))

        # First subplot: UMAP colored by data labels
        plt.subplot(121)
        self._plot_umap_by_labels(umap_proj, full_labels, epoch)

        # Second subplot: UMAP showing real vs fake data
        plt.subplot(122)
        self._plot_umap_real_vs_fake(umap_proj, real_fake_labels, len(real), epoch)

        # Save the figure
        plt.tight_layout()
        if self.path:
            plt.savefig(self.path + self.path_suffix.format(epoch), format="png")
        else:
            plt.show()

    def _plot_umap_by_labels(self, umap_proj: np.ndarray, labels: np.ndarray, epoch: int):
        """
        Plot UMAP projection colored by data labels.

        Parameters:
            umap_proj (np.ndarray): UMAP projection of the data.
            labels (np.ndarray): Labels for each data point.
            epoch (int): Current epoch number.
        """
        plt.scatter(umap_proj[:, 0], umap_proj[:, 1], s=10, cmap='Spectral', c=labels, alpha=0.7)
        plt.title(f"Epoch {epoch}", size=14, fontdict={'family': 'serif'})
        plt.yticks([])
        plt.xticks([])
        plt.colorbar(boundaries=np.arange(len(np.unique(labels)) + 1) - 0.5).set_ticks(np.arange(len(np.unique(labels))))

    def _plot_umap_real_vs_fake(self, umap_proj: np.ndarray, real_fake_labels: np.ndarray, n_real: int, epoch: int):
        """
        Plot UMAP projection distinguishing real and fake data.

        Parameters:
            umap_proj (np.ndarray): UMAP projection of the data.
            real_fake_labels (np.ndarray): Labels indicating real or fake data.
            n_real (int): Number of real data points.
            epoch (int): Current epoch number.
        """
        colors = np.where(real_fake_labels == 'real', 'blue', 'red')
        
        plt.scatter(umap_proj[0, 0], umap_proj[0, 1], s=10, c='blue', alpha=0.7, label='real')
        plt.scatter(umap_proj[n_real, 0], umap_proj[n_real, 1], s=10, c='red', alpha=0.5, label='fake')
        plt.scatter(umap_proj[1:, 0], umap_proj[1:, 1], s=10, c=colors[1:], alpha=0.5)
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3, markerscale=2)
        plt.title(f"Epoch {epoch}", size=14, fontdict={'family': 'serif'})
        plt.yticks([])
        plt.xticks([])

def plot_first_vars(real: np.ndarray, fake: np.ndarray, epoch: int, save_to: Optional[str] = None):
    """
    Plot the first four variables of real and fake data.

    Parameters:
        real (np.ndarray): Array of real data points.
        fake (np.ndarray): Array of fake (generated) data points.
        epoch (int): Current epoch number.
        save_to (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    
    for i, ax in enumerate(axes):
        pc = i * 2
        ax.scatter(real[:, pc], real[:, pc+1], alpha=0.3, label="real data", marker='.', color='blue')
        ax.scatter(fake[:, pc], fake[:, pc+1], alpha=0.3, label="fake data", color='red', marker='.')
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlabel(f'Var {pc+1}', size=10)
        ax.set_ylabel(f'Var {pc+2}', size=10)
        ax.set_title(f'epoch {epoch}', size=10)

    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, format='png')
    else:
        plt.show()

def plot_first_vars_labels(real: np.ndarray, fake: np.ndarray, labels: np.ndarray, 
                           epoch: int, save_to: Optional[str] = None):
    """
    Plot the first four variables of real and fake data, colored by labels.

    Parameters:
        real (np.ndarray): Array of real data points.
        fake (np.ndarray): Array of fake (generated) data points.
        labels (np.ndarray): Array of labels for the data points.
        epoch (int): Current epoch number.
        save_to (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    
    if labels.shape[1] > 1:
        labels = labels.argmax(1)

    for i, ax in enumerate(axes):
        pc = i * 2
        scatter = ax.scatter(real[:, pc], real[:, pc+1], s=10, cmap='Spectral', c=labels, alpha=0.7)
        ax.scatter(fake[:, pc], fake[:, pc+1], s=10, cmap='Spectral', c=labels, alpha=0.7)
        plt.colorbar(scatter, ax=ax, boundaries=np.arange(len(np.unique(labels))+1)-0.5).set_ticks(np.arange(len(np.unique(labels))))
        ax.set_xlabel(f'Var {pc+1}', size=10)
        ax.set_ylabel(f'Var {pc+2}', size=10)
        ax.set_title(f'epoch {epoch}', size=10)

    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, format='png')
    else:
        plt.show()

def plot_first_vars_sampling(real: torch.Tensor, real_counts: torch.Tensor, 
                             dataset: Optional[str] = None, epoch: Optional[int] = None, 
                             save_to: Optional[str] = None):
    """
    Plot the first four variables of real data, colored by their sampling counts.

    Parameters:
        real (torch.Tensor): Tensor of real data points.
        real_counts (torch.Tensor): Tensor of sampling counts for each data point.
        dataset (str, optional): Name of the dataset (not used in current implementation).
        epoch (int, optional): Current epoch number for labeling.
        save_to (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))

    for i, ax in enumerate(axes):
        var = i * 2
        scatter = ax.scatter(real[:, var].cpu().numpy(), 
                             real[:, var+1].cpu().numpy(), 
                             c=real_counts.detach().cpu().numpy(), 
                             alpha=0.5, 
                             cmap='magma')
        ax.set_xlabel(f'Var {var+1}', size=10)
        ax.set_ylabel(f'Var {var+2}', size=10)
        ax.set_title(f'Epoch {epoch}', size=10)
        plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, format='png')
    else:
        plt.show()

def plot_sampling_count(x_real: torch.Tensor, x_count: torch.Tensor, 
                        dataset: Optional[str] = None, epoch: Optional[int] = None, 
                        save_to: Optional[str] = None):
    """
    Plot the sampling count distribution for different datasets.

    Parameters:
        x_real (torch.Tensor): Tensor of real data points.
        x_count (torch.Tensor): Tensor of sampling counts for each data point.
        dataset (str, optional): Name of the dataset to determine plot settings.
        epoch (int, optional): Current epoch number for labeling.
        save_to (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    x_real_np = x_real.detach().cpu().numpy()
    x_count_np = x_count.detach().cpu().numpy()

    if dataset in ['moons_2d', 'circles_2d', 'swiss_roll_2d', 'blobs_2d']:
        _plot_2d_sampling_count(x_real_np, x_count_np, dataset, epoch)
    elif dataset in ['swiss_roll3d', 'blobs3d']:
        _plot_3d_sampling_count(x_real_np, x_count_np, dataset, epoch)
    
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, format='png')
    else:
        plt.show()

def _plot_2d_sampling_count(x_real_np: np.ndarray, x_count_np: np.ndarray, 
                            dataset: str, epoch: Optional[int]):
    """Helper function to plot 2D sampling count."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_real_np[:, 0], x_real_np[:, 1], c=x_count_np, alpha=0.7, cmap='magma')
    plt.colorbar(scatter, location='right')
    
    xlim, ylim, _ = _get_plot_limits(dataset)
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    plt.title(f"Density sampling counts - Epoch {epoch}", size=14, fontdict={'family': 'serif'})

def _plot_3d_sampling_count(x_real_np: np.ndarray, x_count_np: np.ndarray, 
                            dataset: str, epoch: Optional[int]):
    """Helper function to plot 3D sampling count."""
    fig = plt.figure(figsize=(14, 6))
    
    # First 3D Plot
    ax1 = fig.add_subplot(121, projection="3d")
    _plot_3d_scatter(ax1, x_real_np, x_count_np, dataset, epoch, azim=-66, elev=20)
    
    # Second 3D Plot
    ax2 = fig.add_subplot(122, projection="3d")
    azim = -90 if dataset == 'swiss_roll3d' else -6
    scatter = _plot_3d_scatter(ax2, x_real_np, x_count_np, dataset, epoch, azim=azim, elev=20)
    fig.colorbar(scatter, ax=ax2)

def _plot_3d_scatter(ax: plt.Axes, x_real_np: np.ndarray, x_count_np: np.ndarray, 
                     dataset: str, epoch: Optional[int], azim: float, elev: float):
    """Helper function to create a 3D scatter plot."""
    scatter = ax.scatter(x_real_np[:, 0], x_real_np[:, 1], x_real_np[:, 2], 
                         c=x_count_np, alpha=0.7, cmap='magma')
    ax.set_title(f"Epoch {epoch}", size=14, fontdict={'family': 'serif'})
    ax.view_init(azim=azim, elev=elev)
    
    xlim, ylim, zlim = _get_plot_limits(dataset)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    return scatter

def _get_plot_limits(dataset: str):
    """Helper function to get plot limits based on the dataset."""
    if dataset in ['moons_2d', 'circles_2d', 'swiss_roll_2d', 'swiss_roll3d']:
        return (-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)
    elif dataset == 'blobs_2d':
        return (-2.5, 2.5), (-2., 2.), None
    elif dataset == 'blobs3d':
        return (-2.5, 2.5), (-2.5, 2.5), (-2.5, 2.5)
    else:
        return None, None, None

def plot_checkpoints(x_real: torch.Tensor, x_gen: torch.Tensor, full_labels: torch.Tensor,
                     dataset: str, Printer: callable, epoch: int, 
                     hr_bounds: Optional[np.ndarray], fig_dir: str):
    """
    Plot checkpoints for model evaluation.

    Parameters:
        x_real (torch.Tensor): Tensor of real data points.
        x_gen (torch.Tensor): Tensor of generated data points.
        full_labels (torch.Tensor): Tensor of labels for the data points.
        dataset (str): Name of the dataset.
        Printer (callable): Function to print/plot the data.
        epoch (int): Current epoch number.
        hr_bounds (np.ndarray, optional): Hyper-rectangle bounds.
        fig_dir (str): Directory to save figures.
    """
    add_hr = dataset in ['moons_2d', 'circles_2d', 'swiss_roll_2d', 'blobs_2d', 'blobs3d', 'swiss_roll3d']
    
    idx = np.random.choice(np.arange(len(x_real)), min(len(x_real), 1500), replace=False)
    Printer(x_real[idx], x_gen[idx], full_labels[idx], epoch=epoch, add_hr=add_hr, hr_bounds=hr_bounds, dataset=dataset)
    
    if dataset in ['gesture', 'wilt', 'diabetes', 'magic', 'tcga', 'gtex', 'gaussian_multidim']:
        plot_first_vars(x_real[idx], x_gen[idx], epoch=epoch, save_to=f'{fig_dir}/first_vars_epoch_{epoch}.png')
        # TODO: Implement plot_first_vars_labels
        # plot_first_vars_labels(x_real[idx], x_gen[idx], full_labels[idx], epoch=epoch, save_to=f'{fig_dir}/first_vars_labels_epoch_{epoch}.png')