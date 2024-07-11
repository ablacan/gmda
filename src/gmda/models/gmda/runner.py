import os
import sys
import random
import time as t
import csv
import datetime
from typing import Dict, Any
import numpy as np
import torch
import torch.optim
from rich.console import Console
from rich.table import Table

from .generator import Generator
from .tools.utils import TrackLoss
from .tools.utils_sampling import DBSSampler, indicator_func, compute_hr_coverage, maxabs_darkbin_func, maxabs_loss_func_hr
from .visualization.utils_visualization import plot_checkpoints, VisuEpoch3D, VisuEpochMultiDim, VisuEpoch, plot_sampling_count, plot_first_vars_sampling
from ...metrics import get_corr_error, get_precision_recall

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

class GMDARunner:
    """
    GMDA (Generative Model with Dimension Alignment) runner class.
    This class handles the initialization, training, loading, and generation processes.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GMDARunner.

        Parameters:
            config (Dict[str, Any]): Configuration dictionary containing model architecture and training parameters.
        """
        self.device = config['device']

        # Initialize generator
        self.G = Generator(**config['model']).to(self.device)

        # Initialize loss tracker
        self.LossTracker = TrackLoss()

    def load_generator(self, path: str = None, location: str = "cpu"):
        """
        Load a previously trained generator model.

        Parameters:
            path (str): Path to the saved model.
            location (str): Device to load the model to ('cpu' or 'cuda').

        Raises:
            AssertionError: If no path is provided.
            FileNotFoundError: If the model file is not found at the given path.
        """
        assert path is not None, "Please provide a path to load the Generator from."
        try:
            self.G.load_state_dict(torch.load(path, map_location=location))
            print('Generator loaded successfully.')
        except FileNotFoundError:
            print(f"No previously trained weights found at the given path: {path}.")

    def train_gen(self, x, X, z, x_lower_bounds, x_upper_bounds, batch_idx, labels,
                  hr_sampler=None, epoch=0, ite=0):
        """
        Train the generator for one iteration.

        Parameters:
            x (torch.Tensor): Batch of real data.
            X (torch.Tensor): Full dataset.
            z (torch.Tensor): Batch of latent variables.
            x_lower_bounds (torch.Tensor): Lower bounds for hyper-rectangles.
            x_upper_bounds (torch.Tensor): Upper bounds for hyper-rectangles.
            batch_idx (torch.Tensor): Batch indices.
            labels (torch.Tensor): Labels for the batch.
            hr_sampler: Hyper-rectangle sampler.
            epoch (int): Current epoch number.
            ite (int): Current iteration number.

        Returns:
            float: Generator loss for this iteration.
        """
        self.G.train()
        self.G_optimizer.zero_grad()

        # Generate fake data
        gen_outputs = self.G(z, labels)

        # Sample hyper-rectangle bounds
        hr_bounds, idx_to_keep = hr_sampler(x, x_lower_bounds, x_upper_bounds, batch_idx)
        self.hr_bounds = [(hr_bounds[0][i], hr_bounds[1][i]) for i in range(hr_bounds[0].shape[0])]

        # Compute probabilities for real and fake data to fall within the hyper-rectangles
        probs_real, probs_fake = self._compute_probabilities(x, gen_outputs, hr_bounds, idx_to_keep)

        # Update sampling probabilities
        self._update_sampling_probabilities(hr_sampler)

        # Compute new HR bounds for next step
        self.probs_sample_real = hr_sampler.compute_updated_bounds(x_lower_bounds, 
                                                                   x_upper_bounds, 
                                                                   labels, 
                                                                   self.probs_sample_real, 
                                                                   self.probs_sample_dims, 
                                                                   batch_idx)

        # Compute losses
        dark_bin_loss = self.weight_db * self.dark_bin_func(probs_real, probs_fake, labels, labels, is_label_cat=self.is_label_cat)
        hr_loss = self.loss_func(probs_real, probs_fake, labels, labels, is_label_cat=self.is_label_cat)

        # Compute total loss and update parameters
        loss = hr_loss.mean() + dark_bin_loss
        loss.backward()
        self.G_optimizer.step()

        # Store HR loss
        hr_sampler.memory['hr_losses'] = hr_loss.detach()

        # Track loss components
        self._track_loss_components(hr_loss, dark_bin_loss)
        return loss.detach().item()
    
    def train(self, TrainDataLoader, ValDataLoader, X_full=None, config : Dict[str, Any] = {}):
        """
        Main training function for the GMDA model.

        Parameters:
            TrainDataLoader: DataLoader for training data.
            ValDataLoader: DataLoader for validation data.
            X_full (torch.Tensor): Full dataset.
            config (Dict[str, Any]) : dict of hyperparameters.
        """
        # Extract configuration parameters from config with default values
        z_dim = self.G.latent_dim
        epochs = config.get('epochs')
        step = config.get('step', 5)
        self.verbose = config.get('verbose', True)
        checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        log_dir = config.get('log_dir', './logs/')
        fig_dir = config.get('fig_dir', './figures')
        optimizer = config.get('optimizer', 'adam')
        lr_g = config.get('lr_g', 5e-3)
        hyperparameters_search = config.get('hyperparameters_search', False)
        self.HP_SEARCH = hyperparameters_search
        
        # Initialize training settings
        self._init_training_settings(config, X_full)

        # Initialize optimizers and directories
        self._init_train(log_dir, checkpoint_dir, fig_dir, optimizer, lr_g, epochs)

        # Initialize metrics history
        metrics_history = self._init_metrics_history()

        # Initialize HR sampler
        hr_sampler = self._init_hr_sampler(X_full)

        # Main training loop
        self.t_begin = t.time()
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            epoch_loss = self._train_epoch(TrainDataLoader, X_full, z_dim, hr_sampler)

            # Write logs
            self._write_log(epoch, epoch_loss, self.history_loss)
            # Checkpoint 
            if epoch % step == 0:
                self._checkpoint_and_metrics(TrainDataLoader, ValDataLoader)
                self._log_and_print_metrics(epoch, epoch_loss)
                
        # End of training
        self._end_training()
    
    def generate(self, labels: torch.Tensor) -> tuple:
        """
        Generate fake data using the trained generator.

        Parameters:
            labels (torch.Tensor): Labels to condition the generation on.

        Returns:
            tuple: Generated fake data and the corresponding labels.
        """
        self.G.eval()
        labels = labels.to(self.device)
        batch_z = torch.randn(len(labels), self.G.latent_dim, device=self.device)

        with torch.no_grad():
            fake = self.G(batch_z, labels).cpu()

        self.G.train()
        return fake, labels.cpu()
    
    def _real_fake_data(self, DataLoader):
        """
        Returns real data and generated data.
        ----
        Parameters:
            DataLoader (pytorch loader): data loader.
        Returns:
            x_gen (np.array): all generated data
            full_batches (array): all real gene expression data in loader
        """

        x_gen = []
        x_real = []
        full_labels = []
        self.G.eval() # Evaluation mode

        with torch.no_grad():
            for x in DataLoader:
                labels = x[1].to(self.device)
                batch_z = torch.normal(0, 1, size=(labels.shape[0], self.G.latent_dim), device=self.device)
                gen_outputs = self.G(batch_z, labels)
                x_gen.append(gen_outputs.detach().cpu())
                x_real.append(x[0])
                full_labels.append(labels.detach().cpu())

        # Concatenate 
        x_gen = torch.cat(x_gen, 0) 
        x_real = torch.cat(x_real, 0) 
        full_labels = torch.cat(full_labels, 0) 
        x_gen = x_gen.numpy()
        x_real = x_real.numpy()
        full_labels = full_labels.numpy()

        self.G.train()

        return x_real, x_gen, full_labels
    
    def _init_training_settings(self, config, X):
        # Init uniform sampling probabilities for each true point
        self.probs_sample_real = torch.ones((X.shape[0]), device=self.device)/X.shape[0]
        # Init uniform sampling probabilities for each dimension
        self.probs_sample_dims = torch.ones((X.shape[1]), device=self.device)/X.shape[1]

        # Init
        self.dataset = config['dataset']
        self.batch_size = config['batch_size']
        self.lr_decay = config['lr_decay']
        self.save_final_weights = config['save_final_weights']
        # GMDA hyperparameters
        self.weight_db = config['is_dark_bin']
        self.eta = config['eta']
        self.loss_type = config['loss_type']
        self.loss_func = maxabs_loss_func_hr
        self.dark_bin_func = maxabs_darkbin_func
        self.nb_hr = config['K']
        self.nb_dim_to_keep = config['nb_dim_to_keep']
        self.density = config['density']
        self.lamb = config['lambda']
        self.hr_bounds = []
        self.epsilon = 0.9
        self.is_label_cat = config['is_label_cat']
        self.indic_func = config['indic_func']
        self.sampling = config['sampling']
        self.sampling_update = 'uniform'
        # Metrics
        self.nb_nn = config['nb_nn_for_prec_recall']
        self.track_precision_recall = config['track_precision_recall']
        self.track_corr_error = config['track_corr_error']
        self.epochs_checkpoints = config['epochs_checkpoints']
        # Plots
        self.plot_generated_data = config['plot_generated_data']
        self.plot_sampling = config['plot_sampling']
        if '2d' or '3d' in self.dataset:
            self.plot_sampling_func = plot_sampling_count
        else:
            self.plot_sampling_func = plot_first_vars_sampling

    def _init_train(self, log_dir:str, checkpoint_dir:str, fig_dir:str, optimizer:str='rms_prop', 
                   lr_g:float=5e-4, epochs:int=None):
        """
        Training initialization: init directories and callbacks.
        ----
        Parameters:
            log_dir (str): directory where logs will be stored
            checkpoint_dir (str): directory where weights will be stored 
        """
    
        # Optimizers
        if optimizer.lower()=='rms_prop':
            self.G_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=lr_g)
        
        elif optimizer.lower()=='adam':
            self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=(.9, .99))

        elif optimizer.lower()=='sgd':
            self.G_optimizer = torch.optim.SGD(self.G.parameters(), lr=lr_g, momentum=0.9)
        
        if self.lr_decay:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.G_optimizer, 50, gamma=0.98, last_epoch=- 1, verbose=True)

        # Set up logs and checkpoints
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = log_dir + self.dataset +'_'+ current_time

        # Make dir if it does not exist
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
            print("Directory '%s' created" %self.log_dir)

        # Init log path
        self.history_loss = self.log_dir+ "/train_loss.csv"
        self.history_metrics = self.log_dir+ "/train_metrics.csv"

        if os.path.exists(self.history_loss):
                os.remove(self.history_loss)
        if os.path.exists(self.history_metrics):
                os.remove(self.history_metrics)

        # Init log
        with open(self.history_loss, "w") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "loss", "lr", "epoch_time", "total_time"])
            f.close()

        # Init log metrics
        list_rows = ['epoch', 'loss', 'epoch_time', 'duration']
        if self.track_corr_error:
            list_rows.append('corr. error')
        if self.track_precision_recall:
            list_rows.append('precision')
            list_rows.append('recall')
        with open(self.history_metrics, "w") as f:
            w = csv.writer(f)
            w.writerow(list_rows)
            f.close()

        # Define checkpoints path
        if not self.save_final_weights:
            self.checkpoint_dir = ''
            self.G_path = self.checkpoint_dir +'/_gen.pt'
            print("No directory created. Weights will not be saved at the end of training.")
        else:
            self.checkpoint_dir = checkpoint_dir + '/' + self.dataset + '_' + current_time
            # Make dir if it does not exist
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            if not os.path.exists(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)
                print("Directory '%s' created" %self.checkpoint_dir)

            # Initialize the paths for models
            self.G_path = self.checkpoint_dir +'/_gen.pt'

        # Create figures folder
        self.fig_dir = fig_dir + '/' + self.dataset + '_' + current_time
        # Make dir if it does not exist
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)
        if not os.path.exists(self.fig_dir):
            os.mkdir(self.fig_dir)
            print("Directory '%s' created" %self.fig_dir)

        # Initialize the tracking object for loss components
        self.LossTracker = TrackLoss(path=self.log_dir+'/train_history.npy', nb_epochs=epochs, save_history=self.save_final_weights)

        # Initialize visualization printer (to save figures of generated data during training)
        if '3d' in self.dataset:
            self.Printer = VisuEpoch3D(path=self.fig_dir)
        
        elif "2d" in self.dataset:
            self.Printer = VisuEpoch(path=self.fig_dir) 
        
        else:
            self.Printer = VisuEpochMultiDim(path=self.fig_dir)

    def _init_metrics_history(self):
        """Init training metrics history"""
        self.all_epochs_val_score = []
        self.all_epochs_prec_recall_train = []

    def _init_hr_sampler(self, X):
        """HR sampler initialization"""
        return DBSSampler(self.nb_dim_to_keep, self.nb_hr,
                                 self.batch_size, self.probs_sample_dims,
                                  self.probs_sample_real,
                                 eta=self.eta)
    
    def _train_epoch(self, TrainDataLoader, X, z_dim, hr_sampler):
        """ Training epoch. """
        
        epoch_gen_loss = 0
        self.start_epoch_time = t.time()

        # Load batches of expression data, numerical covariates and encoded categorical covariates
        for i, (batch, labels, low_bounds, up_bounds, batch_idx) in enumerate(TrainDataLoader):
            
            batch = batch.to(self.device)
            labels = labels.to(self.device)
            low_bounds = low_bounds.to(self.device)
            up_bounds = up_bounds.to(self.device)
            batch_idx = batch_idx.to(self.device)
            
            ####### Train generator #######
            # Get random latent variables z
            batch_z = torch.normal(0, 1, size=(batch.shape[0], z_dim), device=self.device)
            gen_loss = self.train_gen(batch,
                                    X,
                                    batch_z,
                                    low_bounds,
                                    up_bounds,
                                    batch_idx,
                                    labels,
                                    hr_sampler=hr_sampler,
                                    epoch=self.epoch,
                                    ite = i)
            epoch_gen_loss += gen_loss
            self.LossTracker({"g_loss_batch": gen_loss})

        self.end_epoch_time = t.time()
        self.epoch_time = self.end_epoch_time - self.start_epoch_time
        self.train_time = self.end_epoch_time - self.t_begin
        epoch_gen_loss = epoch_gen_loss/len(TrainDataLoader)

        # LR decay
        if self.lr_decay:
            self.scheduler.step()

        # Store all losses
        self.LossTracker({"g_loss_epoch": epoch_gen_loss})

        # Checkpoint plots
        if self.epoch in self.epochs_checkpoints:
            if self.plot_generated_data:
                # Plot generated data
                self._visualize_generated_data(TrainDataLoader)      
            if self.plot_sampling:
                # Plot sampling
                self._visualize_sampling_distribution(hr_sampler, X) 

        return epoch_gen_loss

    def _checkpoint_and_metrics(self, TrainDataLoader, ValDataLoader):
        """
        Compute metrics of interest at given epochs checkpoints.
        ----
        Parameters:
            TrainDataLoader (pytorch loader): train data loader
            ValDataLoader (pytorch loader): validation data loader
        Returns:
        """

        # Compute validation score
        if self.track_corr_error:
            # Val data
            x_real, x_gen, _ = self._real_fake_data(ValDataLoader)
            val_score,_ = get_corr_error(x_real, x_gen)
            self.epoch_val_score = val_score
        
        if self.track_precision_recall:
            # Train data
            x_real, x_gen, _ = self._real_fake_data(TrainDataLoader)
            # Precision/recall on training data
            prec, recall = get_precision_recall(x_real, x_gen, self.nb_nn)
            self.epoch_prec_recall_train = (prec, recall)

    def _compute_probabilities(self, x, gen_outputs, hr_bounds, idx_to_keep):
        """Compute probabilities over all HRs (final prob = product of probabilities over dimensions).
        Final shape: (H, B)"""
        if self.nb_dim_to_keep != x.shape[1]:
            probs_real_3d = indicator_func(x[:,idx_to_keep].permute(1,0,2), hr_bounds[0].view(-1, 1, self.nb_dim_to_keep), hr_bounds[1].view(-1, 1, self.nb_dim_to_keep), lamb=self.lamb)
            probs_fake_3d = indicator_func(gen_outputs[:, idx_to_keep].permute(1,0,2), hr_bounds[0].view(-1, 1, self.nb_dim_to_keep), hr_bounds[1].view(-1, 1, self.nb_dim_to_keep), lamb=self.lamb)
            probs_real = torch.prod(probs_real_3d, -1)
            probs_fake = torch.prod(probs_fake_3d, -1)
        else:
            probs_real_3d = indicator_func(x, hr_bounds[0].view(-1, 1, x.shape[1]), hr_bounds[1].view(-1, 1, x.shape[1]), lamb=self.lamb)
            probs_fake_3d = indicator_func(gen_outputs, hr_bounds[0].view(-1, 1, x.shape[1]), hr_bounds[1].view(-1, 1, x.shape[1]), lamb=self.lamb)
            probs_real = torch.prod(probs_real_3d, -1)
            probs_fake = torch.prod(probs_fake_3d, -1)
        return probs_real, probs_fake

    def _update_sampling_probabilities(self, hr_sampler):
        """Update sampling weights for each true data point and each dims."""
        if self.sampling_update=='uniform':
            pass
        #TODO implement other sampling updates

    def _visualize_sampling_distribution(self, hr_sampler, X):
        """Plot sampling of the distribution."""
        # Points sampling distribution
        all_idx = torch.hstack(hr_sampler.memory['hr_centers'])
        uniques, counts = torch.unique(all_idx, return_counts = True)
        self.plot_sampling_func(X[uniques], counts, dataset=self.dataset, epoch=self.epoch, save_to=self.fig_dir+f'/sampling_count_epoch_{self.epoch}.png')
    
    def _visualize_generated_data(self, trainloader):
        """ Plot generated data."""
        x_real, x_gen, full_labels = self._real_fake_data(trainloader)
        # Plot distribution
        plot_checkpoints(x_real, x_gen, full_labels, self.dataset, self.Printer, self.epoch, self.hr_bounds, self.fig_dir)
 
    def _track_loss_components(self, hr_loss, dark_bin_loss):
        # Track loss components
        self.LossTracker({"g_loss_hr": hr_loss.mean().detach().item(),
                          "dark_bin_loss": dark_bin_loss.detach().item()})        

    @staticmethod
    def _print_func(epoch: int, verbose_dict: Dict[str, float]):
        """Print checkpoint information."""
        table = Table(title=f"Epoch {epoch} checkpoint")
        table.add_column("Metric", justify="right", style="blue", no_wrap=True)
        table.add_column("Value", style="red")
        for k, v in verbose_dict.items():
            table.add_row(f"{k}", f"{v:.4f}")
        Console().print(table)

    def _write_log(self, epoch, loss, file_path:str=None):
        """ Write training logs in given path"""

        csv_info = []
        csv_info.append(epoch)
        csv_info.append("%.5f" % loss)
        csv_info.append("%.10f" % self.G_optimizer.param_groups[0]['lr']) 
        csv_info.append("%.3f" % self.epoch_time)
        csv_info.append("%.3f" % (t.time() - self.t_begin))

        #save csv
        with open(file_path, "a") as f:
            w = csv.writer(f)
            w.writerow(csv_info)
            f.close()

    def _write_log_metrics(self, epoch,  metrics_dict, file_path:str=None):
        """ Write training logs in given path"""
        csv_info = []
        csv_info.append(epoch)
        for k in metrics_dict.keys():
            csv_info.append("%.3f" % metrics_dict[k])

        #save csv
        with open(file_path, "a") as f:
            w = csv.writer(f)
            w.writerow(csv_info)
            f.close()

    def _log_and_print_metrics(self, epoch: int, epoch_loss: float):
        """Log and print metrics."""
        verbose_dict = {
            'loss': epoch_loss,
            'epoch time': self.epoch_time,
            'duration': self.train_time
        }
        if self.track_corr_error:
            verbose_dict['corr. error'] = self.epoch_val_score
        if self.track_precision_recall:
            verbose_dict['precision'] = self.epoch_prec_recall_train[0]
            verbose_dict['recall'] = self.epoch_prec_recall_train[1]

        self._write_log_metrics(epoch, verbose_dict, self.history_metrics)
        if self.verbose:
            self._print_func(epoch, verbose_dict)

    def _end_training(self):
        """Finalize training process."""
        self.t_end = t.time()
        self.time_sec = self.t_end - self.t_begin
        print(f'Training time: {self.time_sec:.2f} sec = {self.time_sec/60:.2f} min = {self.time_sec/3600:.2f} hours')
         
        # Save last weigths 
        if self.save_final_weights:
            self._save_weights()
            print(f"-------\n Generator saved at {self.G_path}.")

    def _save_weights(self):
        """Save current weights of the generator."""
        torch.save(self.G.state_dict(), self.G_path)


