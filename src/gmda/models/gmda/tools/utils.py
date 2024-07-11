# Imports
import numpy as np
import torch
import random
import json
from pathlib import Path
from typing import Dict, Any

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

class TrackLoss:
    """Track all components of loss."""
    def __init__(self, verbose:bool=False, trace_func=print, path:str='history.npy', nb_epochs:int=0, save_history:bool=False):
        """
        Parameters:
            verbose (bool): If True, prints a message for each validation loss improvement (default False)
            path (str): Path for the losses to be saved to (default 'wgan_history.npy')
            trace_func (function): trace print function (default print)
        """
        self.verbose = verbose
        self.save_final_history = save_history
        self.path = path
        self.trace_func = trace_func
        self.nb_epochs = nb_epochs
        self.history = {"g_loss_batch":[], 
                        "g_loss_epoch":[], 
                        "g_loss_hr":[], 
                        "fake_probs":[], 
                        "real_probs":[],
                         "loss_hr":[], 
                         "dark_bin_loss":[],
                         }

    def __call__(self, hist_dict:dict):
        """
        Main function to call to save current training history.
        ----
        Parameters:
            hist_dict (dict): current training history dictionary
        """
        for key in hist_dict.keys():
            self.history[key].append(hist_dict[key])
            
        if len(self.history["g_loss_epoch"]) ==self.nb_epochs:
            # Save current training history if not HP* search
            if self.save_final_history:
                self.save_history()


    def save_history(self):
        '''Saves training history'''
        if self.verbose:
            self.trace_func(f'All losses components tracked and saved.')
        np.save(self.path, self.history)

def load_json_config(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents as a Python dictionary.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        Dict[str, Any]: A dictionary containing the JSON file's contents.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        IOError: If there's an error reading the file.
    """
    try:
        # Convert the file path to a Path object
        path = Path(file_path)

        # Check if the file exists
        if not path.is_file():
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Open and read the file
        with path.open('r') as file:
            # Load the JSON config
            config = json.load(file)

        return config

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON in {file_path}: {str(e)}", e.doc, e.pos)
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {str(e)}")
    
def get_config(path):
    """Retrieve config at given path."""
    return load_json_config(path)