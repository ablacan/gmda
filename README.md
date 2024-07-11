# GMDA: Generative Modeling Density Alignment

![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/version-0.1.0-green.svg)

GMDA is a Python package for generative modeling with density alignment. This README provides instructions for installation, usage, and key features of the package.

## Table of Contents
1. [Installation](#installation)
2. [Data Processing](#data-processing)
3. [Model Training](#model-training)
4. [Generating Synthetic Data](#generating-synthetic-data)
5. [Metrics](#metrics)
6. [Command-Line Usage](#command-line-usage)

## Installation

**Disclaimer:** Installing this package will result in the installation of a specific version of PyTorch, which may not be compatible with every user's GPU driver. Before installation, please check the compatibility of the included PyTorch version with your GPU driver. If incompatible, you should create your Python environment with the PyTorch version best suited for your system. Visit the official PyTorch website to find the appropriate installation command for your setup.

### Using a Virtual Environment

```bash
python -m venv env_gmda
source env_gmda/bin/activate
pip install .
```

### Using conda
```bash
conda create -n env_gmda python=3.9
conda activate env_gmda
pip install .[conda]
```

For develpoment mode, use:
```bash
pip install -e .[conda]
```

## Data Processing

GMDA provides flexible data processing capabilities through the DataProcessor class.

```python
from gmda.data_utils import DataProcessor

# Define custom data loading and processing functions
def custom_data_loader(train: bool = True, **kwargs):
    # Your custom data loading logic here. Should retun a tuple of tabular data (X, y).
    pass

def custom_data_processor(data, train: bool = True, **kwargs):
    # Your custom data processing logic here. Should retun a tuple of tuples of processed train and test data ((X_train, y_train), (X_test, y_test)).
    pass

# Instantiate the DataProcessor
data_processor = DataProcessor(custom_data_loader, custom_data_processor)

# Create dataloaders
train_loader, val_loader, X, y = data_processor.create_dataloaders(batch_size=64, density=0.1)
```

## Model Training

To train a GMDA model:
```python
from gmda.models import GMDARunner
from gmda.models.gmda.tools import get_config

# Load configuration
config = get_config('path/to/config.json')

# Initialize and train the model
model = GMDARunner(config)
model.train(train_loader, val_loader, X, config['training'])
```

## Generating Synthetic Data
From a Trained Model:

```python
X_synthetic, y_synthetic = model.generate(y)
X_synthetic, y_synthetic = X_synthetic.numpy(), y_synthetic.numpy()
```

From a Pretrained Model:

```python
from gmda.models import generate_from_pretrained

X_synthetic, y_synthetic = generate_from_pretrained(
    y, 
    config['model'], 
    path_pretrained=model.checkpoint_dir,
    device=config['model']['device'], 
    return_as_array=True
)
```

## Metrics

GMDA provides metrics to evaluate the quality of generated data:

```python
from gmda.metrics import get_corr_error, get_precision_recall
import numpy as np

# Correlation Error
idx = np.random.choice(np.arange(len(X)), size=min(len(X), 1500), replace=False)
corr_error, corr_error_matrix = get_corr_error(X[idx], X_synthetic[idx])

# Precision/Recall
precision, recall = get_precision_recall(X, X_synthetic, nb_nn=config['training']['nb_nn_for_prec_recall'])
```

## Command-Line Usage

GMDA can be run from the command line:

```bash
python main.py --dataset '<DATASET>' \
               --path_train '<PATH/TO/TRAIN/CSV>' \
               --path_test '<PATH/TO/TEST/CSV>' \
               --device 'cuda:0' \
               --config '<PATH/TO/CONFIG/JSON>' \
               --output_dir '<PATH/TO/OUTPUT/RESULTS>' \
               --compute_metrics \
               --save_generated
```

For more details on command-line options, run:
```bash
python main.py --help
```

## Contributing
We welcome contributions! Please contact me for more details.

## License
This project is licensed under the MIT License.