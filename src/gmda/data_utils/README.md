This README provides a clear guide on how to use the `DataProcessor` class, including initialization, data preparation, and creating DataLoaders.

# Data Processor

The `DataProcessor` class is designed to handle the loading and processing of custom datasets for machine learning tasks. It provides flexibility by allowing users to define their own data loading and processing functions.

## Usage

### Initialization

To use the `DataProcessor`, you need to provide two custom functions:

1. `custom_data_loader`: A function to load your data.
2. `custom_data_processor`: A function to process your data.

Example:

```python
def my_data_loader(train=True, **kwargs):
    # Your data loading logic here
    return data

def my_data_processor(data, train=True, **kwargs):
    # Your data processing logic here
    return X, y

data_processor = DataProcessor(my_data_loader, my_data_processor)
```

### Preparing Data
To prepare your data, use the prepare_data method:

```python
X_train, y_train = data_processor.prepare_data(train=True)
X_val, y_val = data_processor.prepare_data(train=False)
```
### Creating DataLoaders
To create PyTorch DataLoader objects as inputs to the GMDA algorithm, use the create_dataloaders method:

```python
train_loader, val_loader, X, y = data_processor.create_dataloaders(batch_size=64, density=0.1)
```

For the usual dataloaders (e.g. used as inputs for a comparison model), do not specify the `density` argument:
```python
train_loader, val_loader, X, y = data_processor.create_dataloaders(batch_size=64)
```

#### Parameters
- `batch_size` (int): The batch size for the DataLoaders.
- `density` (float, optional): Density value for hyper-rectangles width. If provided, the DataLoaders will include lower and upper bounds for each sample.

## Notes

Ensure your custom data loader and processor functions are compatible with the expected input and output formats.
The create_dataloaders method automatically converts numpy arrays to PyTorch tensors.
When density is provided, the DataLoaders will include additional tensors for lower and upper bounds of hyper-rectangles.

