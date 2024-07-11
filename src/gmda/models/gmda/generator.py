import random
import torch 
import torch.nn as nn
from typing import Optional

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)

class Generator(nn.Module):
    """
    Generator class for creating synthetic data.

    This class implements a flexible generator network that can be used
    in GMDA, generative adversarial networks (GANs) or other generative models.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Generator.

        Parameters:
            **kwargs: Dictionary containing the following keys:
                latent_dim (int): Dimension of latent noise vector z.
                output_dim (int): Dimension of generated data (number of features).
                labels_dim (int): Dimension of condition labels.
                hidden_dims (list of int): List of hidden layer dimensions.
                activation_func (str, optional): Activation function to use ('relu' or 'leaky_relu'). Defaults to 'relu'.
                negative_slope (float, optional): Negative slope for LeakyReLU. Defaults to 0.1.
                embed_cat (bool, optional): Whether to use embedding for categorical variables. Defaults to False.
                embedding_size (int, optional): Size of embedding vectors. Defaults to 0.
                vocab_size (int, optional): Size of vocabulary for embedding. Defaults to 2.
        """
        super().__init__()

        # Extract parameters from kwargs with default values
        self.latent_dim = kwargs['latent_dim']
        self.output_dim = kwargs['output_dim']
        self.labels_dim = kwargs['labels_dim']
        self.hidden_dims = kwargs['hidden_dims']
        activation_func = kwargs.get('activation_func', 'relu')
        negative_slope = kwargs.get('negative_slope', 0.1)
        self.embed_cat = kwargs.get('embed_cat', False)
        self.embedding_size = kwargs.get('embedding_size', 0)
        self.vocab_size = kwargs.get('vocab_size', 2)

        if self.embed_cat:
            self.labels_dim = self.embedding_size * self.vocab_size
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        self.input_dim = self.latent_dim + self.labels_dim

        # Set activation function
        self.activation_func = nn.LeakyReLU(negative_slope) if activation_func.lower() == 'leaky_relu' else nn.ReLU()
        # Define network layers
        self.layers = nn.ModuleList()
        in_features = self.input_dim 
        
        for hidden_dim in self.hidden_dims:
            self.layers.append(nn.Linear(in_features, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(self.activation_func)
            in_features = hidden_dim + self.labels_dim  # For skip connections

        # Output layer
        self.proj_output = nn.Linear(in_features, self.output_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic data from random noise input.

        Parameters:
            x (torch.Tensor): Input latent noise vector.
            c (torch.Tensor): Conditioning variable.

        Returns:
            torch.Tensor: Generated synthetic data.
        """
   
        if self.embed_cat:
            c = self.embedding(c)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = torch.cat([x, c.view(c.size(0), -1)], dim=1)
            x = layer(x)

        x = torch.cat([x, c.view(c.size(0), -1)], dim=1)
        return self.proj_output(x)
    

def generate_from_pretrained(
    y: torch.Tensor,
    config: dict,
    path_pretrained: str,
    device: str = 'cpu',
    return_as_array: bool = False
) -> tuple:
    """
    Generate fake samples using a pretrained generator model.

    Parameters:
        y (torch.Tensor): Condition tensor for generation.
        config (dict): Configuration dictionary for the generator.
        path_pretrained (str): Path to the pretrained generator weights.
        device (str, optional): Device to run the model on. Defaults to 'cpu'.
        return_as_array (bool, optional): If True, return numpy arrays instead of tensors. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - fake samples (torch.Tensor or np.ndarray)
            - condition y (torch.Tensor or np.ndarray)

    Raises:
        AssertionError: If y is not a torch.Tensor.
        FileNotFoundError: If the pretrained weights file is not found.
    """
    # Initialize the generator with the provided configuration
    generator = Generator(**config)

    # Load pretrained weights
    try:
        generator.load_state_dict(torch.load(path_pretrained, map_location=device))
        print('Generator loaded successfully.')
    except FileNotFoundError:
        print(f"No previously trained weights found at the given path: {path_pretrained}.")
        raise

    # Move generator to the specified device
    generator = generator.to(device)
    generator.eval()  # Set the generator to evaluation mode

    # Convert y to torch.Tensor if it's not already
    if not isinstance(y, torch.Tensor):
        try:
            y = torch.tensor(y)
        except Exception as e:
            raise ValueError(f"Failed to convert y to torch.Tensor: {str(e)}")
    y = y.to(device)

    # Generate random latent vectors
    batch_size = len(y)
    batch_z = torch.randn(batch_size, generator.latent_dim, device=device)

    # Generate fake samples
    with torch.no_grad():  # Disable gradient computation for inference
        fake = generator(batch_z, y).cpu()

    # Convert to numpy arrays if requested
    if return_as_array:
        fake = fake.numpy()
        y = y.cpu().numpy()
    else:
        y = y.cpu()

    return fake, y