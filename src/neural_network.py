"""
Docstring für src.neural_network

The functions used for the neural network implementation using flax.
"""

import jax
import jax.numpy as jnp
from jax import random, jit
from flax import linen as nn
from typing import Sequence, Callable, Tuple
import numpy as np


# Create the Network Class
class RegulatoryNetwork(nn.Module):
    """
    NN class that inherits most of its attributes from the flax.nn.Module.
    """
    hidden_dims: Sequence[int] = (8, 8, 8) # By default three hidden layers with 8 units each
    activation: Callable = nn.tanh 
    
    @nn.compact
    def __call__(self, x):
        """Forward pass computing f(s̄).
        
        Args:
            x: Input neighbor average s̄, shape (1,) or scalar.
            
        Returns:
            Regulatory function output, scalar.
        """
        # Ensure input has batch dimension for Flax (can handle scalar or array)
        x = jnp.atleast_1d(x)
        
        # Hidden layers with activation
        for dim in self.hidden_dims:
            x = nn.Dense(features=dim)(x) # Dense is the name for a layer with weights + biases
            x = self.activation(x) # Apply activation function
        
        # Output layer (no activation for full range)
        x = nn.Dense(features=1)(x)
        
        # Return as scalar (squeeze out dimensions)
        return jnp.squeeze(x)
    

def init_params(model: RegulatoryNetwork, 
                key: jax.random.PRNGKey,
                input_shape: tuple = (1,)) -> dict:
    """
    Initialise the parameters for the neural network model.
    """
    dummy_input = jnp.ones(input_shape)
    params = model.init(key, dummy_input)
    return params

def get_regulatory_function(model: RegulatoryNetwork, 
                           params: dict) -> Callable:
    """Create callable regulatory function f(s̄) compatible with dynamics.simulate().
    
    Args:
        model: RegulatoryNetwork instance.
        params: Trained network parameters.
        
    Returns:
        Function f(s̄_array) → derivative_array, vmapped over all cells.
    """
    def f(neighbor_avg_array):
        """Regulatory function: neighbor_avg_array → derivatives."""
        # Apply network to each cell's neighbor average
        return jax.vmap(lambda s_bar: model.apply(params, s_bar))(neighbor_avg_array)
    
    return f


def visualise_network_function(
    model: RegulatoryNetwork,
    params: dict,
    s_bar_range: jnp.ndarray = jnp.linspace(0, 1, 100)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute network output over a range of neighbor averages for visualization.
    
    Args:
        model: RegulatoryNetwork instance.
        params: Trained network parameters.
        s_bar_range: 1D array of neighbor average values to evaluate.
        
    Returns:
        Tuple of (s_bar_range, f_values) where f_values are the network outputs.
    """
    f = get_regulatory_function(model, params)
    f_values = f(s_bar_range)
    return s_bar_range, f_values
