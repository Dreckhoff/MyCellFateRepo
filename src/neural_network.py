"""
Docstring für src.neural_network

The functions used for the neural network implementation using flax.
"""

import jax
import jax.numpy as jnp
from jax import random, jit
from flax import linen as nn
from typing import Sequence, Callable, Tuple, Dict
import numpy as np
from functools import partial


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


"""
For the neural network, each set of system replicates is using a set of model weights and biases which we call a "population". 
It is the flattened model parameter structure

population = jnp.array([flatten_params(init_params(model, key, (1,)))])
"""

@jit
def flatten_params(params: Dict) -> jnp.ndarray:
    """Flatten nested parameter dict to 1D array for evolution (JIT-compiled)."""
    leaves = jax.tree_util.tree_leaves(params)
    return jnp.concatenate([leaf.flatten() for leaf in leaves])

@jit
def unflatten_params(flat: jnp.ndarray, template: Dict) -> Dict:
    """Reconstruct parameter dict from flat array using template structure (JIT-compiled).
    
    Uses lax.dynamic_slice for dynamic indexing within JIT.
    """
    # Get shapes from template (these are static, not traced)
    shapes = [x.shape for x in jax.tree_util.tree_leaves(template)]
    # Compute sizes from shapes - use numpy prod to avoid tracing
    sizes = [int(np.prod(s)) for s in shapes]  # Use numpy, not jnp
    
    # Compute cumulative indices for slicing
    indices = np.cumsum([0] + sizes[:-1])  # [0, size0, size0+size1, ...]
    
    # Split flat array using dynamic_slice - functional style with list comprehension
    arrays = [
        jax.lax.dynamic_slice(flat, (int(idx),), (size,)).reshape(shape)
        for idx, size, shape in zip(indices, sizes, shapes)
    ]
    
    # Reconstruct tree structure
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template), arrays)


# Now define the function that evaluate the fitness for the neural network
# First import necessary functions
from dynamics import run_multiple_replicates, run_multiple_replicates_fixed_boundary ,apply_threshold
from utility_function import compute_hard_utility

# Define the TrainingSetupDictionary (here a dummy)

SetupDict = {
    'N_CELLS': 7,
    'N_REPLICATES': 50,
    'N_STEPS': 200,
    'DT': 0.1,
    'NOISE_STRENGTH': 0.05,
}

# This function is given a set of parameters (from the population). It runs the specified simulations and 
# computes the fittnes for this set of parameters and outputs it.
# @partial(jit, static_argnames=['model', 'SetupDict'])
@partial(jit, static_argnames=['model', 'N_CELLS', 'N_REPLICATES', 'N_STEPS', 'DT', 'NOISE_STRENGTH'])
def compute_fitness(params: Dict, model: RegulatoryNetwork, 
                    N_CELLS, N_REPLICATES, N_STEPS, DT, NOISE_STRENGTH, 
                    eval_key: jax.random.PRNGKey) -> tuple[float, float, float]:
    """
    Evaluate utility (fitness) for given network parameters.
    
    Higher utility = better fitness.
    Uses hard utility, non differentiable, used for ES (Evolution & Selection)
    Args:
        params: Network parameters as a dict.
        model: RegulatoryNetwork instance defining architecture. (static)
        N_CELLS: Number of cells in the simulation. (static)
        N_REPLICATES: Number of replicates to run. (static)
        N_STEPS: Number of time steps in each simulation. (static)
        DT: Time step size for simulations. (static)
        NOISE_STRENGTH: Noise strength in simulations. (static)
        eval_key: JAX PRNGKey for stochastic simulations.
    Returns:
        utility: Computed utility (fitness) score.
        s_pat: Pattern diversity score.
        s_rep: Replicate consistency score.
    """
    # Get regulatory function from params
    func = get_regulatory_function(model, params)
    
    # Run simulations
    final_states = run_multiple_replicates(
        f=func,
        n_cells=N_CELLS,
        n_replicates=N_REPLICATES,
        n_steps=N_STEPS,
        dt=DT,
        noise_strength=NOISE_STRENGTH,
        key=eval_key
    )

    
    # Apply threshold to get patterns (using STE for technical consistency)
    patterns = apply_threshold(final_states)
    
    # Compute soft utility (allows gradients to flow, even though ES doesn't use them)
    # utility, s_pat, s_rep = compute_soft_utility(patterns, bandwidth=SOFT_BANDWIDTH)
    utility, s_pat, s_rep = compute_hard_utility(patterns)
    
    return utility, s_pat, s_rep

@partial(jit, static_argnames=['model', 'N_CELLS', 'N_REPLICATES', 'N_STEPS', 'DT', 'NOISE_STRENGTH'])
def compute_fitness_fixed_boundary(params: Dict, model: RegulatoryNetwork, 
                    N_CELLS, N_REPLICATES, N_STEPS, DT, NOISE_STRENGTH, 
                    eval_key: jax.random.PRNGKey) -> tuple[float, float, float]:
    """
    Evaluate utility (fitness) for given network parameters.
    
    Higher utility = better fitness.
    Uses hard utility, non differentiable, used for ES (Evolution & Selection)
    Args:
        params: Network parameters as a dict.
        model: RegulatoryNetwork instance defining architecture. (static)
        N_CELLS: Number of cells in the simulation. (static)
        N_REPLICATES: Number of replicates to run. (static)
        N_STEPS: Number of time steps in each simulation. (static)
        DT: Time step size for simulations. (static)
        NOISE_STRENGTH: Noise strength in simulations. (static)
        eval_key: JAX PRNGKey for stochastic simulations.
    Returns:
        utility: Computed utility (fitness) score.
        s_pat: Pattern diversity score.
        s_rep: Replicate consistency score.
    """
    # Get regulatory function from params
    func = get_regulatory_function(model, params)
    
    # Run simulations
    final_states = run_multiple_replicates_fixed_boundary(
        f=func,
        n_cells=N_CELLS,
        n_replicates=N_REPLICATES,
        n_steps=N_STEPS,
        dt=DT,
        noise_strength=NOISE_STRENGTH,
        key=eval_key
    )

    
    # Apply threshold to get patterns (using STE for technical consistency)
    patterns = apply_threshold(final_states)
    
    # Compute soft utility (allows gradients to flow, even though ES doesn't use them)
    # utility, s_pat, s_rep = compute_soft_utility(patterns, bandwidth=SOFT_BANDWIDTH)
    utility, s_pat, s_rep = compute_hard_utility(patterns)
    
    return utility, s_pat, s_rep




@partial(jit, static_argnames=['model', 'N_CELLS', 'N_REPLICATES', 'N_STEPS', 'DT', 'NOISE_STRENGTH'])
def evaluate_population(population: jnp.ndarray, model: RegulatoryNetwork, 
                        N_CELLS, N_REPLICATES, N_STEPS, DT, NOISE_STRENGTH, 
                       template: Dict, eval_key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Evaluate fitness for entire population (generation-level operation).
    
    For each individual in the population, unflatten parameters and compute
    fitness by running simulations and evaluating the utility function.
    
    Args:
        population: Array of shape (pop_size, n_params) containing flattened
            parameters for each individual in the generation.
        model: RegulatoryNetwork instance defining the architecture. (static)
        N_CELLS: Number of cells in the simulation. (static)
        N_REPLICATES: Number of replicates to run. (static)
        N_STEPS: Number of time steps in each simulation. (static)
        DT: Time step size for simulations. (static)
        NOISE_STRENGTH: Noise strength in simulations. (static)
        template: Parameter dict template for unflattening.
        eval_key: JAX PRNGKey for stochastic fitness evaluations.
    
    Returns:
        Array of shape (pop_size,) containing fitness (utility) scores for
        each individual in the population.
    """
    pop_size, n_params = population.shape
    fitnesses = jnp.zeros((pop_size,))
    keys = random.split(eval_key, pop_size)
    
    def body_fn(i, fit_array):
        """Loop body: evaluate fitness for individual i."""
        individual = population[i]
        key_i = keys[i]
        params = unflatten_params(individual, template)
        fitness, _, _ = compute_fitness(params, model, 
                                        N_CELLS, N_REPLICATES, N_STEPS, DT, NOISE_STRENGTH, 
                                        key_i)
        return fit_array.at[i].set(fitness)
    
    # JAX-friendly loop over population
    fitnesses = jax.lax.fori_loop(0, pop_size, body_fn, fitnesses)
    return fitnesses

@partial(jit, static_argnames=['model', 'N_CELLS', 'N_REPLICATES', 'N_STEPS', 'DT', 'NOISE_STRENGTH'])
def evaluate_population_fixed_boundary(population: jnp.ndarray, model: RegulatoryNetwork, 
                        N_CELLS, N_REPLICATES, N_STEPS, DT, NOISE_STRENGTH, 
                       template: Dict, eval_key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Evaluate fitness for entire population (generation-level operation).
    
    For each individual in the population, unflatten parameters and compute
    fitness by running simulations and evaluating the utility function.
    
    Args:
        population: Array of shape (pop_size, n_params) containing flattened
            parameters for each individual in the generation.
        model: RegulatoryNetwork instance defining the architecture. (static)
        N_CELLS: Number of cells in the simulation. (static)
        N_REPLICATES: Number of replicates to run. (static)
        N_STEPS: Number of time steps in each simulation. (static)
        DT: Time step size for simulations. (static)
        NOISE_STRENGTH: Noise strength in simulations. (static)
        template: Parameter dict template for unflattening.
        eval_key: JAX PRNGKey for stochastic fitness evaluations.
    
    Returns:
        Array of shape (pop_size,) containing fitness (utility) scores for
        each individual in the population.
    """
    pop_size, n_params = population.shape
    fitnesses = jnp.zeros((pop_size,))
    keys = random.split(eval_key, pop_size)
    
    def body_fn(i, fit_array):
        """Loop body: evaluate fitness for individual i."""
        individual = population[i]
        key_i = keys[i]
        params = unflatten_params(individual, template)
        fitness, _, _ = compute_fitness_fixed_boundary(params, model, 
                                        N_CELLS, N_REPLICATES, N_STEPS, DT, NOISE_STRENGTH, 
                                        key_i)
        return fit_array.at[i].set(fitness)
    
    # JAX-friendly loop over population
    fitnesses = jax.lax.fori_loop(0, pop_size, body_fn, fitnesses)
    return fitnesses



# Based on the fitnesses, we select the top N parents and mutate the rest

@partial(jit, static_argnames=['N_parents'])
def mutate_population(fitnesses: jnp.ndarray, population: jnp.ndarray, 
                     N_parents: int, mutation_std: float, 
                     mut_key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Generate next generation via elitism + mutation (JIT-compiled).
    
    Creates new generation by: (1) keeping elite parents unchanged, and
    (2) generating children by randomly selecting parents and adding Gaussian noise.
    This implements the (μ+λ) evolution strategy.
    
    Args:
        fitnesses: Array of shape (pop_size,) containing fitness scores.
        population: Array of shape (pop_size, n_params) containing current generation.
        N_parents: Number of elite individuals to keep unchanged.
        mutation_std: Standard deviation of Gaussian mutation noise.
        mut_key: JAX PRNGKey for random parent selection and mutation.
    
    Returns:
        Array of shape (population_size, n_params) containing the new generation:
        first n_parents rows are unchanged parents, remaining rows are mutated children.
    """
    # Sort the populations according to their fitnesses
    sorted_fitnesses_idxs = jnp.argsort(fitnesses)
    sorted_population = population[sorted_fitnesses_idxs[::-1]]

    def body_fn(i, sort_pop_and_key):
        """
        Loops over population, replaces the worst individuals with mutations of the best ones
        """
        sorted_population, key = sort_pop_and_key
        # Split the key for mutation
        key, subkey = random.split(key)
        # Select a random parent from the top N_parents
        parent_idx = random.randint(subkey, (), 0, N_parents)
        parent = sorted_population[parent_idx]
        # Generate mutation noise
        key, subkey = random.split(key)
        noise = random.normal(subkey, parent.shape) * mutation_std
        # Create child by adding noise to parent
        child = parent + noise
        # Replace the i-th individual
        sorted_population = sorted_population.at[i].set(child)
        return sorted_population, key
    
    sort_pop_and_key = jax.lax.fori_loop(N_parents, population.shape[0], body_fn, (sorted_population, mut_key))
    sorted_population, key = sort_pop_and_key

    return sorted_population








