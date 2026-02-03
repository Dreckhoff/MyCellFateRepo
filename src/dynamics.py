"""
Dynamics module.
"""

from functools import partial
import jax
import jax.numpy as jnp
from jax import random, jit
from typing import Callable, Tuple


@jit
def get_neighbor_average(states: jnp.ndarray) -> jnp.ndarray:
    """Compute average state of immediate neighbors using FIXED boundary conditions. 
    Interior cells average left and right neighbors. Edge cells use only their single neighbor.
    
    Args:
        states: Cell states, shape (N_cells,).
        
    Returns:
        Average neighbor state for each cell, shape (N_cells,).
    """
    n_cells = len(states)
    
    # Initialize output array
    neighbor_avgs = jnp.zeros(n_cells)
    
    if n_cells == 1:
        # Single cell: no neighbors, return its own state
        return states
    
    # Leftmost cell (index 0): only has right neighbor
    neighbor_avgs = neighbor_avgs.at[0].set(states[1])
    
    # Interior cells (indices 1 to N-2): have both neighbors
    if n_cells > 2:
        left_neighbors = states[0:-2]   # states[i-1] for i=1,...,N-2
        right_neighbors = states[2:]    # states[i+1] for i=1,...,N-2
        interior_avgs = (left_neighbors + right_neighbors) / 2.0
        neighbor_avgs = neighbor_avgs.at[1:-1].set(interior_avgs)
    
    # Rightmost cell (index N-1): only has left neighbor
    neighbor_avgs = neighbor_avgs.at[-1].set(states[-2])
    
    return neighbor_avgs


@partial(jit, static_argnames=('f',))
def euler_step(
    states: jnp.ndarray,
    f: Callable[[jnp.ndarray], jnp.ndarray],
    noise_strength: float,
    dt: float,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    """Euler-Maruyama step: s(t+dt) = clip(s(t) + f(s̄)·dt + √dt·σ·ξ, 0, 1).
    Uses function f to compute ds/dt as equal to f(neighbor_average).
    States are clipped to [0, 1] to prevent explosion and maintain biological validity.
    
    Args:
        states: Current cell states, shape (N_cells,). Must be in [0, 1].
        f: Regulatory function mapping neighbor averages to ds/dt.
        noise_strength: Standard deviation σ of Gaussian noise.
        dt: Time step size.
        key: JAX random key for reproducible randomness.
        
    Returns:
        Cell states after one time step, clipped to [0, 1], shape (N_cells,).
    """
    n_cells = len(states)
    
    # Compute neighbor averages
    neighbor_avgs = get_neighbor_average(states)
    
    # Apply regulatory function to get ds/dt for each cell
    # f should be vectorized to handle array input
    dsdt = f(neighbor_avgs)
    
    # Generate Gaussian noise for each cell
    noise = random.normal(key, shape=(n_cells,))
    
    # Euler-Maruyama update
    new_states = states + dsdt * dt + jnp.sqrt(dt) * noise_strength * noise
    
    # Clip states to [0, 1]
    new_states = jnp.clip(new_states, 0.0, 1.0)


    
    return new_states


@partial(jit, static_argnames=('f',))
def euler_step_fixed_boundary(
    states: jnp.ndarray,
    f: Callable[[jnp.ndarray], jnp.ndarray],
    noise_strength: float,
    dt: float,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    """Euler-Maruyama step: s(t+dt) = clip(s(t) + f(s̄)·dt + √dt·σ·ξ, 0, 1).
    Uses function f to compute ds/dt as equal to f(neighbor_average).
    States are clipped to [0, 1] to prevent explosion and maintain biological validity.
    
    Args:
        states: Current cell states, shape (N_cells,). Must be in [0, 1].
        f: Regulatory function mapping neighbor averages to ds/dt.
        noise_strength: Standard deviation σ of Gaussian noise.
        dt: Time step size.
        key: JAX random key for reproducible randomness.
        
    Returns:
        Cell states after one time step, clipped to [0, 1], shape (N_cells,).
    """
    n_cells = len(states)
    
    # Compute neighbor averages
    neighbor_avgs = get_neighbor_average(states)
    
    # Apply regulatory function to get ds/dt for each cell
    # f should be vectorized to handle array input
    dsdt = f(neighbor_avgs)
    
    # Generate Gaussian noise for each cell
    noise = random.normal(key, shape=(n_cells,))
    
    # Euler-Maruyama update
    new_states = states + dsdt * dt + jnp.sqrt(dt) * noise_strength * noise
    
    # Clip states to [0, 1]
    new_states = jnp.clip(new_states, 0.0, 1.0)

    # Set the outer cells to fixed state 0.0
    new_states = new_states.at[0].set(0.0)
    new_states = new_states.at[-1].set(0.0)

    
    
    return new_states

@partial(jit, static_argnames=('f', 'n_steps', 'return_trajectory'))
def simulate(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    initial_states: jnp.ndarray,
    n_steps: int,
    dt: float,
    noise_strength: float,
    key: jax.random.PRNGKey,
    return_trajectory: bool = False
) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate cell fate dynamics for n_steps iterations.
    
    JIT-compiled with static n_steps for efficient training loops.
    States clipped to [0, 1] at every step.
    
    Args:
        f: Regulatory function mapping neighbor averages to ds/dt (static).
        initial_states: Initial cell states, shape (N_cells,).
        n_steps: Number of integration steps (static, known at compile time).
        dt: Time step for Euler integration.
        noise_strength: Standard deviation of Gaussian noise.
        key: Random key for reproducible stochasticity.
        return_trajectory: If True, return full trajectory (static). Default False.
        
    Returns:
        If return_trajectory=False: Final states, shape (N_cells,).
        If return_trajectory=True: Tuple of (trajectory, times) with shapes
        (n_steps+1, N_cells) and (n_steps+1,).
    """
    
    if return_trajectory:
        # Define step function for scan (saves trajectory)
        def step_fn(states, key_i):
            new_states = euler_step(states, f, noise_strength, dt, key_i)
            return new_states, new_states  # (carry, output_to_stack)
        
        # Split keys for all steps
        keys = random.split(key, n_steps)
        
        # Run scan: returns (final_state, all_intermediate_states)
        final_states, trajectory_steps = jax.lax.scan(step_fn, initial_states, keys)
        
        # Prepend initial states to trajectory
        trajectory = jnp.concatenate([initial_states[None, :], trajectory_steps], axis=0)
        times = jnp.arange(n_steps + 1) * dt
        
        return trajectory, times
    
    else:
        # Define step function for scan (discards trajectory)
        def step_fn(states, key_i):
            new_states = euler_step(states, f, noise_strength, dt, key_i)
            return new_states, None  # (carry, no output)
        
        # Split keys for all steps
        keys = random.split(key, n_steps)
        
        # Run scan: returns (final_state, _)
        final_states, _ = jax.lax.scan(step_fn, initial_states, keys)
        
        return final_states


def simulate_fixed_boundary(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    initial_states: jnp.ndarray,
    n_steps: int,
    dt: float,
    noise_strength: float,
    key: jax.random.PRNGKey,
    return_trajectory: bool = False
) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate cell fate dynamics for n_steps iterations.
    
    JIT-compiled with static n_steps for efficient training loops.
    States clipped to [0, 1] at every step.
    
    Args:
        f: Regulatory function mapping neighbor averages to ds/dt (static).
        initial_states: Initial cell states, shape (N_cells,).
        n_steps: Number of integration steps (static, known at compile time).
        dt: Time step for Euler integration.
        noise_strength: Standard deviation of Gaussian noise.
        key: Random key for reproducible stochasticity.
        return_trajectory: If True, return full trajectory (static). Default False.
        
    Returns:
        If return_trajectory=False: Final states, shape (N_cells,).
        If return_trajectory=True: Tuple of (trajectory, times) with shapes
        (n_steps+1, N_cells) and (n_steps+1,).
    """
    
    if return_trajectory:
        # Define step function for scan (saves trajectory)
        def step_fn(states, key_i):
            new_states = euler_step_fixed_boundary(states, f, noise_strength, dt, key_i)
            return new_states, new_states  # (carry, output_to_stack)
        
        # Split keys for all steps
        keys = random.split(key, n_steps)
        
        # Run scan: returns (final_state, all_intermediate_states)
        final_states, trajectory_steps = jax.lax.scan(step_fn, initial_states, keys)
        
        # Prepend initial states to trajectory
        trajectory = jnp.concatenate([initial_states[None, :], trajectory_steps], axis=0)
        times = jnp.arange(n_steps + 1) * dt
        
        return trajectory, times
    
    else:
        # Define step function for scan (discards trajectory)
        def step_fn(states, key_i):
            new_states = euler_step_fixed_boundary(states, f, noise_strength, dt, key_i)
            return new_states, None  # (carry, no output)
        
        # Split keys for all steps
        keys = random.split(key, n_steps)
        
        # Run scan: returns (final_state, _)
        final_states, _ = jax.lax.scan(step_fn, initial_states, keys)
        
        return final_states

@jit
def apply_threshold(states: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
    """Convert continuous states to binary fates using Straight-Through Estimator (STE). This makes 
    everything differentiable.
    
    Forward: Hard threshold (binary {0, 1}). Backward: Identity (∂patterns/∂states ≈ 1).
    Enables gradient-based optimization of discrete patterns.
    
    Args:
        states: Continuous cell states, shape (N_cells,) or (..., N_cells). Values in [0, 1].
        threshold: Threshold for binary assignment. Default 0.5.
        
    Returns:
        Binary cell fates (0 or 1), same shape as states.
    """
    # Hard threshold for forward pass (exact binary patterns)
    patterns = (states >= threshold).astype(jnp.float32)
    
    # Straight-through estimator for backward pass
    return states + jax.lax.stop_gradient(patterns - states)

@partial(jit, static_argnums=(0,))
def generate_initial_conditions(
    n_cells: int,
    key: jax.random.PRNGKey,
    mean: float = 0.5,
    spread: float = 0.1
) -> jnp.ndarray:
    """Generate random initial conditions from uniform distribution [mean-spread, mean+spread].
    
    Args:
        n_cells: Number of cells.
        key: Random key.
        mean: Center of distribution. Default 0.5.
        spread: Half-width of distribution. Default 0.1.
        
    Returns:
        Random initial states, shape (n_cells,).
    """
    return random.uniform(key, shape=(n_cells,), minval=mean - spread, maxval=mean + spread)


@partial(jit, static_argnames=('f', 'n_cells', 'n_replicates', 'n_steps'))
def run_multiple_replicates(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    n_cells: int,
    n_replicates: int,
    n_steps: int,
    dt: float,
    noise_strength: float,
    key: jax.random.PRNGKey,
    ic_mean: float = 0.5,
    ic_spread: float = 0.1
) -> jnp.ndarray:
    """Run multiple independent simulation replicates and return binary patterns.
    
    JIT-compiled with static architecture parameters for efficient training.
    Uses jax.vmap for parallel execution. Patterns thresholded via STE:
    forward = exact binary {0,1}, backward = gradients flow for training.
    
    Args:
        f: Regulatory function (static).
        n_cells: Number of cells in system (static).
        n_replicates: Number of independent replicates (static).
        n_steps: Number of integration steps per replicate (static).
        dt: Time step.
        noise_strength: Noise standard deviation.
        key: Random key (split for each replicate).
        ic_mean: Mean of initial condition distribution. Default 0.5.
        ic_spread: Spread of initial conditions. Default 0.1.
        
    Returns:
        Binary fate patterns with STE gradients, shape (n_replicates, n_cells).
    """
    # Split keys for all replicates at once
    keys = random.split(key, n_replicates)
    
    def run_single_replicate(replicate_key):
        """Run one replicate with its own key."""
        # Split key for IC generation and simulation
        ic_key, sim_key = random.split(replicate_key)
        
        # Generate random initial conditions
        initial_states = generate_initial_conditions(n_cells, ic_key, ic_mean, ic_spread)
        
        # Run simulation
        final_states = simulate(f, initial_states, n_steps, dt, noise_strength, sim_key, return_trajectory=False)
        
        # Threshold to binary patterns using STE
        # Forward: exact binary {0, 1}
        # Backward: gradients flow through
        pattern = apply_threshold(final_states)
        return pattern
    
    # Vectorize over all replicates (runs in parallel on GPU/TPU)
    results = jax.vmap(run_single_replicate)(keys)
    
    return results

def run_multiple_replicates_fixed_boundary(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    n_cells: int,
    n_replicates: int,
    n_steps: int,
    dt: float,
    noise_strength: float,
    key: jax.random.PRNGKey,
    ic_mean: float = 0.5,
    ic_spread: float = 0.1
) -> jnp.ndarray:
    """Run multiple independent simulation replicates and return binary patterns.
    
    JIT-compiled with static architecture parameters for efficient training.
    Uses jax.vmap for parallel execution. Patterns thresholded via STE:
    forward = exact binary {0,1}, backward = gradients flow for training.
    
    Args:
        f: Regulatory function (static).
        n_cells: Number of cells in system (static).
        n_replicates: Number of independent replicates (static).
        n_steps: Number of integration steps per replicate (static).
        dt: Time step.
        noise_strength: Noise standard deviation.
        key: Random key (split for each replicate).
        ic_mean: Mean of initial condition distribution. Default 0.5.
        ic_spread: Spread of initial conditions. Default 0.1.
        
    Returns:
        Binary fate patterns with STE gradients, shape (n_replicates, n_cells).
    """
    # Split keys for all replicates at once
    keys = random.split(key, n_replicates)
    
    def run_single_replicate(replicate_key):
        """Run one replicate with its own key."""
        # Split key for IC generation and simulation
        ic_key, sim_key = random.split(replicate_key)
        
        # Generate random initial conditions
        initial_states = generate_initial_conditions(n_cells, ic_key, ic_mean, ic_spread)
        
        # Run simulation
        final_states = simulate_fixed_boundary(f, initial_states, n_steps, dt, noise_strength, sim_key, return_trajectory=False)
        
        # Threshold to binary patterns using STE
        # Forward: exact binary {0, 1}
        # Backward: gradients flow through
        pattern = apply_threshold(final_states)
        return pattern
    
    # Vectorize over all replicates (runs in parallel on GPU/TPU)
    results = jax.vmap(run_single_replicate)(keys)
    
    return results
