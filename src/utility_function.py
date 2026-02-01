"""
The utility function module.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple

@jit
def compute_entropy(probabilities: jnp.ndarray, epsilon: float = 1e-10) -> float:
    """Compute Shannon entropy H = -Σ p(x) log₂ p(x).
    
    Args:
        probabilities: Probability distribution (should sum to 1).
        epsilon: Small value to prevent log(0). Default 1e-10.
        
    Returns:
        Shannon entropy in bits.
    """
    # Clip probabilities to [epsilon, 1] to prevent log(0)
    probs = jnp.clip(probabilities, epsilon, 1.0)
    
    # Shannon entropy: H = -Σ p·log₂(p)
    entropy = -jnp.sum(probabilities * jnp.log2(probs))
    
    return entropy

@jit
def compute_pattern_entropy(patterns: jnp.ndarray, epsilon: float = 1e-10) -> float:
    """Compute pattern entropy from a set of patterns. It is the Shannon entropy of the
    pooled probablility distribution of cell fates.
    
    Args:
        patterns: Array of shape (num_patterns, pattern_length).
        epsilon: Small value to prevent log(0). Default 1e-10.
    Returns:
        Pattern entropy S_pat in bits.
    """
    # Flatten to pool all cell fates: (n_replicates × n_cells,)
    all_fates = patterns.flatten()
    
    # Count occurrences of each fate
    # For binary {0, 1}, we can use mean to get p₁
    p_fate_1 = jnp.mean(all_fates)  # Fraction with fate 1
    p_fate_0 = 1.0 - p_fate_1        # Fraction with fate 0
    
    # Create probability distribution
    probabilities = jnp.array([p_fate_0, p_fate_1])
    
    # Compute Shannon entropy
    return compute_entropy(probabilities, epsilon=epsilon)

@jit
def patterns_to_integers(patterns: jnp.ndarray) -> jnp.ndarray:
    """Convert binary patterns to integer identifiers via binary encoding: Σᵢ zᵢ·2^i. Thereby, every pattern
    has a unique integer representation.
    
    Enables efficient pattern counting and frequency analysis for JIT compilation.
    
    Args:
        patterns: Binary patterns, shape (n_replicates, n_cells).
        
    Returns:
        Integer representation of each pattern, shape (n_replicates,).
    """
    n_replicates, n_cells = patterns.shape  # patterns: (n_replicates, n_cells)
    
    # Create powers of 2: [2^0, 2^1, ..., 2^(n_cells-1)]
    powers = 2 ** jnp.arange(n_cells)  # (n_cells,)
    
    # Convert each pattern to integer: dot product with powers of 2
    # (n_replicates, n_cells) @ (n_cells,) -> (n_replicates,)
    integers = jnp.dot(patterns, powers)  # (n_replicates,)
    
    return integers.astype(jnp.int32)  # (n_replicates,)


# NOT DIFFERENTIABLE!
@jit 
def compute_pattern_prob_distribution(patterns: jnp.ndarray, 
                                     epsilon: float = 1e-10) -> jnp.ndarray:
    """Compute probability distribution over unique patterns.
    
    Uses jnp.unique with size=2^n_cells for JIT compatibility (output is padded).
    This function cannot be used for the calculation of gradients because of the jnp.uniquq call.
    
    Args:
        patterns: Binary cell fate patterns, shape (n_replicates, n_cells).
        epsilon: Small value to prevent division by zero. Default 1e-10.

    Returns:
        Probability distribution, shape (2^n_cells,). Unused entries are zero.
    """
    n_replicates, n_cells = patterns.shape
    
    # Convert patterns to unique integers
    pattern_ids = patterns_to_integers(patterns)
    
    # Calculate maximum possible unique patterns for this system
    # For binary patterns with n_cells, there are 2^n_cells possibilities
    max_possible_patterns = 2 ** n_cells
    
    # Count unique patterns and their frequencies
    # Specify size parameter to make JIT-compatible
    # Output arrays will be padded with zeros to this size
    unique_patterns, counts = jnp.unique(
        pattern_ids, 
        return_counts=True,
        size=max_possible_patterns,
        fill_value=0  # Pad unused entries with 0
    )
    
    # Convert counts to probabilities
    # Zero counts (padding) will become zero probabilities
    # The epsilon parameter in compute_entropy will handle these zeros
    probabilities = counts / n_replicates  # (max_possible_patterns,)

    return probabilities  # (max_possible_patterns,)


@jit
def compute_hard_reproducibility_entropy(patterns: jnp.ndarray, epsilon: float = 1e-10) -> float:
    """Compute reproducibility entropy H_repro/N = -Σ p(z)·log₂(p(z)) / N.
    
    Measures pattern reliability across replicates (normalized by cell count). Is using the non-differentiable
    compute_pattern_prob_distribution function. Therefore, cannot be used for training with gradients.
    
    Args:
        patterns: Binary cell fate patterns, shape (n_replicates, n_cells).
        epsilon: Small value to prevent log(0). Default 1e-10.
        
    Returns:
        Normalized reproducibility entropy [0, log₂(n_replicates)/N].
        0 = perfect reproducibility, high = random patterns.
    """
    n_replicates, n_cells = patterns.shape
    # Get probability distribution of patterns
    probabilities = compute_pattern_prob_distribution(patterns, epsilon=epsilon)
    
    # Compute Shannon entropy
    # The epsilon in compute_entropy will handle the padded zeros (they contribute 0 to entropy)
    entropy = compute_entropy(probabilities, epsilon=epsilon)
    # Divide entropy by cell number
    S_rep = entropy / n_cells

    return S_rep

@jit
def compute_hard_utility(patterns: jnp.ndarray, epsilon: float = 1e-10) -> tuple[float, float, float]:
    """Compute hard utility U_hard = S_pat - S_rep.
    
    Measures the effective number of reliable patterns. Not differentiable!
    
    Args:
        patterns: Binary cell fate patterns, shape (n_replicates, n_cells).
        epsilon: Small value to prevent log(0). Default 1e-10.
        
    Returns:
        Hard utility in bits.
    """
    S_pat = compute_pattern_entropy(patterns, epsilon=epsilon)
    S_rep = compute_hard_reproducibility_entropy(patterns, epsilon=epsilon)
    
    U_hard = S_pat - S_rep
    
    return U_hard, S_pat, S_rep


@jit
def compute_hard_loss(patterns: jnp.ndarray, epsilon: float = 1e-10) -> float:
    """Compute hard loss L_hard = -U_hard = S_rep - S_pat.
    
    Measures the negative effective number of reliable patterns. Not differentiable!
    
    Args:
        patterns: Binary cell fate patterns, shape (n_replicates, n_cells).
        epsilon: Small value to prevent log(0). Default 1e-10.
    """
    U_hard, S_pat, S_rep = compute_hard_utility(patterns, epsilon=epsilon)
    L_hard = -U_hard
    return L_hard



"""
No we get to the problem of defining a differentiable utility function.
The first point where differentiability breaks is when we assign binary fates to patterns."""

@jit
def apply_soft_threshold(states: jnp.ndarray, threshold: float = 0.5, temperature: float=20.0) -> jnp.ndarray:
    """
    Applies a differentiable sigmoid function to the states to obtain soft cell fates.
    Args:
        states: Continuous cell states, shape (N_cells,) or (..., N_cells). Values in [0, 1].
        threshold: Threshold for sigmoid midpoint. Default 0.5.
        temperature: Steepness of the sigmoid. Higher values make it steeper. Default 10.0.
    Returns:
        Soft cell fates in [0, 1], same shape as states.
    """
    # Sigmoid function for soft thresholding
    return 1.0 / (1.0 + jnp.exp(-temperature * (states - threshold)))

"""
Next smooth our discrete pattern probability distribution using the Kernel density estimation.
It essentially calculates the overlap of patterns with a Gaussian kernel.
"""

@jit
def compute_soft_pattern_prob_distribution(patterns: jnp.ndarray, 
                                          bandwidth: float = 0.1) -> jnp.ndarray:
    """
    Compute a smoothed probability distribution over patterns using Kernel Density Estimation (KDE).
    
    Args:
        patterns: Soft cell fate patterns, shape (n_replicates, n_cells).
        bandwidth: The bandwidth (standard deviation) of the Gaussian kernel. Default 0.1.
        
    Returns:
        Smoothed probability distribution, shape (n_replicates,).
    """
    n_replicates, n_cells = patterns.shape

    # Compute vector distance between different (soft) patterns
    vec_dist = patterns[:, jnp.newaxis, :] - patterns[jnp.newaxis, :, :]  # (n_replicates, n_replicates, n_cells)
    vec_dist_sq = vec_dist ** 2  # Squared differences (n_replicates, n_replicates, n_cells)
    vec_dist_sq_sum = jnp.sum(vec_dist_sq, axis=-1)  # Sum over cells (n_replicates, n_replicates)

    # Apply Gaussian kernel
    pattern_counts = jnp.exp(-vec_dist_sq_sum / (2 * bandwidth ** 2))  # (n_replicates, n_replicates)

    # Compute probabilities by summing over rows, subtracting 1 and normalizing by the number of replicates
    prob_distribution = jnp.sum(pattern_counts, axis=1) / (n_replicates)
    
    
    return prob_distribution  # (n_replicates,)

@jit
def compute_soft_utility(patterns: jnp.ndarray, bandwidth: float = 0.1, epsilon: float = 1e-10) -> tuple[float, float, float]:
    """Compute soft utility U_soft = S_pat - S_rep using differentiable soft patterns.
    
    Args:
        patterns: Soft cell fate patterns, shape (n_replicates, n_cells).
        bandwidth: Bandwidth for KDE smoothing. Default 0.1.
        epsilon: Small value to prevent log(0). Default 1e-10.
    Returns:
        Soft utility in bits.
    """
    n_replicates, n_cells = patterns.shape

    # Compute the pattern entropy
    S_pat = compute_pattern_entropy(patterns, epsilon=epsilon)

    # Compute smoothed pattern probability distribution
    probabilities = compute_soft_pattern_prob_distribution(patterns, bandwidth=bandwidth)
    
    # Compute Shannon entropy for pattern entropy
    # H_rep = compute_entropy(probabilities, epsilon=epsilon)
    H_rep = - (1 / n_replicates) * jnp.sum(jnp.log2(jnp.clip(probabilities, epsilon, 1.0))) 

    # Normalise by the number of cells to get reproducibility entropy
    S_rep = H_rep / n_cells
    
    U_soft = S_pat - S_rep
    
    return U_soft, S_pat, S_rep

@jit
def compute_soft_loss(patterns: jnp.ndarray, bandwidth: float = 0.1, epsilon: float = 1e-10) -> float:
    """Compute soft loss L_soft = -U_soft = S_rep - S_pat.
    
    Measures the negative effective number of reliable patterns using differentiable soft patterns.
    
    Args:
        patterns: Soft cell fate patterns, shape (n_replicates, n_cells).
        bandwidth: Bandwidth for KDE smoothing. Default 0.1.
        epsilon: Small value to prevent log(0). Default 1e-10.
        
    Returns:
        Soft loss value.
    """
    U_soft, S_pat, S_rep = compute_soft_utility(patterns, bandwidth=bandwidth, epsilon=epsilon)
    L_soft = -U_soft
    return L_soft