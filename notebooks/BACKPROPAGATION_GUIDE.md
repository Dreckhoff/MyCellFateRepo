# Backpropagation Training Guide

## Overview

The notebook `05Backpropagation.ipynb` now contains a complete implementation of gradient-based neural network training using JAX's automatic differentiation. This is a more efficient alternative to the evolution strategy used in notebook 04.

## What Was Implemented

### 1. **Differentiable Loss Function**
```python
def compute_loss(params, model, key):
    # params → network → f(s̄) → dynamics → patterns → soft_utility → loss
    ...
    utility, s_pat, s_rep = compute_soft_utility(patterns, bandwidth=SOFT_BANDWIDTH)
    loss = -utility  # Maximize utility = minimize negative utility
    return loss
```

**Key Points:**
- Uses `compute_soft_utility` (differentiable) instead of `compute_hard_utility` (non-differentiable)
- Loss = negative utility (we want to maximize utility)
- Fully JIT-compiled for performance

### 2. **Adam Optimizer**
```python
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)
```

**Why Adam?**
- Combines momentum and adaptive learning rates
- Robust to noisy gradients (from stochastic dynamics)
- Industry standard for neural network training

### 3. **Training Step with Backpropagation**
```python
@jit
def train_step(params, opt_state, model, key):
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(compute_loss)(params, model, key)
    
    # Apply optimizer update
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss
```

**Magic Happening Here:**
- `jax.value_and_grad`: Computes both loss value AND gradients automatically
- Gradients flow backward through: loss → utility → patterns → dynamics → network
- Optimizer uses gradients to update parameters efficiently

### 4. **Training Loop**
```python
for epoch in range(N_EPOCHS):
    key, train_key = random.split(key)
    params, opt_state, loss = train_step(params, opt_state, model, train_key)
    # Track metrics...
```

**Simple but Powerful:**
- Single network (no population needed)
- Uses different random seeds each epoch for stochastic simulations
- Tracks loss and utility over time

## Key Differences from Evolution Strategy

| Aspect | Evolution Strategy (Notebook 04) | Backpropagation (Notebook 05) |
|--------|----------------------------------|-------------------------------|
| **Method** | Population-based, random mutations | Gradient-based optimization |
| **Fitness** | Hard utility (non-differentiable) | Soft utility (differentiable) |
| **Parameters** | Population of networks | Single network |
| **Updates** | Selection + mutation | Gradient descent |
| **Efficiency** | 🐌 Slower, many fitness evaluations | ⚡ Faster, direct gradient path |
| **Sample Efficiency** | Lower (explores randomly) | Higher (guided by gradients) |
| **JIT Compilation** | ✓ Yes | ✓ Yes |

## Understanding the Gradient Flow

The backpropagation chain:

```
1. Random initialization → params₀

2. Forward pass:
   params → f(s̄) → dynamics → final_states → patterns → utility

3. Loss computation:
   loss = -utility

4. Backward pass (automatic!):
   ∂loss/∂params ← backpropagate gradients

5. Parameter update:
   params ← params - learning_rate × ∂loss/∂params
```

**Challenge: Stochasticity**
- The dynamics are stochastic (noise in Euler-Maruyama)
- Gradients are noisy estimates
- Solution: Use many replicates (N_REPLICATES=50) for stable gradient estimates

## Configuration Parameters

```python
SetupDict = {
    "N_CELLS": 7,              # System size
    "N_REPLICATES": 50,        # More replicates = more stable gradients
    "N_STEPS": 1000,           # Simulation length
    "DT": 0.01,                # Time step
    "NOISE_STRENGTH": 0.1,     # Stochastic noise
    
    "HIDDEN_DIMS": (8, 8),     # Neural network architecture
    
    "N_EPOCHS": 500,           # Training epochs
    "LEARNING_RATE": 1e-3,     # Adam learning rate
    "SOFT_BANDWIDTH": 0.1,     # Soft utility bandwidth
}
```

**Tuning Tips:**
- **Learning rate too high**: Training unstable, loss oscillates
- **Learning rate too low**: Training too slow
- **Too few replicates**: Noisy gradients, unstable training
- **Too many replicates**: Slower per-epoch, but more stable

## Visualizations Included

The notebook generates several plots:

1. **Training Curves**: Loss and utility over epochs
2. **Learned Function**: Trained f(s̄) vs target tanh function
3. **Generated Patterns**: Heatmap of cell fate patterns
4. **Performance Comparison**: Random vs trained network

## Expected Results

After training, you should see:

1. **Loss decreases** (utility increases) over epochs
2. **Regulatory function** resembles tanh-like lateral inhibition
3. **Patterns** show alternating on-off structure (e.g., 0101010, 1010101)
4. **High utility**: U ≈ 0.5-0.7 (depends on parameters)

## Troubleshooting

### Problem: Training doesn't improve
**Possible causes:**
- Learning rate too low → increase to 1e-2
- Too few replicates → increase to 100
- Network too small → try (16, 16) hidden dims

### Problem: Loss is NaN
**Possible causes:**
- Learning rate too high → decrease to 1e-4
- Gradient explosion → add gradient clipping

### Problem: Training is slow
**Possible causes:**
- Not JIT compiled → check `@jit` decorators
- Too many replicates → reduce to 30
- Too long simulations → reduce N_STEPS

## Advanced Extensions

Want to go further? Try:

1. **Learning rate scheduling**: Decrease learning rate over time
   ```python
   scheduler = optax.exponential_decay(1e-3, 100, 0.9)
   optimizer = optax.adam(learning_rate=scheduler)
   ```

2. **Gradient clipping**: Prevent gradient explosion
   ```python
   optimizer = optax.chain(
       optax.clip_by_global_norm(1.0),
       optax.adam(1e-3)
   )
   ```

3. **Batch training**: Train on multiple independent simulations
   ```python
   # Use vmap to parallelize over batches
   batch_loss = jax.vmap(compute_loss, in_axes=(None, None, 0))
   ```

4. **Curriculum learning**: Start with less noise, gradually increase
   ```python
   noise = initial_noise * (1 + epoch / N_EPOCHS)
   ```

## Code Structure

```
05Backpropagation.ipynb
├── Imports and setup
├── Parameter configuration
├── Model initialization
├── Loss function definition       ← Core: differentiable!
├── Optimizer setup                ← Adam optimizer
├── Training step definition       ← Gradient computation
├── Training loop                  ← Main training
├── Visualization                  ← Results
└── Save trained model
```

## References

- **JAX Documentation**: https://jax.readthedocs.io/
- **Optax (Optimizers)**: https://optax.readthedocs.io/
- **Adam Optimizer Paper**: Kingma & Ba (2014)
- **Straight-Through Estimator**: Used in `apply_threshold` for gradient flow

## Next Steps

1. **Run the notebook**: Execute all cells and observe training
2. **Experiment**: Try different learning rates, architectures
3. **Compare**: Benchmark against evolution strategy (notebook 04)
4. **Analyze**: Study gradient magnitudes, convergence speed
5. **Extend**: Implement advanced techniques listed above

---

**Happy Training! 🚀**

If you have questions about the implementation, feel free to ask!
