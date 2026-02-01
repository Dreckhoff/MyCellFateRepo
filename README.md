# Cell Fate Decision Modeling

PhD warmup project (1-2 days): Train neural networks to model Gene Regulatory Networks (GRNs) that optimize cell fate pattern formation.

## Quick Start

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install jax jaxlib flax numpy matplotlib seaborn jupyter

# Run notebooks
jupyter notebook notebooks/
```

## Concept

Cells with state `s ∈ [0,1]` evolve via `ds/dt = f(s̄) + noise`, where `s̄` is neighbor average. After time T, threshold states to binary patterns. Train `f` (a small NN) to maximize **utility** = pattern_entropy - reproducibility_entropy.

**Expected result**: NN learns tanh-like function → lateral inhibition → alternating on-off patterns.

## Structure

- `src/` - Core modules (utility, dynamics, neural network)
- `notebooks/` - Interactive exploration and training
- `figures/` - Generated plots

## Status

- ✅ Utility function (hard + soft differentiable versions)
- ✅ Dynamics simulation with Euler-Maruyama
- ✅ Neural network architecture (Flax)
- 🔲 Evolutionary training loop
- 🔲 Results visualization

## Key Technologies

JAX (auto-diff, JIT), Flax (NN), NumPy, Matplotlib

See `agents.md` for AI assistant guidelines.
