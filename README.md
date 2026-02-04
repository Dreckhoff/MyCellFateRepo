# Cell Fate Decision Modeling

PhD warmup project: Train neural networks to model Gene Regulatory Networks (GRNs) that optimize cell fate pattern formation using information-theoretic objectives.

## Quick Start

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install jax jaxlib flax numpy matplotlib seaborn jupyter

# Run notebooks in order
jupyter notebook notebooks/
```

## Concept

Cells with state `s ∈ [0,1]` evolve via `ds/dt = f(s̄) + noise`, where `s̄` is neighbor average. After time T, threshold states to binary patterns. Train `f` (a small NN) to maximize **utility** = pattern_entropy - reproducibility_entropy.

**Result**: NN successfully learns tanh-like function → lateral inhibition → alternating on-off patterns.

## Documentation

- **[NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md)** - Detailed guide to all notebooks with purpose and key features
- **[FixTheBackprop.md](FixTheBackprop.md)** - Analysis of gradient-based training challenges
- **agents.md** - AI assistant guidelines and project context

## Project Structure

```
src/
  utility_function.py  - Entropy calculations (hard + soft/differentiable versions)
  dynamics.py          - Stochastic dynamics simulation (Euler-Maruyama)
  neural_network.py    - Flax NN architecture and evolutionary training

notebooks/
  01TheUtilityFunction.ipynb           - Utility function implementation
  02TheDynamics.ipynb                  - SDE simulation and integration
  03TheRegulatoryNetwork.ipynb         - Neural network setup
  04TrainingTheNetwork.ipynb           - Evolution strategy training ✅
  05Backpropagation.ipynb              - Gradient-based training attempt ⚠️
  06EvolveAndSelectFixedBoundary.ipynb - Training with fixed boundaries ✅

figures/                - Generated plots and visualizations
```

## Status

- ✅ Utility function (hard + soft differentiable versions)
- ✅ Dynamics simulation with Euler-Maruyama
- ✅ Neural network architecture (Flax)
- ✅ Evolutionary training (successfully discovers lateral inhibition)
- ✅ Fixed boundary variant
- ⚠️ Backpropagation training (investigated but unsuccessful - see FixTheBackprop.md)

## Key Results

- **Evolutionary Strategy (ES)** successfully trains networks to discover lateral inhibition patterns
- **JIT compilation + vmap** provides 15-40× speedup for parallel fitness evaluation
- **Backpropagation** fails due to vanishing gradients through long stochastic simulations
- Trained networks converge to alternating on-off patterns with maximal utility

## Key Technologies

- **JAX** - Auto-differentiation, JIT compilation, GPU-ready
- **Flax** - Neural network library
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Visualization

## Mathematical Framework

- **Stochastic dynamics**: `ds/dt = f(s̄) + η` (Euler-Maruyama integration)
- **Utility function**: `U = S_pattern - S_reproducibility` (information-theoretic objective)
- **Differentiable entropy**: Kernel Density Estimation (KDE) for soft pattern probabilities
- **Evolution Strategy**: (μ+λ)-ES with Gaussian mutations and elitism
