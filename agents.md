# agents.md - AI Assistant Guidelines

**Note**: Keep all documentation CONCISE. This is a 1-2 day warmup project - avoid verbosity.

## Project Overview

Model cell fate decisions using neural network-based Gene Regulatory Networks (GRNs). Cells evolve state `s` via regulator function `f`, receive noise, then threshold to binary patterns. Goal: Train NN to maximize pattern utility (information - unreliability).

## Mathematical Framework

- **State evolution**: `ds/dt = f(s̄) + η` where `s̄` = neighbor average, `η` = Gaussian noise
- **Pattern utility**: `U = S_pat - S_rep`
  - `S_pat`: Pattern entropy (information content)
  - `S_rep`: Reproducibility entropy (pattern variability/unreliability)
- **Soft versions**: Differentiable via soft thresholding + KDE for gradient-based training
- **Hard versions**: Exact binary patterns for final evaluation

## System Architecture

- **Communication**: Cells sense average state of direct neighbors (diffusion-like)
- **Boundaries**: Fixed (edge cells have 1 neighbor only)
- **State space**: [0, 1] with clipping
- **Expected outcome**: NN converges to tanh-like lateral inhibition → on-off patterns

## Code Structure

```
src/
  utility_function.py  - Entropy calculations (hard + soft)
  dynamics.py          - State evolution, Euler-Maruyama, pattern generation
  neural_network.py    - Flax NN, initialization, visualization
notebooks/
  01TheUtilityFunction.ipynb     - Explore U = S_pat - S_rep
  02TheDynamics.ipynb             - Simulate cell state evolution
  03TheRegulatoryNetwork.ipynb    - NN setup and testing
  04TrainingTheNetwork.ipynb      - (TODO) Evolutionary training
  05VisualizationAndAnalysis.ipynb - (TODO) Results analysis
```

## Key Design Decisions

- **JAX/Flax**: Auto-differentiation, JIT compilation, GPU-ready
- **STE (Straight-Through Estimator)**: Enables gradients through binary thresholds
- **KDE smoothing**: Makes pattern probability distribution differentiable
- **vmap**: Parallel execution of replicates

## Working with AI Assistants

- **Context**: Read notebooks 01-03 to understand current implementation
- **Next steps**: Implement evolutionary training (CMA-ES or simple ES)
- **Style**: Match existing code style (docstrings, type hints, @jit decorators)
- **Testing**: Verify utilities work with soft patterns before training
- **Brevity**: This is a warmup project - keep everything minimal and focused
