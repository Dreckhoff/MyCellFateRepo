# Backpropagation Quick Reference

## 🎯 Core Concept

**Gradient-based optimization**: Use JAX's automatic differentiation to compute gradients and update network parameters to maximize utility.

## 📝 Essential Code Snippets

### Loss Function (Differentiable!)
```python
@partial(jit, static_argnames=['model'])
def compute_loss(params, model, key):
    f = get_regulatory_function(model, params)
    final_states = run_multiple_replicates(f, ...)
    patterns = apply_threshold(final_states)
    utility, _, _ = compute_soft_utility(patterns, bandwidth=0.1)
    return -utility  # Maximize utility = minimize -utility
```

### Training Step
```python
@partial(jit, static_argnames=['model'])
def train_step(params, opt_state, model, key):
    loss, grads = jax.value_and_grad(compute_loss)(params, model, key)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss
```

### Training Loop
```python
for epoch in range(N_EPOCHS):
    key, train_key = random.split(key)
    params, opt_state, loss = train_step(params, opt_state, model, train_key)
```

## ⚙️ Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `LEARNING_RATE` | 1e-3 | Step size for parameter updates |
| `N_EPOCHS` | 500 | Number of training iterations |
| `N_REPLICATES` | 50 | Number of simulations per gradient estimate |
| `SOFT_BANDWIDTH` | 0.1 | Smoothness of soft utility approximation |

## 🔧 Hyperparameter Tuning

### Learning Rate
- **Too high** (>1e-2): Unstable, oscillating loss
- **Too low** (<1e-5): Very slow convergence
- **Good range**: 1e-4 to 1e-2

### Number of Replicates
- **Too few** (<20): Noisy gradients, unstable training
- **Too many** (>100): Slower training, diminishing returns
- **Good range**: 30-100

## 🐛 Common Issues

### Issue: Loss is NaN
```python
# Solution 1: Lower learning rate
optimizer = optax.adam(learning_rate=1e-4)

# Solution 2: Add gradient clipping
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3)
)
```

### Issue: No improvement
```python
# Check if gradients are flowing
loss, grads = jax.value_and_grad(compute_loss)(params, model, key)
print(jax.tree_map(lambda x: jnp.mean(jnp.abs(x)), grads))
```

### Issue: Training too slow
```python
# Reduce simulation length
SetupDict["N_STEPS"] = 500  # Instead of 1000

# Reduce replicates (less stable but faster)
SetupDict["N_REPLICATES"] = 30  # Instead of 50
```

## 📊 Monitoring Training

### Good Training Signs
✅ Loss steadily decreases  
✅ Utility increases  
✅ Learned function resembles target  
✅ Patterns show structure  

### Warning Signs
⚠️ Loss oscillates wildly → reduce learning rate  
⚠️ Loss stays flat → increase learning rate or check gradients  
⚠️ Loss becomes NaN → gradient explosion, add clipping  

## 🚀 Quick Start

```python
# 1. Setup
model = RegulatoryNetwork(hidden_dims=(8, 8))
params = init_params(model, key, (1,))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# 2. Train
for epoch in range(500):
    key, train_key = random.split(key)
    params, opt_state, loss = train_step(params, opt_state, model, train_key)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# 3. Evaluate
f_trained = get_regulatory_function(model, params)
final_states = run_multiple_replicates(f_trained, ...)
patterns = apply_threshold(final_states)
utility, _, _ = compute_hard_utility(patterns)
print(f"Final utility: {utility:.4f}")
```

## 🎓 Mathematical Background

### Gradient Descent Update
```
θ_{t+1} = θ_t - η ∇_θ L(θ_t)
```
- θ: parameters
- η: learning rate
- L: loss function
- ∇_θ: gradient w.r.t. parameters

### Adam Optimizer
Combines momentum + adaptive learning rates:
```
m_t = β₁ m_{t-1} + (1-β₁) ∇_θ L
v_t = β₂ v_{t-1} + (1-β₂) (∇_θ L)²
θ_t = θ_{t-1} - η m_t / (√v_t + ε)
```

### Chain Rule (Backpropagation)
```
∂L/∂θ = ∂L/∂U × ∂U/∂patterns × ∂patterns/∂states × ∂states/∂f × ∂f/∂θ
```
JAX computes this automatically!

## 📚 Comparison Table

| Feature | Backprop | Evolution Strategy |
|---------|----------|-------------------|
| Speed | ⚡⚡⚡ Fast | 🐌 Slower |
| Sample Efficiency | High | Low |
| Convergence | Direct | Stochastic |
| Differentiable Loss | Required | Not required |
| Population Size | 1 | 10-50 |
| Typical Epochs | 100-1000 | 50-200 |

## 💡 Pro Tips

1. **Warm start**: Initialize with pre-trained weights from evolution strategy
2. **Learning rate decay**: Reduce LR over time for fine-tuning
3. **Early stopping**: Stop when validation utility plateaus
4. **Ensemble**: Train multiple networks with different seeds
5. **Regularization**: Add L2 penalty if overfitting

## 🔗 Related Functions

```python
# From neural_network.py
compute_fitness()      # Uses hard utility (for ES)
get_regulatory_function()
flatten_params()
unflatten_params()

# From utility_function.py
compute_soft_utility()  # Differentiable! Use for backprop
compute_hard_utility()  # Non-differentiable, for evaluation

# From dynamics.py
run_multiple_replicates()
apply_threshold()       # Uses STE for gradient flow
```

## 📖 Further Reading

- JAX Tutorial: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
- Optax Guide: https://optax.readthedocs.io/en/latest/
- Backpropagation: https://en.wikipedia.org/wiki/Backpropagation
- Adam Paper: https://arxiv.org/abs/1412.6980

---

**Remember**: Backpropagation requires differentiable operations throughout the computational graph. That's why we use `soft_utility` instead of `hard_utility`!
