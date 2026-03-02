# Model Specification

## Delayed Kuramoto Network with Adaptive Coupling

### State Variables

| Symbol | Domain | Description |
|:---|:---|:---|
| θ_i(t) | [0, 2π) | Phase of oscillator i |
| ω_i | ℝ | Natural frequency of oscillator i |
| w_ij(t) | [w_min, w_max] | Synaptic weight from j to i |
| τ_ij | {0, ..., max_delay} | Synaptic delay from j to i (integer steps) |
| ε_i(t) | ℝ | Prediction error at node i |
| θ̂_i(t) | ℝ | Predicted phase (internal model) |
| g(t) | [g_min, g_max] | Global coupling gain |
| r(t) | [0, 1] | Order parameter (global synchrony) |

### Dynamics

**Phase evolution:**

```
dθ_i/dt = ω_i + g · Σ_j w_ij · sin(θ_j(t - τ_ij) - θ_i) - α · ε_i
```

where the sum is over all incoming edges j → i in the CSR graph.

**Prediction error:**

```
ε_i = sin(θ_i - θ̂_i)
```

**Synaptic plasticity (per edge):**

```
dw_ij/dt = η · cos(Δθ_ij) · (1 - γ · |ε_i|) - λ · w_ij
```

- First term: Hebbian learning gated by prediction precision
- Second term: Weight decay (prevents runaway growth)
- Weights clamped to [w_min, w_max] after each update

**Metastability controller:**

```
dg/dt = β · (r_0 - r) - μ · (g - g_rest)
```

Homeostatic gain modulation that targets r ≈ r_0 (typically 0.5).

**Order parameter:**

```
r = |1/N · Σ_j exp(i · θ_j)|
```

r = 0 → fully desynchronized, r = 1 → fully synchronized.

### Topology

Graph is stored in CSR (Compressed Sparse Row) format:

- `row_ptr[N+1]`: Start index of each node's incoming edge list
- `col_idx[num_edges]`: Source node for each edge
- `delays[num_edges]`: Delay for each edge (uint8)
- `weights[num_edges]`: Synaptic weight for each edge (float32)

Phase history is stored in a circular buffer `theta_buffer[N × buffer_size]` for delay lookups.

### Default Parameters

| Parameter | Value | Description |
|:---|:---:|:---|
| dt | 0.05 | Integration timestep |
| η (eta) | 0.01 | Learning rate |
| λ (lambda) | 0.001 | Weight decay |
| γ (gamma) | 0.05 | Prediction error gating strength |
| α (alpha_eps) | 0.1 | Prediction error feedback |
| κ_w (kappa_w) | 0.01 | Frequency adaptation rate |
| w_min | 0.0 | Minimum weight |
| w_max | 2.0 | Maximum weight |
| g_0 | 1.5 | Initial coupling gain |
| r_0 | 0.5 | Target order parameter |
| β (beta) | 0.01 | Gain adaptation rate |
| g_min, g_max | 0.1, 5.0 | Gain bounds |

### Numerical Considerations

- Float32 arithmetic throughout
- `--use_fast_math` is **disabled** for numerical integrity
- NaN guards on phase and weight updates
- Results are stable run-to-run on the same GPU, but not guaranteed bitwise across architectures
