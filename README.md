<div align="center">

# LEVIATHAN

**GPU-Accelerated Delayed Kuramoto Network with Adaptive Coupling**

[![CUDA](https://img.shields.io/badge/CUDA-12.0+-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

</div>

## What This Is

A CUDA-accelerated **phase oscillator network** with delayed coupling, prediction-gated Hebbian plasticity, and homeostatic gain control. Built on a node-centric CSR kernel that eliminates atomic contention.

Suitable for:

- Synchronization phase transition studies
- Adaptive network plasticity experiments
- Reservoir computing substrates
- Temporal pattern locking research
- Large-scale phase coherence analysis

See [`docs/MODEL.md`](docs/MODEL.md) for the full mathematical specification.

## What This Is Not

- Not a neural simulator (no spiking, no ion channels)
- Not an IIT implementation (the Φ module is experimental, not IIT 4.0 compliant)
- Not deterministic across GPU architectures (float32 arithmetic, run-to-run stable on same hardware)
- Tested up to ~50k nodes on consumer GPUs; larger networks need memory layout redesign

## Architecture

```
src/
├── cuda/
│   ├── leviathan_kernel.cu         # Core engine: CSR physics, plasticity, r reduction
│   ├── leviathan_phi_kernel.cu     # Experimental GPU Φ accumulation
│   └── leviathan_multigpu.cu       # NCCL multi-GPU (optional, OFF by default)
└── bindings/
    ├── leviathan_pybind11.cpp      # Core engine → Python
    └── leviathan_phi_pybind.cpp    # Φ solver → Python

python/
├── leviathan_h100.py               # High-level Observatory API
├── dashboard.py                    # Real-time Plotly/Dash dashboard
├── leviathan_iit_integration.py    # Φ backend selector (GPU or PyPhi fallback)
├── partition.py                    # METIS graph partitioning utility
└── bench.py                        # Benchmark + invariant validation

docs/
├── MODEL.md                        # Mathematical model specification
└── DEPLOYMENT_SUMMARY.md           # Deployment reference
```

## Quick Start

### Prerequisites

```bash
# Ubuntu 22.04 + NVIDIA GPU (SM86+ recommended)
sudo apt-get install -y build-essential cmake cuda-toolkit-12-3 python3-dev python3-pip
pip3 install numpy scipy networkx pybind11
```

### Build

```bash
chmod +x scripts/build_h100.sh
./scripts/build_h100.sh
```

Verify:

```bash
python3 -c "from leviathan_cuda import LeviathanEngine; print('core: OK')"
python3 -c "from leviathan_phi import GPUPhiWorker; print('phi: OK')"
```

### Run

```bash
# Simulation
python3 python/leviathan_h100.py

# Benchmark
python3 python/bench.py

# Dashboard (requires: pip install dash plotly)
python3 python/dashboard.py
```

## API

```python
from python.leviathan_h100 import LeviathanObservatory
import numpy as np

# Build a 10k-node Watts-Strogatz network
obs = LeviathanObservatory(N=10_000, k=20, max_delay=50)

# Let system settle
obs.run_baseline(num_steps=500)

# Step and observe
r = obs.step()            # Order parameter (synchrony)
theta = obs.get_phase_snapshot()  # [N] float32 array (device → host copy)

print(f"r = {r:.4f}")     # Target: ~0.5 (edge of chaos)
```

## Engine Design

**Node-centric CSR** — one thread per node, each iterates its incoming edge row. No atomics needed for coupling accumulation or weight updates.

**Prediction-gated plasticity** — synaptic learning is modulated by each node's prediction error. Nodes that can predict their dynamics wire normally; surprised nodes suppress plasticity (Free Energy Principle–inspired).

**Homeostatic gain** — a controller adjusts global coupling strength to maintain metastability (r ≈ 0.5), preventing the network from collapsing to full synchrony or chaos.

**GIL released** — Python's GIL is released during `step()` and `accumulate()`, so GPU work doesn't block Python threads.

## Technical Notes

| Aspect | Status |
|:---|:---|
| Atomics | None in accumulation or weight update (CSR design) |
| Precision | float32, `--use_fast_math` disabled |
| Reproducibility | Stable run-to-run on same GPU; not bitwise across architectures |
| NaN safety | `isfinite()` guards on phase and weight updates |
| GIL | Released during GPU calls |
| CUDA arch | SM86 (3080 Ti) + SM90 (H100) |
| Φ module | Experimental — not IIT 4.0 compliant |

## Parameters

| Parameter | Default | Description |
|:---|:---:|:---|
| N | 10,000 | Network size |
| k | 20 | Average degree |
| max_delay | 50 | Maximum synaptic delay (steps) |
| dt | 0.05 | Integration timestep |
| η | 0.01 | Hebbian learning rate |
| λ | 0.001 | Weight decay |
| γ | 0.05 | Prediction error gating |

Full parameter list in [`docs/MODEL.md`](docs/MODEL.md).

## References

- Strogatz, S. (2003). *Sync*. Hyperion.
- Friston, K. (2010). "The Free-Energy Principle." *Nature Reviews Neuroscience*, 13(2), 126–136.
- Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence*. Springer.

## Citation

```bibtex
@software{leviathan_2026,
  title  = {Leviathan: GPU-Accelerated Delayed Kuramoto Network},
  author = {Dawson Block},
  year   = {2026},
  url    = {https://github.com/dawsonblock/LEVIATHAN_V1}
}
```

## License

MIT
