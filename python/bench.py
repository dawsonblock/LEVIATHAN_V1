#!/usr/bin/env python3
"""
bench.py — Benchmark harness for Leviathan phase dynamics engine.

Measures throughput, validates invariants, and prints a clean summary.
"""

import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Setup: import the engine, handle missing module gracefully
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")
try:
    from python.leviathan_h100 import LeviathanObservatory
except ImportError:
    print("ERROR: Cannot import LeviathanObservatory.")
    print("Build first: ./scripts/build_h100.sh")
    sys.exit(1)


def validate_invariants(obs):
    """Check simulation state for NaN/Inf and range violations."""
    theta = obs.get_phase_snapshot()
    errors = []

    if np.any(np.isnan(theta)):
        errors.append(f"NaN in theta ({np.isnan(theta).sum()} nodes)")
    if np.any(np.isinf(theta)):
        errors.append(f"Inf in theta ({np.isinf(theta).sum()} nodes)")
    if np.any(theta < 0) or np.any(theta > 2 * np.pi + 1e-5):
        oob = ((theta < 0) | (theta > 2 * np.pi + 1e-5)).sum()
        errors.append(f"theta out of [0, 2π): {oob} nodes")

    return errors


def run_bench(N, k=8, max_delay=8, steps=200, warmup=50):
    """Run a single benchmark configuration."""
    obs = LeviathanObservatory(N=N, k=k, max_delay=max_delay)

    # Warmup (let gain controller settle)
    for _ in range(warmup):
        obs.step()

    # Timed run
    t0 = time.perf_counter()
    r_vals = []
    for _ in range(steps):
        r = obs.step()
        r_vals.append(r)
    elapsed = time.perf_counter() - t0

    # Validate
    invariant_errors = validate_invariants(obs)

    # Stats
    r_arr = np.array(r_vals)
    num_edges = N * k  # approximate
    edges_per_sec = num_edges * steps / elapsed

    return {
        "N": N,
        "k": k,
        "steps": steps,
        "elapsed_s": elapsed,
        "steps_per_sec": steps / elapsed,
        "ms_per_step": (elapsed / steps) * 1000,
        "edges_per_sec": edges_per_sec,
        "r_mean": float(r_arr.mean()),
        "r_std": float(r_arr.std()),
        "r_min": float(r_arr.min()),
        "r_max": float(r_arr.max()),
        "r_valid": 0.0 <= r_arr.min() and r_arr.max() <= 1.0,
        "invariant_errors": invariant_errors,
        "vram_est_mb": (N * (4 * 5 + k * (4 + 4 + 1)) + N * (max_delay + 1) * 4) / 1e6,
    }


def main():
    configs = [
        {"N": 1_000, "k": 8},
        {"N": 5_000, "k": 8},
        {"N": 10_000, "k": 8},
        {"N": 20_000, "k": 8},
    ]

    print("=" * 70)
    print("  LEVIATHAN — Benchmark")
    print("=" * 70)
    print()

    results = []
    for cfg in configs:
        print(f"  N={cfg['N']:>6,d}  k={cfg['k']}  ...", end="", flush=True)
        try:
            r = run_bench(**cfg)
            results.append(r)
            status = "OK" if not r["invariant_errors"] else "WARN"
            print(
                f"  {r['steps_per_sec']:>7.1f} steps/s  "
                f"{r['ms_per_step']:>6.2f} ms/step  "
                f"r={r['r_mean']:.3f}±{r['r_std']:.3f}  [{status}]"
            )
            for err in r["invariant_errors"]:
                print(f"    ⚠ {err}")
        except Exception as e:
            print(f"  FAILED: {e}")

    print()
    print("-" * 70)
    print(
        f"{'N':>8s}  {'steps/s':>9s}  {'ms/step':>8s}  {'M edges/s':>10s}  "
        f"{'VRAM (MB)':>10s}  {'r range':>14s}  {'status':>6s}"
    )
    print("-" * 70)
    for r in results:
        status = "OK" if (r["r_valid"] and not r["invariant_errors"]) else "FAIL"
        print(
            f"{r['N']:>8,d}  {r['steps_per_sec']:>9.1f}  {r['ms_per_step']:>8.2f}  "
            f"{r['edges_per_sec']/1e6:>10.1f}  {r['vram_est_mb']:>10.1f}  "
            f"[{r['r_min']:.3f},{r['r_max']:.3f}]  {status:>6s}"
        )
    print("-" * 70)


if __name__ == "__main__":
    main()
