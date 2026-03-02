#!/usr/bin/env python3
"""
experiments/phase_sweep.py — Phase transition sweep experiment.

Sweeps coupling gain g, measures steady-state r statistics,
and plots the classic synchronization phase transition curve.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import the engine
sys.path.insert(0, ".")
try:
    from python.leviathan_h100 import LeviathanObservatory
except ImportError:
    print("ERROR: Cannot import LeviathanObservatory.")
    print("Build first: ./scripts/build_h100.sh")
    sys.exit(1)


def run_sweep(N=10000, k=20, max_delay=10):
    print(f"Starting Phase Transition Sweep (N={N}, k={k}, max_delay={max_delay})")
    obs = LeviathanObservatory(N=N, k=k, max_delay=max_delay)

    # Disable adaptive control for manual sweep
    obs.set_gain_controller(False)

    g_vals = np.linspace(0.1, 5.0, 30)
    r_means = []
    r_stds = []

    print(f"\n{'g':>6s} | {'mean r':>8s} | {'std r':>8s}")
    print("-" * 30)

    for g in g_vals:
        obs.set_gain(g)

        # Warmup for each g value
        for _ in range(200):
            obs.step()

        # Measurement
        samples = []
        for _ in range(100):
            r = obs.step()
            samples.append(r)

        mu = np.mean(samples)
        std = np.std(samples)
        r_means.append(mu)
        r_stds.append(std)

        print(f"{g:6.2f} | {mu:8.4f} | {std:8.4f}")

    return g_vals, np.array(r_means), np.array(r_stds)


def plot_results(g_vals, r_means, r_stds):
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        g_vals,
        r_means,
        yerr=r_stds,
        fmt="-o",
        color="#3b82f6",
        ecolor="#93c5fd",
        capsize=3,
        label="Order Parameter r",
    )

    plt.axhline(
        y=0.5,
        color="#10b981",
        linestyle="--",
        alpha=0.5,
        label="Criticality Target (0.5)",
    )

    plt.title(f"Kuramoto Phase Transition Sweep (N={len(r_means)})", fontsize=14)
    plt.xlabel("Coupling Gain (g)", fontsize=12)
    plt.ylabel("Synchrony (r)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/phase_transition.png", dpi=300)
    print("\nPlot saved to results/phase_transition.png")

    # Also save raw data
    data = np.column_stack((g_vals, r_means, r_stds))
    np.savetxt(
        "results/phase_sweep_data.csv",
        data,
        delimiter=",",
        header="g,r_mean,r_std",
        comments="",
    )
    print("Data saved to results/phase_sweep_data.csv")


if __name__ == "__main__":
    try:
        g, r_mu, r_std = run_sweep(N=5000)
        plot_results(g, r_mu, r_std)
    except Exception as e:
        print(f"Sweep failed: {e}")
