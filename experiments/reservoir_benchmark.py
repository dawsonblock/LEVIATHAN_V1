#!/usr/bin/env python3
"""
experiments/reservoir_benchmark.py — Reservoir Computing Benchmarks.

Tests the Kuramoto reservoir on NARMA-10 and Sine Generation tasks.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import
sys.path.insert(0, ".")
try:
    from python.reservoir import ReservoirComputer
except ImportError:
    from reservoir import ReservoirComputer


def generate_narma10(T=1000):
    """Generate NARMA-10 benchmark sequence"""
    u = np.random.uniform(0, 0.5, T)
    y = np.zeros(T)
    for t in range(10, T):
        y[t] = (
            0.3 * y[t - 1]
            + 0.05 * y[t - 1] * np.sum(y[t - 10 : t])
            + 1.5 * u[t - 10] * u[t - 1]
            + 0.1
        )
    return u, y


def run_narma_benchmark():
    print("\n--- NARMA-10 Benchmark ---")
    u, y = generate_narma10(T=2000)

    train_len = 1500
    rc = ReservoirComputer(N=5000, n_readout=500)

    print("Training...")
    train_mse = rc.train(u[:train_len], y[:train_len])
    print(f"Train MSE: {train_mse:.6f}")

    print("Testing...")
    y_pred = rc.predict(u[train_len:])
    test_mse = np.mean((y[train_len:] - y_pred) ** 2)
    test_nmse = test_mse / np.var(y[train_len:])
    print(f"Test MSE:  {test_mse:.6f}")
    print(f"Test NMSE: {test_nmse:.6f}")

    return y[train_len:], y_pred


def run_sine_benchmark():
    print("\n--- Sine Generation Benchmark ---")
    # Task: predict sin(t+1) given sin(t)
    t = np.linspace(0, 100, 1000)
    u = np.sin(t)
    y = np.sin(t + 0.1)

    train_len = 800
    rc = ReservoirComputer(N=2000, n_readout=200)

    train_mse = rc.train(u[:train_len], y[:train_len])
    print(f"Train MSE: {train_mse:.6f}")

    y_pred = rc.predict(u[train_len:])
    test_mse = np.mean((y[train_len:] - y_pred) ** 2)
    print(f"Test MSE: {test_mse:.6f}")

    return y[train_len:], y_pred


if __name__ == "__main__":
    y_narma, p_narma = run_narma_benchmark()
    y_sine, p_sine = run_sine_benchmark()

    # Plot NARMA results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(y_narma[:200], "k-", alpha=0.5, label="Actual")
    plt.plot(p_narma[:200], "r--", label="Predicted")
    plt.title("NARMA-10 Reservoir Readout")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(y_sine[:200], "k-", alpha=0.5, label="Actual")
    plt.plot(p_sine[:200], "b--", label="Predicted")
    plt.title("Sine Prediction Reservoir Readout")
    plt.legend()

    plt.tight_layout()
    import os

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/reservoir_benchmarks.png")
    print("\nBenchmark plots saved to results/reservoir_benchmarks.png")
