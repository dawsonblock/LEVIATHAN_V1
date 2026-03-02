#!/usr/bin/env python3
"""
python/vram_calculator.py — VRAM Budget Calculator for Leviathan.

Analyzes memory scaling of the delayed Kuramoto model on modern GPUs.
Targets: RTX 3080/4090 (10-24GB) and H100 (80GB).
"""

import numpy as np


def calculate_vram(N, k, max_delay):
    nnz = N * k
    n_float = 4
    n_int = 4
    n_uint8 = 1

    # Components
    history = N * (max_delay + 1) * n_float
    topology = (N + 1) * n_int + nnz * (n_int + n_uint8 + n_float)
    node_states = N * n_float * 6  # theta, hat, omega, hat, eps, sum_coupling
    reduction = (N // 256 + 1) * n_float * 2 + n_float

    total_bytes = history + topology + node_states + reduction
    return {
        "History Buffer": history,
        "Topology (CSR)": topology,
        "Node States": node_states,
        "Reduction Buffers": reduction,
        "TOTAL": total_bytes,
    }


def print_table():
    scenarios = [
        (10000, 20, 50, "Small Home Rig"),
        (100000, 20, 50, "Mid-range Research"),
        (1000000, 20, 50, "Large Scale (H100)"),
        (10000000, 20, 10, "Extreme Sparse"),
    ]

    print(f"{'N':>10s} | {'k':>4s} | {'Delay':>6s} | {'VRAM (MB)':>12s} | {'Label'}")
    print("-" * 65)

    for N, k, d, label in scenarios:
        res = calculate_vram(N, k, d)
        vram_mb = res["TOTAL"] / (1024 * 1024)
        print(f"{N:10,d} | {k:4d} | {d:6d} | {vram_mb:12.2f} | {label}")


if __name__ == "__main__":
    print("Leviathan VRAM Scaling Analysis\n")
    print_table()

    print("\nScaling Bottleneck Analysis:")
    print("1. History Buffer scales with O(N * max_delay)")
    print("2. Topology scales with O(N * k)")
    print("3. Node states scale with O(N)")
    print("\nResult: For large N (>1M), max_delay becomes the dominant factor.")
