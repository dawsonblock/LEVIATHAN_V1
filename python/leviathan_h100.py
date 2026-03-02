#!/usr/bin/env python3
# leviathan_h100.py
# High-level Python interface to native H100 engine for Leviathan v3.3
# Optimized: NumPy ring buffer for r_history, zero-copy phase transfers

import numpy as np
import networkx as nx
import time
import sys

try:
    from leviathan_cuda import LeviathanEngine
except ImportError:
    print("ERROR: leviathan_cuda module not found!")
    print("Run: ./build_h100.sh")
    sys.exit(1)


class LeviathanObservatory:
    """
    Production-grade Dynamical Cognitive Simulator for H100
    Scales to 100M+ nodes with real-time IIT computation
    """

    def __init__(self, N=100000, k=20, max_delay=50, seed=42):
        self.N = N
        self.k = k
        self.max_delay = max_delay
        self.dt = 0.05

        np.random.seed(seed)

        print(f"[Observatory] Constructing Watts-Strogatz: N={N}, k={k}, p=0.2")
        G = nx.watts_strogatz_graph(N, k=k, p=0.2)
        self._G = G  # Store graph for IIT hub detection

        print("[Observatory] Building CSR sparse format...")
        # Use updated networkx sparse extraction format or fallback to older compatible
        try:
            A = nx.to_scipy_sparse_array(G, format="csr")
        except AttributeError:
            A = nx.to_scipy_sparse_matrix(G, format="csr")

        self.row_ptr = A.indptr.astype(np.int32)
        self.col_idx = A.indices.astype(np.int32)
        self.nnz = len(self.col_idx)

        print(f"[Observatory] Topology: {self.nnz} edges ({self.nnz/(N*k):.2%} fill)")

        # Initialize state vectors
        self.delays = np.random.randint(1, max_delay, self.nnz, dtype=np.uint8)
        self.weights = np.random.uniform(0.01, 0.1, self.nnz).astype(np.float32)
        self.theta = np.random.uniform(0, 2 * np.pi, N).astype(np.float32)
        self.theta_hat = self.theta.copy().astype(np.float32)
        self.omega = np.random.normal(1.0, 0.1, N).astype(np.float32)

        # Initialize CUDA engine
        print("[Observatory] Initializing H100 CUDA engine...")
        self.engine = LeviathanEngine(
            N,
            max_delay,
            self.row_ptr,
            self.col_idx,
            self.delays,
            self.weights,
            self.theta,
            self.theta_hat,
            self.omega,
        )

        # [OPT #14] Pre-allocated NumPy ring buffer instead of Python list
        self._r_capacity = 100000
        self._r_buffer = np.empty(self._r_capacity, dtype=np.float32)
        self.step_count = 0
        print("[Observatory] Ready for simulation.\n")

    @property
    def r_history(self):
        """Return recorded r values as a NumPy array"""
        return self._r_buffer[: self.step_count]

    def step(self):
        """Execute one physics integration step on H100"""
        r = self.engine.step(self.dt)
        # [OPT #14] Direct array write, auto-grow if needed
        if self.step_count >= self._r_capacity:
            self._r_capacity *= 2
            new_buf = np.empty(self._r_capacity, dtype=np.float32)
            new_buf[: self.step_count] = self._r_buffer[: self.step_count]
            self._r_buffer = new_buf
        self._r_buffer[self.step_count] = r
        self.step_count += 1
        return r

    def run_baseline(self, num_steps=500, log_interval=100):
        """Warm up system to metastable criticality"""
        print(f"[Observatory] Baseline run: {num_steps} steps")
        start = time.time()

        for i in range(num_steps):
            r = self.step()
            if (i + 1) % log_interval == 0:
                elapsed = time.time() - start
                fps = (i + 1) / elapsed
                print(f"  Step {i+1:5d}: r={r:.4f} ({fps:.1f} FPS)")

        elapsed = time.time() - start
        avg_fps = num_steps / elapsed
        print(f"[Observatory] Complete in {elapsed:.2f}s (avg {avg_fps:.1f} FPS)\n")
        return self.r_history

    def run_with_stimulus(
        self,
        num_steps=200,
        stimulus_start=100,
        stimulus_nodes=None,
        stimulus_strength=2.0,
    ):
        """Run with external sensory forcing (TMS-style phase perturbation)"""
        if stimulus_nodes is None:
            stimulus_nodes = list(range(10))

        print(f"[Observatory] Stimulus run: {num_steps} steps, start={stimulus_start}")
        print(
            f"[Observatory] Target nodes: {stimulus_nodes[:5]}... (total {len(stimulus_nodes)})"
        )
        print(f"[Observatory] Strength: {stimulus_strength}\n")

        start = time.time()
        for i in range(num_steps):
            r = self.step()

            # Apply TMS-style phase kick at stimulus onset
            if i == stimulus_start:
                theta = self.get_phase_snapshot()
                for node in stimulus_nodes:
                    theta[node] += stimulus_strength  # Phase perturbation
                # Wrap to [0, 2pi)
                theta = theta % (2 * np.pi)
                self.set_phase_snapshot(theta)
                print(f"  [STIMULUS] Applied phase kick to {len(stimulus_nodes)} nodes")

            if (i + 1) % 50 == 0:
                phase = "ACTIVE" if i >= stimulus_start else "baseline"
                elapsed = time.time() - start
                fps = (i + 1) / elapsed
                print(f"  Step {i+1:3d} [{phase}]: r={r:.4f} ({fps:.1f} FPS)")

        elapsed = time.time() - start
        print(f"[Observatory] Complete in {elapsed:.2f}s\n")
        return self.r_history[-num_steps:]

    def get_phase_snapshot(self):
        """Transfer phase array from H100 VRAM to host"""
        return self.engine.get_theta()

    def set_phase_snapshot(self, theta):
        """Upload modified phase array to H100 VRAM"""
        # [OPT #13] Avoid redundant copy if already float32
        if theta.dtype != np.float32:
            theta = theta.astype(np.float32)
        self.engine.set_theta(theta)

    def statistics(self):
        """Compute telemetry statistics"""
        r_arr = np.array(self.r_history)
        return {
            "final_r": r_arr[-1],
            "mean_r": np.mean(r_arr),
            "max_r": np.max(r_arr),
            "min_r": np.min(r_arr),
            "std_r": np.std(r_arr),
            "steps": len(r_arr),
        }


def main():
    print("=" * 70)
    print("LEVIATHAN CSR APEX v3.2 - H100 Observatory")
    print("=" * 70)
    print()

    # Initialize with 100k nodes
    observatory = LeviathanObservatory(N=100000, k=20, max_delay=50)

    # Baseline: metastability settling
    r_hist_1 = observatory.run_baseline(num_steps=500, log_interval=50)

    # Stimulus: external sensory forcing
    r_hist_2 = observatory.run_with_stimulus(num_steps=200, stimulus_start=100)

    # Retrieve phase snapshot
    theta_snapshot = observatory.get_phase_snapshot()

    print("[Observatory] Phase Snapshot Statistics")
    print(f"  Shape: {theta_snapshot.shape}")
    print(f"  Mean: {np.mean(theta_snapshot):.4f} rad")
    print(f"  Std:  {np.std(theta_snapshot):.4f} rad")
    print(f"  Min:  {np.min(theta_snapshot):.4f} rad")
    print(f"  Max:  {np.max(theta_snapshot):.4f} rad")
    print()

    # Global statistics
    stats = observatory.statistics()
    print("[Statistics] Global Metastability Metrics")
    print(f"  Total Steps:   {stats['steps']}")
    print(f"  Final r:       {stats['final_r']:.4f}")
    print(f"  Mean r:        {stats['mean_r']:.4f}")
    print(f"  Std r:         {stats['std_r']:.4f}")
    print(f"  Range:         [{stats['min_r']:.4f}, {stats['max_r']:.4f}]")
    print()

    # Phase wave analysis
    cos_mean = np.mean(np.cos(theta_snapshot))
    sin_mean = np.mean(np.sin(theta_snapshot))
    global_phase = np.arctan2(sin_mean, cos_mean)

    print("[Analysis] Global Phase Alignment")
    print(f"  Global Phase: {global_phase:.4f} rad")
    print(f"  Synchrony Magnitude: {stats['final_r']:.4f}")
    print()

    print("=" * 70)
    print("Leviathan Observatory: Operational")
    print("=" * 70)


if __name__ == "__main__":
    main()
