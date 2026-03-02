#!/usr/bin/env python3
# leviathan_h100.py
# High-level Python interface to the CUDA phase dynamics engine.
# Builds a Watts-Strogatz graph, converts to CSR, and drives the GPU kernel.

import numpy as np
import networkx as nx
import time
import sys

try:
    from leviathan_cuda import LeviathanEngine
except ImportError:
    print("ERROR: leviathan_cuda module not found!")
    print("Run: ./scripts/build_h100.sh")
    sys.exit(1)


class LeviathanObservatory:
    """
    GPU-accelerated delayed Kuramoto network with adaptive coupling.
    Wraps the CUDA engine with graph construction and telemetry.
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

    def set_gain(self, g):
        self.engine.set_gain(g)

    def get_gain(self):
        return self.engine.get_gain()

    def set_gain_controller(self, enabled):
        self.engine.set_gain_controller(enabled)

    def get_vram_usage(self):
        return self.engine.get_vram_usage()

    def reset_weights(self, weights=None):
        if weights is None:
            weights = self.weights
        self.engine.reset_weights(weights)

    def get_vram_report(self):
        """Return a formatted VRAM usage report"""
        bytes_used = self.get_vram_usage()
        mb_used = bytes_used / (1024 * 1024)

        # Theoretical breakdown
        n_float = 4
        n_int = 4
        n_uint8 = 1

        breakdown = {
            "History Buffer": self.N * (self.max_delay + 1) * n_float,
            "Topology (CSR)": (self.N + 1) * n_int
            + self.nnz * (n_int + n_uint8 + n_float),
            "Node States": self.N
            * n_float
            * 6,  # theta, theta_hat, omega, omega_hat, eps, sum_coupling
            "Reduction Buffers": (self.N // 256 + 1) * n_float * 2 + n_float,
        }

        report = f"VRAM Usage Report (N={self.N}, E={self.nnz})\n"
        report += f"{'-'*40}\n"
        for k, v in breakdown.items():
            report += f"{k:20s} : {v/(1024*1024):8.2f} MB\n"
        report += f"{'-'*40}\n"
        report += f"{'TOTAL (Measured)':20s} : {mb_used:8.2f} MB\n"
        return report

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
    print("LEVIATHAN — Phase Dynamics Observatory")
    print("=" * 70)
    print()

    # Initialize with 100k nodes
    observatory = LeviathanObservatory(N=100000, k=20, max_delay=50)

    # Baseline: metastability settling
    observatory.run_baseline(num_steps=500, log_interval=50)

    # Stimulus: external sensory forcing
    observatory.run_with_stimulus(num_steps=200, stimulus_start=100)

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
    import argparse

    parser = argparse.ArgumentParser(
        description="Leviathan — Phase Dynamics Observatory"
    )
    parser.add_argument(
        "--memory-report", action="store_true", help="Print VRAM usage report and exit"
    )
    parser.add_argument("--N", type=int, default=100000, help="Number of nodes")
    args = parser.parse_args()

    if args.memory_report:
        obs = LeviathanObservatory(N=args.N, k=20, max_delay=50)
        print(obs.get_vram_report())
        sys.exit(0)

    main()
