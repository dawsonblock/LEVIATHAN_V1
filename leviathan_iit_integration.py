# leviathan_iit_integration.py
# Connects native H100 physics engine to IIT computation
# Supports two backends: GPU-native Φ solver (preferred) or PyPhi (fallback)

import numpy as np
from queue import Queue
import time
from leviathan_h100 import LeviathanObservatory

# Backend 1: GPU-native Φ solver (preferred — 100x faster)
try:
    from leviathan_phi import GPUPhiWorker

    GPU_PHI_AVAILABLE = True
    print("[IIT] GPU Φ backend available (CUDA-accelerated)")
except ImportError:
    GPU_PHI_AVAILABLE = False

# Backend 2: PyPhi (CPU fallback)
try:
    import pyphi
    from pyphi import Network, compute

    PYPHI_AVAILABLE = True
except ImportError:
    PYPHI_AVAILABLE = False

if not GPU_PHI_AVAILABLE and not PYPHI_AVAILABLE:
    print("WARNING: No IIT backend available.")
    print("  GPU:   Build leviathan_phi (./build_h100.sh)")
    print("  CPU:   pip install pyphi")


class LeviathanIITWorker:
    """
    Async IIT computation worker thread
    Computes empirical Φ on rich-club hubs while physics runs at 1000+ FPS
    """

    def __init__(self, hub_indices, bin_resolution=2, tpm_window=5000):
        """
        Args:
            hub_indices: List of high-degree node indices (typically top 6-8 hubs)
            bin_resolution: Binning for phase (2=binary, 4=quaternary)
            tpm_window: Number of transitions to accumulate before computing Φ
        """
        if not PYPHI_AVAILABLE:
            raise RuntimeError("PyPhi not installed. Cannot compute IIT.")

        self.hub_indices = np.array(hub_indices, dtype=np.int32)
        self.num_hubs = len(hub_indices)
        self.bin_resolution = bin_resolution
        self.tpm_window = tpm_window

        # For binary (2 bins):
        #   - num_states = 2^num_hubs
        #   - Example: 6 hubs = 64 possible states
        self.num_states = bin_resolution**self.num_hubs

        # Transition counter: [source_state][target_state]
        self.transition_count = np.zeros(
            (self.num_states, self.num_states), dtype=np.int64
        )
        self.transition_counter = 0

        # State history (for binarization)
        self.state_history = Queue(maxsize=tpm_window + 1)

        # Result storage
        self.phi_result = None
        self.phi_timestamp = None
        self.is_computing = False

        print(f"[IITWorker] Initialized for {self.num_hubs} hub nodes")
        print(f"[IITWorker] Binary resolution: {self.bin_resolution} bins")
        print(f"[IITWorker] State space: {self.num_states} possible states")
        print(f"[IITWorker] TPM window: {self.tpm_window} transitions\n")

    def binarize_phases(self, theta_snapshot):
        """
        Convert continuous phases to discrete state index
        theta_snapshot: Full phase array [N]
        Returns: Integer state index (0 to num_states-1)
        """
        # Extract hub phases only
        hub_phases = theta_snapshot[self.hub_indices]

        # Convert to binary (phase > pi → 1, else 0)
        if self.bin_resolution == 2:
            bits = (hub_phases > np.pi).astype(int)
            # Convert binary array to integer: [b0, b1, b2, b3, b4, b5] → int
            state = 0
            for i, bit in enumerate(bits):
                state += bit * (2**i)
            return state

        elif self.bin_resolution == 4:
            # Quaternary: [0, pi/2) → 0, [pi/2, pi) → 1, [pi, 3pi/2) → 2, [3pi/2, 2pi) → 3
            quarts = np.floor(hub_phases / (np.pi / 2)).astype(int)
            quarts[quarts >= 4] = 3
            state = 0
            for i, q in enumerate(quarts):
                state += q * (4**i)
            return state

        else:
            raise ValueError(f"Unsupported bin_resolution: {self.bin_resolution}")

    def accumulate_transition(self, current_state, next_state):
        """Record a state transition for TPM estimation"""
        self.transition_count[current_state, next_state] += 1
        self.transition_counter += 1

    def build_tpm(self):
        """
        Build empirical Transition Probability Matrix from observed transitions.
        Returns: PyPhi-compatible state-by-node TPM of shape (num_states, num_hubs).

        Each entry TPM[s, n] = P(node n is ON at t+1 | system is in state s at t)
        """
        # First build the full state-by-state TPM
        full_tpm = np.zeros_like(self.transition_count, dtype=float)

        for source_state in range(self.num_states):
            total_transitions = np.sum(self.transition_count[source_state, :])
            if total_transitions > 0:
                full_tpm[source_state, :] = (
                    self.transition_count[source_state, :] / total_transitions
                )
            else:
                # Uniform default if state never observed as source
                full_tpm[source_state, :] = 1.0 / self.num_states

        # Convert to state-by-node format: P(node n ON | state s)
        # For each target state, check if node n is ON (bit n set)
        sbn_tpm = np.zeros((self.num_states, self.num_hubs), dtype=float)
        for s in range(self.num_states):
            for n in range(self.num_hubs):
                # Sum probabilities of all target states where node n is ON
                for t in range(self.num_states):
                    if self.bin_resolution == 2:
                        node_on = (t >> n) & 1
                    else:
                        node_on = int(
                            (t // (self.bin_resolution**n)) % self.bin_resolution > 0
                        )
                    sbn_tpm[s, n] += full_tpm[s, t] * node_on

        return sbn_tpm

    def compute_phi(self, theta_snapshot, prev_state=None):
        """
        Empirical Φ computation (potentially expensive)
        Call this periodically (e.g., every 100 steps), not every step
        """
        if self.is_computing:
            return None  # Skip if already computing

        current_state = self.binarize_phases(theta_snapshot)

        # Need state history for transition context
        if prev_state is not None:
            self.accumulate_transition(prev_state, current_state)

        # Only compute Φ once we have sufficient transitions
        if self.transition_counter < self.tpm_window:
            return None

        # Trigger async computation
        self.is_computing = True

        try:
            # Build empirical state-by-node TPM
            tpm = self.build_tpm()

            # Create PyPhi network
            node_labels = tuple(f"N{i}" for i in range(self.num_hubs))

            net = Network(
                tpm=tpm,
                node_labels=node_labels,
            )

            # Compute Φ for the whole system
            phi_result = compute.phi(net)

            self.phi_result = phi_result
            self.phi_timestamp = time.time()

            # Reset transition counter for next window
            self.transition_count.fill(0)
            self.transition_counter = 0

            return phi_result

        except Exception as e:
            print(f"[IITWorker] Error computing Φ: {e}")
            return None

        finally:
            self.is_computing = False

    def get_latest_phi(self):
        """Return most recent Φ value (or None if not computed yet)"""
        return self.phi_result


class LeviathanWithIIT:
    """
    Complete Leviathan + IIT system
    Runs physics at high framerate, computes IIT asynchronously
    """

    def __init__(self, N=100000, k=20, max_delay=50, num_hubs=6):
        self.observatory = LeviathanObservatory(N=N, k=k, max_delay=max_delay)

        # Detect highest-degree nodes via rich-club analysis
        print(f"[System] Detecting {num_hubs} highest-degree hub nodes...")
        G = self.observatory._G
        if G is not None:
            degree_sorted = sorted(G.degree, key=lambda x: x[1], reverse=True)
            hub_indices = [node for node, _ in degree_sorted[:num_hubs]]
            print(
                f"[System] Hub nodes: {hub_indices} (degrees: {[d for _, d in degree_sorted[:num_hubs]]})"
            )
        else:
            hub_indices = list(range(num_hubs))
            print(
                f"[System] WARNING: Graph unavailable, using placeholder hubs: {hub_indices}"
            )

        self.iit_worker = LeviathanIITWorker(
            hub_indices=hub_indices, bin_resolution=2, tpm_window=5000  # Binary phases
        )

        self.prev_state = None
        self.step_count = 0
        self.phi_history = []

    def step(self):
        """One physics step + optional IIT update"""
        r = self.observatory.step()
        theta = self.observatory.get_phase_snapshot()

        # Update IIT worker with new state
        current_state = self.iit_worker.binarize_phases(theta)
        if self.prev_state is not None:
            self.iit_worker.accumulate_transition(self.prev_state, current_state)
        self.prev_state = current_state

        self.step_count += 1
        return r

    def compute_phi_periodic(self, interval=100):
        """Compute Φ every N steps"""
        if self.step_count % interval == 0:
            theta = self.observatory.get_phase_snapshot()
            phi = self.iit_worker.compute_phi(theta, prev_state=self.prev_state)
            if phi is not None:
                self.phi_history.append(phi)
                return phi
        return None

    def run_experiment(self, num_steps=1000, phi_interval=100, log_interval=50):
        """Run integrated physics + IIT experiment"""
        print(
            f"[System] Running {num_steps} steps with Φ computed every {phi_interval} steps\n"
        )

        start = time.time()
        for step in range(num_steps):
            r = self.step()
            phi = self.compute_phi_periodic(interval=phi_interval)

            if (step + 1) % log_interval == 0:
                elapsed = time.time() - start
                fps = (step + 1) / elapsed

                phi_str = f"Φ={phi:.4f}" if phi is not None else "Φ=---"
                print(f"Step {step+1:5d}: r={r:.4f} {phi_str} ({fps:.1f} FPS)")

        elapsed = time.time() - start
        print(
            f"\n[System] Complete in {elapsed:.2f}s (avg {num_steps/elapsed:.1f} FPS)"
        )

        return {
            "r_history": self.observatory.r_history,
            "phi_history": self.phi_history,
            "elapsed": elapsed,
        }


def main():
    print("=" * 70)
    print("LEVIATHAN CSR APEX v3.2 + IIT Integration Test")
    print("=" * 70)
    print()

    # Initialize combined system
    system = LeviathanWithIIT(N=50000, k=20, max_delay=50, num_hubs=6)

    # Run experiment: 500 steps, compute Φ every 50 steps
    results = system.run_experiment(num_steps=500, phi_interval=50, log_interval=50)

    # Summary statistics
    r_history = np.array(results["r_history"])
    phi_history = np.array(results["phi_history"])

    print()
    print("[Results] Physics Metrics")
    print(f"  Mean r: {np.mean(r_history):.4f}")
    print(f"  Std r:  {np.std(r_history):.4f}")
    print(f"  Range:  [{np.min(r_history):.4f}, {np.max(r_history):.4f}]")

    print()
    print("[Results] IIT Metrics")
    if len(phi_history) > 0:
        print(f"  Φ measurements: {len(phi_history)}")
        print(f"  Mean Φ: {np.mean(phi_history):.4f}")
        print(f"  Max Φ:  {np.max(phi_history):.4f}")
        print(f"  Correlation (r vs Φ): [pending analysis]")
    else:
        print("  (Φ history empty - increase num_steps or reduce phi_interval)")

    print()
    print("=" * 70)


if __name__ == "__main__":
    if not PYPHI_AVAILABLE:
        print("Install PyPhi: pip install pyphi")
        exit(1)

    main()
