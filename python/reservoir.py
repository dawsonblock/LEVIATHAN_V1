#!/usr/bin/env python3
"""
python/reservoir.py — Reservoir Computing interface for Leviathan.

Uses the Kuramoto phase dynamics engine as a nonlinear reservoir substrate.
Features: signal injection via phase perturbation and ridge regression readout.
"""

import numpy as np
import sys
from sklearn.linear_model import Ridge

# Ensure we can import the engine
sys.path.insert(0, ".")
try:
    from python.leviathan_h100 import LeviathanObservatory
except ImportError:
    from leviathan_h100 import LeviathanObservatory


class ReservoirComputer:
    """
    Kuramoto Reservoir Computing (KRC) Wrapper.

    Attributes:
        obs: LeviathanObservatory instance
        input_nodes: Indices of nodes receiving input
        readout_nodes: Indices of nodes sampled for readout
        readout_model: Trained Ridge regression model
    """

    def __init__(self, N=5000, k=20, max_delay=10, n_input=10, n_readout=100, seed=42):
        self.obs = LeviathanObservatory(N=N, k=k, max_delay=max_delay, seed=seed)

        # Select random nodes for input and readout
        rng = np.random.default_rng(seed)
        all_nodes = np.arange(N)
        rng.shuffle(all_nodes)

        self.input_nodes = all_nodes[:n_input]
        self.readout_nodes = all_nodes[n_input : n_input + n_readout]

        self.readout_model = None

    def inject(self, value, strength=1.0):
        """Inject scalar value into reservoir via phase perturbation"""
        theta = self.obs.get_phase_snapshot()
        # Map value to phase kick
        kick = value * strength
        theta[self.input_nodes] += kick
        # Wrap to [0, 2pi)
        theta = theta % (2 * np.pi)
        self.obs.set_phase_snapshot(theta)

    def get_state(self):
        """Sample readout nodes and return cos/sin features"""
        theta = self.obs.get_phase_snapshot()
        phases = theta[self.readout_nodes]
        # Return [cos, sin] features for linear separability
        return np.concatenate([np.cos(phases), np.sin(phases)])

    def run_sequence(self, input_sequence, strength=1.0, warmup=100):
        """Run a sequence and collect reservoir states"""
        # Warmup
        for _ in range(warmup):
            self.obs.step()

        states = []
        for val in input_sequence:
            self.inject(val, strength=strength)
            self.obs.step()
            states.append(self.get_state())

        return np.array(states)

    def train(self, input_sequence, target_sequence, alpha=1.0):
        """Train a ridge regression readout on the reservoir states"""
        X = self.run_sequence(input_sequence)
        y = target_sequence

        self.readout_model = Ridge(alpha=alpha)
        self.readout_model.fit(X, y)

        # Return training error (MSE)
        y_pred = self.readout_model.predict(X)
        return np.mean((y - y_pred) ** 2)

    def predict(self, input_sequence):
        """Predict target from input using trained readout"""
        if self.readout_model is None:
            raise ValueError("Model not trained.")

        X = self.run_sequence(input_sequence)
        return self.readout_model.predict(X)


if __name__ == "__main__":
    # Quick test: Copy an input signal with delay
    print("Testing Reservoir Computer: Memory Recall Task")
    rc = ReservoirComputer(N=1000, n_readout=200)

    # Generate random input
    T = 500
    u = np.random.uniform(-1, 1, T)
    # Target: input delayed by 5 steps
    delay = 5
    y_target = np.zeros(T)
    y_target[delay:] = u[:-delay]

    # Train
    mse = rc.train(u[:400], y_target[:400])
    print(f"Train MSE: {mse:.6f}")

    # Test
    y_pred = rc.predict(u[400:])
    test_mse = np.mean((y_target[400:] - y_pred) ** 2)
    print(f"Test MSE:  {test_mse:.6f}")
