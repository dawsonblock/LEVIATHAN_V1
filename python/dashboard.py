#!/usr/bin/env python3
"""
Leviathan CSR Apex v3.3 — Real-Time Observatory Dashboard
Live visualization of phase dynamics, weight evolution, and IIT metrics.
v3.3 Optimized: decimation, WebGL rendering, efficient updates

Usage:
    python3 dashboard.py                    # Simulated data mode
    python3 dashboard.py --live             # Connect to running engine
    python3 dashboard.py --live --port 8051 # Custom port
"""

import argparse
import json
import threading
import time
from collections import deque

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, callback_context
from dash.dependencies import Input, Output, State

# ---------------------------------------------------------------------------
# Telemetry Data Store (Thread-safe ring buffer)
# ---------------------------------------------------------------------------


class TelemetryStore:
    """Thread-safe ring buffer for real-time telemetry data."""

    # [OPT #16] Max grid dimension for phase heatmap
    MAX_HEATMAP_SIDE = 64

    def __init__(self, max_history=2000):
        self.max_history = max_history
        self.lock = threading.Lock()

        # Time series
        self.steps = deque(maxlen=max_history)
        self.r_history = deque(maxlen=max_history)
        self.g_history = deque(maxlen=max_history)
        self.phi_history = deque(maxlen=max_history)
        self.phi_steps = deque(maxlen=max_history)

        # Snapshots (latest only)
        self.theta_snapshot = np.zeros(100)
        self.weight_stats = {
            "mean": 0,
            "std": 0,
            "min": 0,
            "max": 0,
            "histogram": [],
            "bin_edges": [],
        }

        # Metadata
        self.N = 0
        self.total_steps = 0
        self.fps = 0.0
        self.is_running = False

    def push_step(self, step, r, g, fps=0.0):
        with self.lock:
            self.steps.append(step)
            self.r_history.append(r)
            self.g_history.append(g)
            self.total_steps = step
            self.fps = fps
            self.is_running = True

    def push_phi(self, step, phi):
        with self.lock:
            self.phi_steps.append(step)
            self.phi_history.append(phi)

    def push_snapshot(self, theta, weight_stats=None):
        with self.lock:
            self.theta_snapshot = np.array(theta)
            if weight_stats:
                self.weight_stats = weight_stats

    def get_time_series(self):
        with self.lock:
            return {
                "steps": list(self.steps),
                "r": list(self.r_history),
                "g": list(self.g_history),
                "phi_steps": list(self.phi_steps),
                "phi": list(self.phi_history),
            }

    def get_snapshot(self):
        with self.lock:
            return {
                "theta": self.theta_snapshot.copy(),
                "weight_stats": dict(self.weight_stats),
                "total_steps": self.total_steps,
                "fps": self.fps,
                "is_running": self.is_running,
            }


# ---------------------------------------------------------------------------
# Simulated Data Source (for standalone testing without H100)
# ---------------------------------------------------------------------------


class SimulatedEngine(threading.Thread):
    """Generates synthetic Kuramoto-like telemetry for dashboard testing."""

    def __init__(self, store: TelemetryStore, N=1000):
        super().__init__(daemon=True)
        self.store = store
        self.N = N
        self.store.N = N
        self.running = True
        self.paused = False
        self.gain = 1.5

        # State
        self.theta = np.random.uniform(0, 2 * np.pi, N).astype(np.float32)
        self.omega = np.random.normal(1.0, 0.1, N).astype(np.float32)
        self.weights = np.random.uniform(0.01, 0.1, N * 10).astype(np.float32)

    def run(self):
        step = 0
        dt = 0.05
        r0 = 0.5
        beta = 0.01
        t_start = time.time()

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            # Simplified Kuramoto step
            cos_sum = np.mean(np.cos(self.theta))
            sin_sum = np.mean(np.sin(self.theta))
            r = np.sqrt(cos_sum**2 + sin_sum**2)
            psi = np.arctan2(sin_sum, cos_sum)

            coupling = self.gain * r * np.sin(psi - self.theta)
            self.theta += dt * (self.omega + coupling)
            self.theta %= 2 * np.pi

            # Gain controller
            self.gain += dt * beta * (r0 - r)
            self.gain = np.clip(self.gain, 0.1, 5.0)

            # Weight evolution (drift + decay)
            self.weights += dt * (
                0.001 * np.random.randn(len(self.weights)) - 0.0001 * self.weights
            )
            self.weights = np.clip(self.weights, 0.0, 2.0)

            elapsed = time.time() - t_start
            fps = (step + 1) / elapsed if elapsed > 0 else 0

            self.store.push_step(step, float(r), float(self.gain), fps)

            # Periodic snapshots
            if step % 5 == 0:
                hist, bin_edges = np.histogram(self.weights, bins=50, range=(0, 0.5))
                self.store.push_snapshot(
                    self.theta,
                    {
                        "mean": float(np.mean(self.weights)),
                        "std": float(np.std(self.weights)),
                        "min": float(np.min(self.weights)),
                        "max": float(np.max(self.weights)),
                        "histogram": hist.tolist(),
                        "bin_edges": bin_edges.tolist(),
                    },
                )

            # Simulated Φ (every 100 steps)
            if step % 100 == 0 and step > 0:
                phi_fake = float(
                    0.3 + 0.2 * np.sin(step / 200) + 0.05 * np.random.randn()
                )
                self.store.push_phi(step, phi_fake)

            step += 1
            time.sleep(0.005)  # ~200 Hz simulated rate


# ---------------------------------------------------------------------------
# Dashboard Layout
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#0a0e17",
    "card": "#111827",
    "border": "#1e293b",
    "text": "#e2e8f0",
    "text_dim": "#64748b",
    "accent": "#3b82f6",
    "accent2": "#8b5cf6",
    "green": "#10b981",
    "red": "#ef4444",
    "orange": "#f59e0b",
    "cyan": "#06b6d4",
}


def make_card(title, graph_id, height="340px"):
    return html.Div(
        [
            html.Div(
                title,
                style={
                    "color": COLORS["text_dim"],
                    "fontSize": "11px",
                    "fontWeight": "600",
                    "textTransform": "uppercase",
                    "letterSpacing": "1.5px",
                    "padding": "12px 16px 0 16px",
                },
            ),
            dcc.Graph(
                id=graph_id,
                style={"height": height},
                config={"displayModeBar": False, "staticPlot": False},
            ),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "borderRadius": "12px",
            "border": f"1px solid {COLORS['border']}",
            "overflow": "hidden",
        },
    )


def make_stat_pill(label, value_id, color=COLORS["accent"]):
    return html.Div(
        [
            html.Span(
                label,
                style={
                    "color": COLORS["text_dim"],
                    "fontSize": "10px",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                },
            ),
            html.Div(
                id=value_id,
                style={
                    "color": color,
                    "fontSize": "22px",
                    "fontWeight": "700",
                    "fontFamily": "JetBrains Mono, monospace",
                },
            ),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "borderRadius": "10px",
            "border": f"1px solid {COLORS['border']}",
            "padding": "12px 18px",
            "textAlign": "center",
            "flex": "1",
        },
    )


def create_layout():
    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.Div(
                        [
                            html.H1(
                                "LEVIATHAN OBSERVATORY",
                                style={
                                    "margin": "0",
                                    "fontSize": "18px",
                                    "fontWeight": "700",
                                    "letterSpacing": "3px",
                                    "color": COLORS["text"],
                                },
                            ),
                            html.Div(
                                "CSR APEX v3.3 — Real-Time Cognitive Simulator",
                                style={
                                    "fontSize": "11px",
                                    "color": COLORS["text_dim"],
                                    "letterSpacing": "1px",
                                    "marginTop": "2px",
                                },
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                id="status-indicator",
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "8px",
                                },
                            ),
                        ]
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "padding": "16px 24px",
                    "borderBottom": f"1px solid {COLORS['border']}",
                },
            ),
            # Stat pills row
            html.Div(
                [
                    make_stat_pill("Synchrony (r)", "stat-r", COLORS["cyan"]),
                    make_stat_pill("Gain (g)", "stat-g", COLORS["orange"]),
                    make_stat_pill("Φ (IIT)", "stat-phi", COLORS["accent2"]),
                    make_stat_pill("FPS", "stat-fps", COLORS["green"]),
                    make_stat_pill("Steps", "stat-steps", COLORS["text"]),
                ],
                style={"display": "flex", "gap": "10px", "padding": "16px 24px"},
            ),
            # Main grid
            html.Div(
                [
                    # Left column: Phase heatmap + Weight distribution
                    html.Div(
                        [
                            make_card(
                                "Phase Topology — θ ∈ [0, 2π)", "phase-heatmap", "300px"
                            ),
                            html.Div(style={"height": "10px"}),
                            make_card(
                                "Synaptic Weight Distribution",
                                "weight-histogram",
                                "260px",
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                    # Right column: Time series
                    html.Div(
                        [
                            make_card(
                                "Metastability — Order Parameter r & Gain g",
                                "r-timeseries",
                                "272px",
                            ),
                            html.Div(style={"height": "10px"}),
                            make_card(
                                "Integrated Information — Φ", "phi-timeseries", "288px"
                            ),
                        ],
                        style={"flex": "1.2"},
                    ),
                ],
                style={"display": "flex", "gap": "10px", "padding": "0 24px 24px 24px"},
            ),
            # Update interval
            dcc.Interval(id="update-interval", interval=250, n_intervals=0),
        ],
        style={
            "backgroundColor": COLORS["bg"],
            "minHeight": "100vh",
            "fontFamily": "'Inter', -apple-system, sans-serif",
            "color": COLORS["text"],
        },
    )


# ---------------------------------------------------------------------------
# Plotly Figure Builders
# ---------------------------------------------------------------------------

PLOT_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text_dim"], size=10),
    margin=dict(l=45, r=15, t=10, b=35),
    xaxis=dict(
        gridcolor="#1e293b", zerolinecolor="#1e293b", showgrid=True, gridwidth=1
    ),
    yaxis=dict(
        gridcolor="#1e293b", zerolinecolor="#1e293b", showgrid=True, gridwidth=1
    ),
)


def build_phase_heatmap(theta):
    """Render node phases on a 2D grid heatmap with decimation."""
    n = len(theta)
    max_side = TelemetryStore.MAX_HEATMAP_SIDE

    # [OPT #16] Subsample for large node counts
    if n > max_side * max_side:
        stride = max(1, n // (max_side * max_side))
        theta = theta[::stride]
        # Truncate to fit in grid
        max_n = max_side * max_side
        theta = theta[:max_n]
        n = len(theta)

    side = int(np.ceil(np.sqrt(n)))
    side = min(side, max_side)
    grid = np.full(side * side, np.nan)
    grid[:n] = theta
    grid = grid.reshape(side, side)

    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            colorscale=[
                [0.0, "#0d1b2a"],
                [0.15, "#1b263b"],
                [0.3, "#3b82f6"],
                [0.5, "#06b6d4"],
                [0.65, "#10b981"],
                [0.8, "#f59e0b"],
                [0.95, "#ef4444"],
                [1.0, "#7c3aed"],
            ],
            zmin=0,
            zmax=2 * np.pi,
            showscale=True,
            colorbar=dict(
                title=dict(text="θ", side="right"),
                thickness=8,
                len=0.8,
                tickvals=[0, np.pi, 2 * np.pi],
                ticktext=["0", "π", "2π"],
            ),
            hovertemplate="Node (%{x},%{y})<br>θ = %{z:.3f} rad<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text_dim"], size=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        margin=dict(l=5, r=5, t=5, b=5),
    )
    return fig


def build_weight_histogram(weight_stats):
    """Render synaptic weight distribution."""
    hist = weight_stats.get("histogram", [])
    edges = weight_stats.get("bin_edges", [])

    if not hist or not edges:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT_BASE)
        return fig

    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]

    fig = go.Figure(
        data=go.Bar(
            x=centers,
            y=hist,
            marker=dict(
                color=hist,
                colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#06b6d4"]],
                line=dict(width=0),
            ),
            hovertemplate="w = %{x:.4f}<br>Count: %{y}<extra></extra>",
        )
    )
    layout = dict(PLOT_LAYOUT_BASE)
    layout["xaxis"] = dict(layout["xaxis"], title="Weight (w)")
    layout["yaxis"] = dict(layout["yaxis"], title="Count")
    fig.update_layout(**layout, bargap=0.02)

    # Annotations
    mean = weight_stats.get("mean", 0)
    fig.add_vline(x=mean, line=dict(color=COLORS["orange"], dash="dash", width=1))
    fig.add_annotation(
        x=mean,
        y=max(hist) * 0.9 if hist else 1,
        text=f"μ={mean:.4f}",
        showarrow=False,
        font=dict(color=COLORS["orange"], size=10),
    )
    return fig


def build_r_timeseries(ts):
    """Render synchrony r and gain g time series."""
    fig = go.Figure()

    if ts["steps"]:
        fig.add_trace(
            go.Scattergl(
                x=ts["steps"],
                y=ts["r"],
                mode="lines",
                name="r (synchrony)",
                line=dict(color=COLORS["cyan"], width=1.5),
                hovertemplate="Step %{x}<br>r = %{y:.4f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scattergl(
                x=ts["steps"],
                y=ts["g"],
                mode="lines",
                name="g (gain)",
                line=dict(color=COLORS["orange"], width=1.5, dash="dot"),
                yaxis="y2",
                hovertemplate="Step %{x}<br>g = %{y:.4f}<extra></extra>",
            )
        )
        # Target line
        fig.add_hline(
            y=0.5, line=dict(color=COLORS["green"], dash="dash", width=1), opacity=0.4
        )
        fig.add_annotation(
            x=ts["steps"][-1],
            y=0.5,
            text="r₀ = 0.5",
            showarrow=False,
            xanchor="right",
            font=dict(color=COLORS["green"], size=9),
        )

    layout = dict(PLOT_LAYOUT_BASE)
    layout["yaxis"] = dict(layout["yaxis"], title="r", range=[0, 1])
    layout["yaxis2"] = dict(
        title=dict(text="g", font=dict(color=COLORS["orange"])),
        overlaying="y",
        side="right",
        gridcolor="rgba(0,0,0,0)",
        range=[0, 5],
        tickfont=dict(color=COLORS["orange"]),
    )
    layout["legend"] = dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=9))
    fig.update_layout(**layout)
    return fig


def build_phi_timeseries(ts):
    """Render Φ over time."""
    fig = go.Figure()

    if ts["phi_steps"]:
        fig.add_trace(
            go.Scattergl(
                x=ts["phi_steps"],
                y=ts["phi"],
                mode="lines+markers",
                name="Φ",
                line=dict(color=COLORS["accent2"], width=2),
                marker=dict(size=4, color=COLORS["accent2"]),
                hovertemplate="Step %{x}<br>Φ = %{y:.4f}<extra></extra>",
            )
        )

        # Rolling average
        if len(ts["phi"]) >= 5:
            kernel = np.ones(5) / 5
            smooth = np.convolve(ts["phi"], kernel, mode="valid")
            smooth_x = ts["phi_steps"][4:]
            fig.add_trace(
                go.Scattergl(
                    x=smooth_x,
                    y=smooth.tolist(),
                    mode="lines",
                    name="Φ (smoothed)",
                    line=dict(color=COLORS["accent2"], width=1, dash="dash"),
                    opacity=0.5,
                )
            )

    layout = dict(PLOT_LAYOUT_BASE)
    layout["yaxis"] = dict(layout["yaxis"], title="Φ (Integrated Information)")
    layout["legend"] = dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=9))
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------


def create_app(store: TelemetryStore):
    app = Dash(
        __name__,
        title="Leviathan Observatory",
        update_title=None,
    )
    app.layout = create_layout()

    @app.callback(
        [
            Output("phase-heatmap", "figure"),
            Output("weight-histogram", "figure"),
            Output("r-timeseries", "figure"),
            Output("phi-timeseries", "figure"),
            Output("stat-r", "children"),
            Output("stat-g", "children"),
            Output("stat-phi", "children"),
            Output("stat-fps", "children"),
            Output("stat-steps", "children"),
            Output("status-indicator", "children"),
        ],
        [Input("update-interval", "n_intervals")],
    )
    def update_all(n):
        ts = store.get_time_series()
        snap = store.get_snapshot()

        # Figures
        fig_phase = build_phase_heatmap(snap["theta"])
        fig_weight = build_weight_histogram(snap["weight_stats"])
        fig_r = build_r_timeseries(ts)
        fig_phi = build_phi_timeseries(ts)

        # Stats
        r_val = f"{ts['r'][-1]:.4f}" if ts["r"] else "—"
        g_val = f"{ts['g'][-1]:.3f}" if ts["g"] else "—"
        phi_val = f"{ts['phi'][-1]:.4f}" if ts["phi"] else "—"
        fps_val = f"{snap['fps']:.0f}" if snap["fps"] else "—"
        steps_val = f"{snap['total_steps']:,}"

        # Status indicator
        if snap["is_running"]:
            status = html.Div(
                [
                    html.Div(
                        style={
                            "width": "8px",
                            "height": "8px",
                            "borderRadius": "50%",
                            "backgroundColor": COLORS["green"],
                            "boxShadow": f"0 0 6px {COLORS['green']}",
                        }
                    ),
                    html.Span(
                        "LIVE",
                        style={
                            "fontSize": "11px",
                            "fontWeight": "600",
                            "color": COLORS["green"],
                            "letterSpacing": "1px",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "6px"},
            )
        else:
            status = html.Span(
                "IDLE", style={"fontSize": "11px", "color": COLORS["text_dim"]}
            )

        return (
            fig_phase,
            fig_weight,
            fig_r,
            fig_phi,
            r_val,
            g_val,
            phi_val,
            fps_val,
            steps_val,
            status,
        )

    return app


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Leviathan Observatory Dashboard")
    parser.add_argument(
        "--live", action="store_true", help="Connect to running Leviathan engine"
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Dashboard port (default: 8050)"
    )
    parser.add_argument(
        "--nodes", type=int, default=1000, help="Simulated node count (default: 1000)"
    )
    args = parser.parse_args()

    store = TelemetryStore()

    if args.live:
        print("[Dashboard] Live mode — expecting telemetry on port 8051")
        print(
            "[Dashboard] (Live engine integration requires running leviathan_h100.py with --telemetry)"
        )
    else:
        print(f"[Dashboard] Simulated mode — {args.nodes} synthetic Kuramoto nodes")
        engine = SimulatedEngine(store, N=args.nodes)
        engine.start()

    app = create_app(store)

    print(f"\n{'='*60}")
    print(f"  LEVIATHAN OBSERVATORY — http://localhost:{args.port}")
    print(f"{'='*60}\n")

    app.run(debug=False, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
