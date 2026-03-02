#!/usr/bin/env python3
"""
partition.py — METIS-based graph partitioning for multi-GPU Leviathan

Partitions a Watts-Strogatz (or arbitrary) graph into K subgraphs
with minimal edge cuts, builds per-partition CSR arrays and halo maps.

Usage:
    from partition import partition_graph
    parts = partition_graph(G, num_gpus=4)
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, NamedTuple

try:
    import metis

    METIS_AVAILABLE = True
except ImportError:
    METIS_AVAILABLE = False
    print("WARNING: python-metis not installed. Using naive partitioning.")
    print("Install with: pip install metis")


class PartitionData(NamedTuple):
    """Data for one GPU partition."""

    gpu_id: int
    local_N: int
    row_ptr: np.ndarray  # int32, CSR row pointers
    col_idx: np.ndarray  # int32, CSR column indices (LOCAL indices)
    delays: np.ndarray  # uint8
    weights: np.ndarray  # float32
    theta: np.ndarray  # float32, initial phases
    theta_hat: np.ndarray  # float32, predicted phases
    omega: np.ndarray  # float32, natural frequencies
    global_to_local: Dict[int, int]
    local_to_global: np.ndarray

    # Halo info
    halo_recv_global_ids: np.ndarray  # int32 — global IDs of nodes we need
    halo_send_local_ids: np.ndarray  # int32 — local IDs of nodes to send


def partition_graph(
    G: nx.Graph,
    num_gpus: int,
    max_delay: int = 50,
    theta: np.ndarray = None,
    omega: np.ndarray = None,
    seed: int = 42,
) -> List[PartitionData]:
    """
    Partition a networkx graph into num_gpus parts.

    Args:
        G: NetworkX graph (undirected)
        num_gpus: Number of GPU partitions
        max_delay: Max coupling delay
        theta: Initial phase array [N] (random if None)
        omega: Natural frequency array [N] (random if None)
        seed: Random seed

    Returns:
        List of PartitionData, one per GPU
    """
    N = G.number_of_nodes()
    np.random.seed(seed)

    if theta is None:
        theta = np.random.uniform(0, 2 * np.pi, N).astype(np.float32)
    if omega is None:
        omega = np.random.normal(1.0, 0.1, N).astype(np.float32)

    print(f"[Partition] Partitioning {N} nodes into {num_gpus} parts...")

    # --- Step 1: Partition assignment ---
    if METIS_AVAILABLE and num_gpus > 1:
        G_metis = metis.networkx_to_metis(G)
        edgecuts, membership = metis.part_graph(G_metis, nparts=num_gpus, objtype="cut")
        print(f"[Partition] METIS: {edgecuts} edge cuts")
    else:
        # Naive: contiguous blocks
        membership = [0] * N
        block_size = (N + num_gpus - 1) // num_gpus
        for i in range(N):
            membership[i] = min(i // block_size, num_gpus - 1)
        if not METIS_AVAILABLE:
            print("[Partition] Using naive contiguous partitioning")

    # --- Step 2: Build per-partition data ---
    partitions = []
    nodes_list = list(G.nodes())

    for gpu_id in range(num_gpus):
        # Identify local nodes
        local_nodes = [n for n in nodes_list if membership[n] == gpu_id]
        local_N = len(local_nodes)

        # Build index maps
        global_to_local = {g: l for l, g in enumerate(local_nodes)}
        local_to_global = np.array(local_nodes, dtype=np.int32)

        # Identify halo nodes: remote nodes connected to local nodes
        halo_set = set()
        for local_node in local_nodes:
            for neighbor in G.neighbors(local_node):
                if membership[neighbor] != gpu_id:
                    halo_set.add(neighbor)
        halo_recv_global = np.array(sorted(halo_set), dtype=np.int32)

        # Build local CSR (edges within this partition only)
        row_ptr = [0]
        col_indices = []
        delay_list = []
        weight_list = []

        for local_idx, global_node in enumerate(local_nodes):
            count = 0
            for neighbor in G.neighbors(global_node):
                if neighbor in global_to_local:
                    col_indices.append(global_to_local[neighbor])
                    delay_list.append(np.random.randint(1, max_delay))
                    weight_list.append(np.random.uniform(0.01, 0.1))
                    count += 1
            row_ptr.append(row_ptr[-1] + count)

        row_ptr = np.array(row_ptr, dtype=np.int32)
        col_idx = np.array(col_indices, dtype=np.int32)
        delays = np.array(delay_list, dtype=np.uint8)
        weights = np.array(weight_list, dtype=np.float32)

        # Extract state for local nodes
        local_theta = theta[local_nodes].copy()
        local_theta_hat = local_theta.copy()
        local_omega = omega[local_nodes].copy()

        # Identify which local nodes other partitions need (halo send)
        send_set = set()
        for local_node in local_nodes:
            for neighbor in G.neighbors(local_node):
                if membership[neighbor] != gpu_id:
                    send_set.add(global_to_local[local_node])
        halo_send_local = np.array(sorted(send_set), dtype=np.int32)

        part = PartitionData(
            gpu_id=gpu_id,
            local_N=local_N,
            row_ptr=row_ptr,
            col_idx=col_idx,
            delays=delays,
            weights=weights,
            theta=local_theta,
            theta_hat=local_theta_hat,
            omega=local_omega,
            global_to_local=global_to_local,
            local_to_global=local_to_global,
            halo_recv_global_ids=halo_recv_global,
            halo_send_local_ids=halo_send_local,
        )
        partitions.append(part)

        print(
            f"[Partition] GPU {gpu_id}: {local_N} nodes, "
            f"{len(col_idx)} edges, "
            f"{len(halo_recv_global)} halo_recv, "
            f"{len(halo_send_local)} halo_send"
        )

    return partitions


def demo():
    """Demonstrate partitioning on a small graph."""
    print("=" * 60)
    print("Partition Demo: 10000 nodes → 4 GPUs")
    print("=" * 60)
    print()

    G = nx.watts_strogatz_graph(10000, k=20, p=0.2)
    parts = partition_graph(G, num_gpus=4)

    print()
    total_edges = sum(len(p.col_idx) for p in parts)
    total_halo = sum(len(p.halo_recv_global_ids) for p in parts)
    print(f"[Summary] Total internal edges: {total_edges}")
    print(f"[Summary] Total halo nodes: {total_halo}")
    print(f"[Summary] Halo overhead: {total_halo / G.number_of_nodes():.1%}")


if __name__ == "__main__":
    demo()
