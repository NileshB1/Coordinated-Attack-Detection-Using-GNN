"""One heterogeneous graph per flagged anomaly episode.

An episode is the set of anomaly events sharing an ``episode_id``. Its graph holds
the episode's VMs together with their backup images, storage pools and network
fabrics, and the edges among them. The question each graph answers is the one
from RQ1: given these flagged anomalies, are they coordinated (y = 1) or
coincidental (y = 0)?

Graph contents (``HeteroData``):

``data["vm"]``
    ``x``: float32 [n_vm, 4] raw telemetry in ``VM_FEATURES`` order (not scaled;
    scaling must be fitted on the training split later on).
    ``timestamp_hours``: absolute event time; ``rel_time_hours``: time since the
    episode's first event; ``event_order``: 0..n_vm-1 by time.
    ``anomaly_score``: the generator's weighted score, kept for reference.
``data[kind].global_index`` for every node type
    Index into ``InfrastructureGraph.node_ids[kind]``, to trace nodes back to ids.
``data[edge_type].edge_index`` for all six edge types (forward and reverse),
    empty when an episode has no such edge, so every graph has the same metadata.
``data.y``: long [1]; ``data.episode_id``: str.

VM nodes are ordered by event time for both labels. The episode id names the
label (``coordinated_3``), so it is for provenance only and must not be used as
a model input.
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from cadgnn.generator import EDGES_FILE, NODES_FILE, TELEMETRY_FILE
from cadgnn.graph.events import anomaly_episodes, clean_events, load_events
from cadgnn.graph.infrastructure import InfrastructureGraph
from cadgnn.graph.schema import (
    EDGE_TYPES,
    NODE_TYPES,
    TARGETS,
    VM_FEATURES,
    GraphConstructionError,
    reverse,
)

log = logging.getLogger(__name__)

EPISODE_GRAPHS_FILE = "episode_graphs.pt"
EPISODE_SUMMARY_FILE = "episode_summary.csv"


def load_generator_output(output_dir: str | Path) -> tuple[InfrastructureGraph, pd.DataFrame]:
    """Read the generator's node, edge and telemetry CSVs and validate them."""
    output_dir = Path(output_dir)
    paths = [output_dir / name for name in (NODES_FILE, EDGES_FILE, TELEMETRY_FILE)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise GraphConstructionError(
            "generator output not found: " + ", ".join(missing) + " (run the generator first)"
        )
    nodes = pd.read_csv(paths[0], dtype=str)
    edges = pd.read_csv(paths[1], dtype=str)
    infra = InfrastructureGraph.from_frames(nodes, edges)
    return infra, clean_events(load_events(paths[2]), infra)


def build_episode_graph(
    infra: InfrastructureGraph, episode: pd.DataFrame, episode_id: str
) -> HeteroData:
    """Graph for one episode; ``episode`` must be sorted by time with one row per VM."""
    labels = episode["label"].unique()
    if len(labels) != 1 or labels[0] not in TARGETS:
        raise GraphConstructionError(f"episode {episode_id} has label(s) {list(labels)}")

    vm_index = np.array([infra.locate(vm)[1] for vm in episode["vm"]], dtype=np.int64)
    selected = infra.neighbourhood(vm_index)
    edges = infra.subgraph_edges(selected)

    times = episode["timestamp_hours"].to_numpy(dtype=np.float64)
    data = HeteroData()
    data["vm"].x = torch.tensor(episode[list(VM_FEATURES)].to_numpy(), dtype=torch.float32)
    data["vm"].timestamp_hours = torch.tensor(times, dtype=torch.float32)
    data["vm"].rel_time_hours = torch.tensor(times - times.min(), dtype=torch.float32)
    data["vm"].event_order = torch.arange(len(times))
    data["vm"].anomaly_score = torch.tensor(
        episode["anomaly_score"].to_numpy(), dtype=torch.float32
    )
    for kind in NODE_TYPES:
        data[kind].num_nodes = len(selected[kind])
        data[kind].global_index = torch.from_numpy(selected[kind])
    for et in EDGE_TYPES:
        edge_index = torch.from_numpy(edges[et])
        data[et].edge_index = edge_index
        data[reverse(et)].edge_index = edge_index.flip(0)

    data.y = torch.tensor([TARGETS[labels[0]]], dtype=torch.long)
    data.episode_id = episode_id
    return data


def build_episode_graphs(
    infra: InfrastructureGraph, events: pd.DataFrame, min_events: int = 2
) -> list[HeteroData]:
    """One graph per anomaly episode, in sorted episode-id order.

    Episodes with fewer than ``min_events`` events are skipped: a single event
    cannot be called coordinated or coincidental. ``events`` should already have
    been through ``clean_events``.
    """
    if min_events < 1:
        raise ValueError(f"min_events must be at least 1, got {min_events}")
    graphs, skipped = [], 0
    for episode_id, episode in anomaly_episodes(events).items():
        if len(episode) < min_events:
            skipped += 1
            continue
        graphs.append(build_episode_graph(infra, episode, episode_id))
    if skipped:
        log.warning("skipped %d episode(s) with fewer than %d events", skipped, min_events)
    if not graphs:
        log.warning("no anomaly episodes to build graphs from")
    return graphs


def episode_summary(graphs: list[HeteroData]) -> pd.DataFrame:
    """Per-episode size, time span and connectivity, mainly for inspection and reporting."""
    rows = []
    for data in graphs:
        rel = data["vm"].rel_time_hours
        rows.append(
            {
                "episode_id": data.episode_id,
                "y": int(data.y.item()),
                "n_vm": data["vm"].num_nodes,
                "n_backup_image": data["backup_image"].num_nodes,
                "n_storage": data["storage"].num_nodes,
                "n_network": data["network"].num_nodes,
                "n_edges": sum(data[et].edge_index.size(1) for et in EDGE_TYPES),
                "time_span_hours": float(rel.max()) if len(rel) else 0.0,
                "n_components": _count_components(data),
            }
        )
    columns = ["episode_id", "y", "n_vm", "n_backup_image", "n_storage", "n_network",
               "n_edges", "time_span_hours", "n_components"]  # fmt: skip
    return pd.DataFrame(rows, columns=columns)


def save_episode_graphs(graphs: list[HeteroData], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, path)


def load_episode_graphs(path: str | Path) -> list[HeteroData]:
    path = Path(path)
    if not path.is_file():
        raise GraphConstructionError(f"episode graph file not found: {path}")
    # HeteroData is not on torch's safe-load list, so a full unpickle is needed.
    # Only load files this project wrote.
    return torch.load(path, weights_only=False)


def _count_components(data: HeteroData) -> int:
    g = nx.Graph()
    for kind in NODE_TYPES:
        g.add_nodes_from((kind, i) for i in range(data[kind].num_nodes))
    for et in EDGE_TYPES:
        src, _, dst = et
        for s, d in data[et].edge_index.t().tolist():
            g.add_edge((src, s), (dst, d))
    return nx.number_connected_components(g)
