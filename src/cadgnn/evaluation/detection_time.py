"""Time to detect: how far into a campaign the model first flags it.

Each coordinated episode is replayed as it would have unfolded: after its k-th
event the model sees only those k events (and the infrastructure they touch).
The detection time is the hours from the first flagged event to the event at
which the score first reaches the threshold. The model was trained on whole
episodes, so very short prefixes are outside what it saw in training.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch_geometric.data import HeteroData

from cadgnn.evaluation.inference import predict
from cadgnn.fusion import CoordinationModel


def prefix_graph(graph: HeteroData, k: int) -> HeteroData:
    """The episode graph as it looked after its first k events."""
    n = graph["vm"].num_nodes
    if not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and {n}, got {k}")
    vms = torch.nonzero(graph["vm"].event_order < k).flatten()

    def targets(edge_type, sources):
        edge_index = graph[edge_type].edge_index
        return torch.unique(edge_index[1, torch.isin(edge_index[0], sources)])

    images = targets(("vm", "has_backup", "backup_image"), vms)
    subset = {
        "vm": vms,
        "backup_image": images,
        "storage": targets(("backup_image", "stored_on", "storage"), images),
        "network": targets(("vm", "attached_to", "network"), vms),
    }
    return graph.subgraph(subset)


def time_to_detect(
    model: CoordinationModel,
    graphs: Sequence[HeteroData],
    threshold: float,
    min_events: int = 2,
    device: torch.device | str = "cpu",
) -> dict:
    """Detection times for the coordinated episodes in ``graphs``."""
    campaigns = [g for g in graphs if int(g.y) == 1]
    prefixes, owners = [], []
    for g in campaigns:
        for k in range(min(min_events, g["vm"].num_nodes), g["vm"].num_nodes + 1):
            prefixes.append(prefix_graph(g, k))
            owners.append((g.episode_id, k))
    if not prefixes:
        return {
            "n_campaigns": 0,
            "detected": 0,
            "detection_rate": None,
            "mean_hours_to_detect": None,
            "median_hours_to_detect": None,
            "episodes": [],
        }

    scores = predict(model, prefixes, device=device).scores
    first_hit: dict[str, int] = {}
    for (episode_id, k), score in zip(owners, scores, strict=True):
        if score >= threshold and episode_id not in first_hit:
            first_hit[episode_id] = k

    episodes = []
    for g in campaigns:
        k = first_hit.get(g.episode_id)
        times = g["vm"].rel_time_hours[torch.argsort(g["vm"].event_order)]
        episodes.append({
            "episode_id": g.episode_id,
            "n_events": g["vm"].num_nodes,
            "detected": k is not None,
            "events_needed": k,
            "hours_to_detect": float(times[k - 1]) if k is not None else None,
        })  # fmt: skip

    hours = [e["hours_to_detect"] for e in episodes if e["detected"]]
    return {
        "n_campaigns": len(campaigns),
        "detected": len(hours),
        "detection_rate": len(hours) / len(campaigns),
        "mean_hours_to_detect": float(np.mean(hours)) if hours else None,
        "median_hours_to_detect": float(np.median(hours)) if hours else None,
        "episodes": episodes,
    }
