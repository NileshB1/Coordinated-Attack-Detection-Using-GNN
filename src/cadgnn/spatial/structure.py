from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Batch

from cadgnn.graph import EDGE_TYPES, NODE_TYPES


def node_degrees(batch: Batch) -> dict[str, Tensor]:
    """Number of edges touching each node within its episode graph.

    Only the forward edge types are counted, so each link counts once.
    A network fabric used by four of the episode's VMs gets degree 4.
    """
    degrees = {
        kind: torch.zeros(batch[kind].num_nodes, dtype=torch.long, device=batch["vm"].x.device)
        for kind in NODE_TYPES
    }
    for et in EDGE_TYPES:
        src, _, dst = et
        edge_index = batch[et].edge_index
        degrees[src] += torch.bincount(edge_index[0], minlength=batch[src].num_nodes)
        degrees[dst] += torch.bincount(edge_index[1], minlength=batch[dst].num_nodes)
    return degrees
