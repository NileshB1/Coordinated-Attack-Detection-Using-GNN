"""Spatial topology stream: is there a plausible path between the flagged VMs?

Graph attention runs over the episode's VMs, backup images, storage pools and
network fabrics. Only VM telemetry and graph structure go in; event times,
labels, episode ids and global node indices are not used.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, HeteroConv, global_max_pool, global_mean_pool

from cadgnn.config import ModelConfig
from cadgnn.graph import NODE_TYPES, VM_FEATURES, all_edge_types
from cadgnn.models import NodeEncoder
from cadgnn.spatial.structure import node_degrees


class SpatialTopologyStream(nn.Module):
    """Heterogeneous GAT with one attention layer per edge type.

    Resource nodes have no features, so every node also gets log1p of its degree
    in the episode graph; that is how a network shared by several flagged VMs
    differs from one used by a single VM. Two layers let a VM hear from the other
    VMs on its network (vm -> network -> vm).
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.0,
        n_features: int = len(VM_FEATURES),
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.encoder = NodeEncoder(hidden_dim, n_features=n_features)
        self.degree = nn.Linear(1, hidden_dim)
        self.convs = nn.ModuleList(
            HeteroConv(
                {
                    et: GATConv(
                        (hidden_dim, hidden_dim),
                        hidden_dim,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                        add_self_loops=False,
                    )
                    for et in all_edge_types()
                },
                aggr="sum",
            )
            for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.ModuleDict({kind: nn.LayerNorm(hidden_dim) for kind in NODE_TYPES})
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        # Mean and max pooling for each node type.
        self.head = nn.Linear(2 * len(NODE_TYPES) * hidden_dim, out_dim)

    def node_embeddings(self, batch: Batch) -> dict[str, Tensor]:
        h = self.encoder(batch)
        degrees = node_degrees(batch)
        h = {kind: h[kind] + self.degree(torch.log1p(degrees[kind].float())[:, None]) for kind in h}
        edge_index = {et: batch[et].edge_index for et in all_edge_types()}
        for conv, norms in zip(self.convs, self.norms, strict=True):
            out = conv(h, edge_index)
            h = {
                kind: norms[kind](h[kind] + self.dropout(torch.relu(out[kind])))
                if kind in out
                else h[kind]
                for kind in h
            }
        return h

    def forward(self, batch: Batch) -> Tensor:
        h = self.node_embeddings(batch)
        size = batch.num_graphs
        pooled = []
        for kind in NODE_TYPES:
            index = batch[kind].batch
            pooled.append(global_mean_pool(h[kind], index, size=size))
            pooled.append(global_max_pool(h[kind], index, size=size))
        return self.head(torch.cat(pooled, dim=-1))


def build_spatial_stream(config: ModelConfig) -> SpatialTopologyStream:
    return SpatialTopologyStream(
        hidden_dim=config.hidden_dim,
        out_dim=config.spatial_dim,
        num_layers=config.spatial_layers,
        heads=config.spatial_heads,
        dropout=config.dropout,
    )
