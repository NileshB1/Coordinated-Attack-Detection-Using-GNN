"""Batching, input checks and feature scaling for episode graphs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, HeteroData

from cadgnn.graph import NODE_TYPES, VM_FEATURES, all_edge_types

VM_ATTRIBUTES = ("rel_time_hours", "event_order")


class ModelInputError(ValueError):
    pass


def collate(graphs: Sequence[HeteroData]) -> Batch:
    if not graphs:
        raise ModelInputError("no graphs to batch")
    return Batch.from_data_list(list(graphs))


def check_batch(batch: Batch, n_features: int = len(VM_FEATURES)) -> None:
    """Raise ModelInputError if the batch does not look like collated episode graphs."""
    if not isinstance(batch, Batch):
        raise ModelInputError("expected a Batch, use collate() or a PyG DataLoader")

    missing = [kind for kind in NODE_TYPES if kind not in batch.node_types]
    if missing:
        raise ModelInputError(f"batch is missing node type(s): {', '.join(missing)}")

    vm = batch["vm"]
    x = getattr(vm, "x", None)
    if not isinstance(x, Tensor) or x.dim() != 2 or x.size(1) != n_features:
        shape = tuple(x.shape) if isinstance(x, Tensor) else None
        raise ModelInputError(f"vm.x must have shape [n, {n_features}], got {shape}")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ModelInputError("vm.x must be finite floating point values")
    for name in VM_ATTRIBUTES:
        value = getattr(vm, name, None)
        if not isinstance(value, Tensor) or value.size(0) != x.size(0):
            raise ModelInputError(f"vm.{name} is missing or does not match the number of VMs")
    for kind in NODE_TYPES:
        if getattr(batch[kind], "batch", None) is None:
            raise ModelInputError(f"{kind} nodes have no batch vector")

    for et in all_edge_types():
        edge_index = getattr(batch[et], "edge_index", None) if et in batch.edge_types else None
        if edge_index is None:
            raise ModelInputError(f"batch is missing edge type {et}")
        if edge_index.dtype != torch.long or edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ModelInputError(f"{et} edge_index must be a long tensor of shape [2, E]")
        if edge_index.numel():
            src, _, dst = et
            too_big = edge_index.max(dim=1).values >= torch.tensor(
                [batch[src].num_nodes, batch[dst].num_nodes], device=edge_index.device
            )
            if edge_index.min() < 0 or too_big.any():
                raise ModelInputError(f"{et} edge_index points outside the node range")


class FeatureScaler(nn.Module):
    """Z-scores VM features. Fit it on the training graphs only."""

    def __init__(self, n_features: int = len(VM_FEATURES)) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, graphs: Sequence[HeteroData]) -> FeatureScaler:
        if not graphs:
            raise ModelInputError("cannot fit the scaler on an empty set of graphs")
        x = torch.cat([g["vm"].x for g in graphs]).to(self.mean)
        if x.size(0) == 0:
            raise ModelInputError("cannot fit the scaler: the graphs have no VM nodes")
        std = x.std(dim=0, unbiased=False)
        # A feature that never varies in training is only centred.
        std[std < 1e-8] = 1.0
        self.mean.copy_(x.mean(dim=0))
        self.std.copy_(std)
        self.fitted.fill_(True)
        return self

    def forward(self, x: Tensor) -> Tensor:
        if not bool(self.fitted):
            raise ModelInputError("feature scaler is not fitted, call fit() on the training graphs")
        return (x - self.mean) / self.std
