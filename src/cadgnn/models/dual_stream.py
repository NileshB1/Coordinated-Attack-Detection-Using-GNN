"""Container that runs the temporal and spatial streams side by side.

A stream is any ``nn.Module`` with an ``out_dim`` attribute whose forward takes
a batch of episode graphs (VM features already scaled) and returns a tensor of
shape [num_graphs, out_dim]. The two outputs are returned separately; combining
them is the job of the fusion layer.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, HeteroData

from cadgnn.config import ModelConfig
from cadgnn.models.inputs import FeatureScaler, ModelInputError, check_batch


class StreamOutputs(NamedTuple):
    temporal: Tensor
    spatial: Tensor


class DualStreamModel(nn.Module):
    def __init__(self, config: ModelConfig, temporal: nn.Module, spatial: nn.Module) -> None:
        super().__init__()
        for name, stream, expected in (
            ("temporal", temporal, config.temporal_dim),
            ("spatial", spatial, config.spatial_dim),
        ):
            out_dim = getattr(stream, "out_dim", None)
            if out_dim != expected:
                raise ModelInputError(
                    f"{name} stream out_dim is {out_dim}, config expects {expected}"
                )
        self.config = config
        self.scaler = FeatureScaler()
        self.temporal = temporal
        self.spatial = spatial

    def fit_scaler(self, train_graphs: Sequence[HeteroData]) -> DualStreamModel:
        self.scaler.fit(train_graphs)
        return self

    def forward(self, batch: Batch) -> StreamOutputs:
        check_batch(batch)
        # Shallow copy so the caller's batch keeps its raw features.
        scaled = copy.copy(batch)
        scaled["vm"].x = self.scaler(batch["vm"].x)

        temporal = self.temporal(scaled)
        spatial = self.spatial(scaled)
        _check_output(temporal, "temporal", batch.num_graphs, self.config.temporal_dim)
        _check_output(spatial, "spatial", batch.num_graphs, self.config.spatial_dim)
        return StreamOutputs(temporal, spatial)


def _check_output(out: Tensor, name: str, num_graphs: int, dim: int) -> None:
    if not isinstance(out, Tensor) or tuple(out.shape) != (num_graphs, dim):
        shape = tuple(out.shape) if isinstance(out, Tensor) else type(out).__name__
        raise ModelInputError(f"{name} stream returned {shape}, expected ({num_graphs}, {dim})")
    if not torch.isfinite(out).all():
        raise ModelInputError(f"{name} stream returned non-finite values")
