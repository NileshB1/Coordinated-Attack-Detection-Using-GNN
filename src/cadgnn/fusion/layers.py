"""Ways of combining the temporal and spatial embeddings into one vector.

The streams look at different things (timing and telemetry vs. infrastructure
structure) and their embeddings live in different spaces, so each one is first
projected to the fusion size. GatedFusion then learns, per episode and per
dimension, how much to take from each; ConcatFusion is the plain baseline.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from cadgnn.config import ModelConfig


class GatedFusion(nn.Module):
    """Gated multimodal unit (Arevalo et al., 2017).

    h = z * tanh(W_t t) + (1 - z) * tanh(W_s s), with z = sigmoid(W_z [t; s]).
    The mean of z over the dimensions is returned as the temporal share, so a
    report can say which stream an episode's decision leaned on.
    """

    def __init__(self, temporal_dim: int, spatial_dim: int, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.temporal = nn.Linear(temporal_dim, out_dim)
        self.spatial = nn.Linear(spatial_dim, out_dim)
        self.gate = nn.Linear(temporal_dim + spatial_dim, out_dim)

    def forward(self, temporal: Tensor, spatial: Tensor) -> tuple[Tensor, Tensor]:
        _check_pair(temporal, spatial)
        z = torch.sigmoid(self.gate(torch.cat([temporal, spatial], dim=-1)))
        h = z * torch.tanh(self.temporal(temporal)) + (1 - z) * torch.tanh(self.spatial(spatial))
        return h, z.mean(dim=-1)


class ConcatFusion(nn.Module):
    def __init__(self, temporal_dim: int, spatial_dim: int, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.project = nn.Linear(temporal_dim + spatial_dim, out_dim)

    def forward(self, temporal: Tensor, spatial: Tensor) -> tuple[Tensor, None]:
        _check_pair(temporal, spatial)
        return torch.relu(self.project(torch.cat([temporal, spatial], dim=-1))), None


class SingleStreamProjection(nn.Module):
    """Used when one stream is switched off, for the single-stream baselines."""

    def __init__(self, in_dim: int, out_dim: int, stream: str) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.stream = stream
        self.project = nn.Linear(in_dim, out_dim)

    def forward(self, temporal: Tensor | None, spatial: Tensor | None) -> tuple[Tensor, None]:
        x = temporal if self.stream == "temporal" else spatial
        other = spatial if self.stream == "temporal" else temporal
        if x is None or other is not None:
            raise ValueError(f"expected only the {self.stream} embedding")
        return torch.relu(self.project(x)), None


def build_fusion(config: ModelConfig) -> nn.Module:
    if config.use_temporal and config.use_spatial:
        fusion_cls = GatedFusion if config.fusion == "gated" else ConcatFusion
        return fusion_cls(config.temporal_dim, config.spatial_dim, config.fusion_dim)
    if config.use_temporal:
        return SingleStreamProjection(config.temporal_dim, config.fusion_dim, "temporal")
    return SingleStreamProjection(config.spatial_dim, config.fusion_dim, "spatial")


def _check_pair(temporal: Tensor | None, spatial: Tensor | None) -> None:
    if temporal is None or spatial is None:
        raise ValueError("both stream embeddings are needed for this fusion method")
    if temporal.size(0) != spatial.size(0):
        raise ValueError(
            f"streams disagree on the number of episodes: {temporal.size(0)} vs {spatial.size(0)}"
        )
