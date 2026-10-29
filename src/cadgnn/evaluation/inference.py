from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from cadgnn.fusion import CoordinationModel


@dataclass
class Predictions:
    episode_ids: list[str]
    y: np.ndarray  # true labels, 1 = coordinated
    scores: np.ndarray  # uncalibrated sigmoid(logit)
    gates: np.ndarray | None  # temporal share per episode, gated fusion only


@torch.no_grad()
def predict(
    model: CoordinationModel,
    graphs: Sequence[HeteroData],
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> Predictions:
    if not graphs:
        raise ValueError("no graphs to predict on")
    model.eval()
    model.to(device)
    ids, ys, scores, gates = [], [], [], []
    for batch in DataLoader(list(graphs), batch_size=batch_size):
        batch = batch.to(device)
        out = model(batch)
        ids += list(batch.episode_id)
        ys.append(batch.y.cpu().numpy())
        scores.append(torch.sigmoid(out.logit).cpu().numpy())
        if out.gate is not None:
            gates.append(out.gate.cpu().numpy())
    return Predictions(
        episode_ids=ids,
        y=np.concatenate(ys),
        scores=np.concatenate(scores),
        gates=np.concatenate(gates) if gates else None,
    )
