"""Training loop for the coordination model, with early stopping on validation loss."""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from cadgnn.config import TrainingConfig
from cadgnn.fusion import CoordinationModel, coordination_loss, positive_weight

log = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    best_epoch: int
    best_val_loss: float
    pos_weight: float
    history: list[dict[str, float]] = field(default_factory=list)


def train_model(
    model: CoordinationModel,
    train: Sequence[HeteroData],
    val: Sequence[HeteroData],
    config: TrainingConfig,
    seed: int,
    device: torch.device | str = "cpu",
) -> TrainingResult:
    """Fit the scaler and class weight on ``train``, then train with Adam.

    The weights from the epoch with the lowest validation loss are restored at
    the end. ``val`` is only used for that choice, never for gradient updates.
    """
    if not train or not val:
        raise ValueError("training and validation sets must not be empty")
    model.fit_scaler(train)
    pos_weight = positive_weight(train)
    model.to(device)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        list(train), batch_size=config.batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(list(val), batch_size=config.batch_size)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_loss, best_epoch, best_state = float("inf"), 0, copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            loss = coordination_loss(model(batch).logit, batch.y, pos_weight)
            loss.backward()
            optimiser.step()
            train_losses.append(loss.item())

        val_loss = evaluate_loss(model, val_loader, pos_weight, device)
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss}
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            log.info("stopping early at epoch %d, best was %d", epoch, best_epoch)
            break

    model.load_state_dict(best_state)
    log.info("best validation loss %.4f at epoch %d", best_loss, best_epoch)
    return TrainingResult(best_epoch, best_loss, float(pos_weight), history)


@torch.no_grad()
def evaluate_loss(
    model: CoordinationModel, loader: DataLoader, pos_weight: torch.Tensor, device
) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        loss = coordination_loss(model(batch).logit, batch.y, pos_weight)
        total += loss.item() * batch.num_graphs
        count += batch.num_graphs
    return total / count
