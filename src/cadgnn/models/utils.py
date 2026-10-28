from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from cadgnn import __version__
from cadgnn.models.inputs import ModelInputError


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ModelInputError(f"device {name!r} requested but CUDA is not available")
    return device


def save_checkpoint(model: nn.Module, path: str | Path, **extra: Any) -> None:
    """Save weights, scaler statistics and model config. ``extra`` must hold plain values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": asdict(model.config),
            "state_dict": model.state_dict(),
            "package_version": __version__,
            **extra,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module, path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ModelInputError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if checkpoint.get("model_config") != asdict(model.config):
        raise ModelInputError(
            f"checkpoint was saved with model config {checkpoint.get('model_config')}, "
            f"current model uses {asdict(model.config)}"
        )
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint
