"""Benign, coincidental and coordinated backup events.

Port of sections 5 and 6 of notebooks/random_anomaly_generator_simple.ipynb. All
randomness comes from one numpy Generator and draws happen in the notebook's
order, so the same seed and scenario reproduce the notebook's graph and events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from cadgnn.config import GeneratorConfig
from cadgnn.data.calibration import ParamProfile, weighted_score
from cadgnn.generator.infrastructure import build_infrastructure, find_attack_chain, vm_nodes

log = logging.getLogger(__name__)

LABELS = ("none", "coincidental", "coordinated")
EVENT_COLUMNS = [
    "vm", "timestamp_hours",
    "change_rate", "file_entropy", "backup_size_gb",
    "file_count", "anomaly_score",
    "state", "episode_id",
    "label",
]


@dataclass
class SyntheticDataset:
    graph: nx.Graph
    events: pd.DataFrame
    config: GeneratorConfig
    seed: int


def make_event(
    vm: str,
    timestamp_hours: float,
    state: str,
    episode_id: str | None,
    label: str,
    profile: ParamProfile,
    rng: np.random.Generator,
) -> dict:
    values = {name: float(rng.uniform(*feature.bounds(state))) for name, feature in profile.items()}
    # The score uses unrounded values; only the stored values are rounded.
    score = weighted_score(profile, values)
    return {
        "vm": vm,  "timestamp_hours": round(timestamp_hours, 3),
        **{name: round(value, 4) for name, value in values.items()},
        "anomaly_score": round(score, 4),
        "state": state,  "episode_id": episode_id,
        "label": label,
    }


def generate_events(
    graph: nx.Graph, profile: ParamProfile, config: GeneratorConfig, rng: np.random.Generator
) -> pd.DataFrame:
    vms = vm_nodes(graph)
    if not vms:
        raise ValueError("graph has no VM nodes")
    if config.n_coincidental_episodes and config.coincidental_vms[1] > len(vms):
        raise ValueError(
            f"coincidental episodes need up to {config.coincidental_vms[1]} VMs, "
            f"graph has {len(vms)}"
        )
    vm_array = np.array(vms)
    hours = config.observation_hours
    events = []

    for vm in vms:
        for t in rng.uniform(0, hours, size=config.benign_backups_per_vm):
            events.append(make_event(vm, t, "benign", None, "none", profile, rng))

    low, high = config.coincidental_vms
    for ep in range(config.n_coincidental_episodes):
        episode_id = f"coincidental_{ep}"
        size = int(rng.integers(low, high + 1))
        for vm in rng.choice(vm_array, size=size, replace=False):
            t = float(rng.uniform(0, hours))
            events.append(
                make_event(str(vm), t, "infected", episode_id, "coincidental", profile, rng)
            )

    low, high = config.coordinated_chain_length
    delay_low, delay_high = config.hop_delay_minutes
    latest_start = hours - config.campaign_start_margin_hours
    short_chains = 0
    for ep in range(config.n_coordinated_episodes):
        episode_id = f"coordinated_{ep}"
        length = int(rng.integers(low, high + 1))
        start_vm = str(rng.choice(vm_array))
        chain = find_attack_chain(graph, start_vm, length, rng)
        short_chains += len(chain) < length
        t = float(rng.uniform(0, latest_start))
        for vm in chain:
            events.append(make_event(vm, t, "infected", episode_id, "coordinated", profile, rng))
            # A delay is drawn after the last VM as well; the notebook does the same and
            # skipping it would shift every later draw.
            t += float(rng.uniform(delay_low, delay_high)) / 60.0
    if short_chains:
        log.warning(
            "%d of %d coordinated chains ended early for lack of unvisited neighbours",
            short_chains,
            config.n_coordinated_episodes,
        )

    df = pd.DataFrame(events, columns=EVENT_COLUMNS)
    return df.sort_values("timestamp_hours").reset_index(drop=True)


def generate(config: GeneratorConfig, profile: ParamProfile, seed: int) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    graph = build_infrastructure(config.n_vms, config.n_storage, config.n_network, rng)
    events = generate_events(graph, profile, config, rng)
    log.info(
        "generated %d events on %d nodes / %d edges (seed %d)",
        len(events),
        graph.number_of_nodes(),
        graph.number_of_edges(), seed
    )
    return SyntheticDataset(graph=graph, events=events, config=config, seed=seed)
