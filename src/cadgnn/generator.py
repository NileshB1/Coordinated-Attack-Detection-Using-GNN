"""Random anomaly generator for synthetic VM backup telemetry.

Port of sections 5, 6 and 8 of notebooks/random_anomaly_generator_simple.ipynb.
All randomness comes from one ``numpy.random.Generator`` and the draws are made
in the same order as in the notebook, so the same seed and scenario reproduce
the notebook's graph and events exactly.

Events carry one of three labels:

* ``none``: routine backups, ``benign_backups_per_vm`` per VM at uniform times.
* ``coincidental``: episodes of randomly chosen VMs with infected-range
  telemetry at independent uniform times (no temporal or structural link).
* ``coordinated``: campaigns that move from VM to VM through a shared network
  fabric, a few minutes apart, with infected-range telemetry.

Coincidental and coordinated events are drawn from the same feature ranges; only
their timing and placement on the graph differ. These labels are properties of
the synthetic scenario, not observations of real attacks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from cadgnn import __version__
from cadgnn.calibration import Calibration, ParamProfile, weighted_score
from cadgnn.config import GeneratorConfig

log = logging.getLogger(__name__)

NODE_KINDS = ("vm", "backup_image", "storage", "network")
LABELS = ("none", "coincidental", "coordinated")
EVENT_COLUMNS = [
    "vm",
    "timestamp_hours",
    "change_rate",
    "file_entropy",
    "backup_size_gb",
    "file_count",
    "anomaly_score",
    "state",
    "episode_id",
    "label",
]

TELEMETRY_FILE = "synthetic_backup_telemetry.csv"
NODES_FILE = "infrastructure_graph_nodes.csv"
EDGES_FILE = "infrastructure_graph_edges.csv"
MANIFEST_FILE = "experiment_manifest.json"


@dataclass
class SyntheticDataset:
    graph: nx.Graph
    events: pd.DataFrame
    config: GeneratorConfig
    seed: int


def build_infrastructure(
    n_vms: int, n_storage: int, n_network: int, rng: np.random.Generator
) -> nx.Graph:
    """Each VM has its own backup image; each image sits on one random storage pool
    and each VM on one random network fabric."""
    for name, value in (("n_vms", n_vms), ("n_storage", n_storage), ("n_network", n_network)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    g = nx.Graph()
    for i in range(n_vms):
        g.add_node(f"vm_{i}", kind="vm")
        g.add_node(f"image_{i}", kind="backup_image")
        g.add_edge(f"vm_{i}", f"image_{i}")
    for i in range(n_storage):
        g.add_node(f"storage_{i}", kind="storage")
    for i in range(n_network):
        g.add_node(f"network_{i}", kind="network")
    for i in range(n_vms):
        storage = f"storage_{rng.integers(0, n_storage)}"
        network = f"network_{rng.integers(0, n_network)}"
        g.add_edge(f"image_{i}", storage)
        g.add_edge(f"vm_{i}", network)
    return g


def vm_nodes(graph: nx.Graph) -> list[str]:
    # Sorted as strings (vm_0, vm_1, vm_10, ...), as in the notebook; the order
    # determines which random draws each VM receives.
    return sorted(n for n, d in graph.nodes(data=True) if d["kind"] == "vm")


def find_attack_chain(
    graph: nx.Graph, start_vm: str, n_vms: int, rng: np.random.Generator
) -> list[str]:
    """Random walk over unvisited VMs that are two edges apart.

    With the graph from build_infrastructure, the only VM-to-VM paths of length
    two pass through a shared network fabric. The chain stops early when the
    current VM has no unvisited neighbour.
    """
    if n_vms < 1:
        raise ValueError(f"chain length must be at least 1, got {n_vms}")
    chain = [start_vm]
    visited = {start_vm}
    current = start_vm
    for _ in range(n_vms - 1):
        candidates = {
            n2
            for n1 in graph.neighbors(current)
            for n2 in graph.neighbors(n1)
            if graph.nodes[n2]["kind"] == "vm" and n2 not in visited
        }
        if not candidates:
            break
        current = str(rng.choice(sorted(candidates)))
        chain.append(current)
        visited.add(current)
    return chain


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
        "vm": vm,
        "timestamp_hours": round(timestamp_hours, 3),
        **{name: round(value, 4) for name, value in values.items()},
        "anomaly_score": round(score, 4),
        "state": state,
        "episode_id": episode_id,
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
        graph.number_of_edges(),
        seed,
    )
    return SyntheticDataset(graph=graph, events=events, config=config, seed=seed)


def build_manifest(dataset: SyntheticDataset, calibration: Calibration) -> dict:
    graph, events, config = dataset.graph, dataset.events, dataset.config
    profile = calibration.profile
    return {
        "random_seed": dataset.seed,
        "num_vms": config.n_vms,
        "num_storage": config.n_storage,
        "num_network": config.n_network,
        "total_graph_nodes": graph.number_of_nodes(),
        "total_graph_edges": graph.number_of_edges(),
        "num_features": len(profile),
        "features": list(profile),
        "feature_weights": {name: f.weight for name, f in profile.items()},
        "total_events": len(events),
        "label_counts": {k: int(v) for k, v in events["label"].value_counts().items()},
        "calibration_sources": {
            "ransap": "Real ransomware storage write & Shannon entropy traces",
            "napierone": "Ransomware vs normal file size encryption ratios",
            "filesystems": (
                f"FSL Filesystems Archive ({len(calibration.fsl)} snapshot trace files found, "
                f"max file_count={calibration.fsl_file_count})"
            ),
        },
        "generator": asdict(config),
        "param_profile": {name: asdict(f) for name, f in profile.items()},
        "calibration": {
            "ransap": calibration.ransap.to_dict(orient="records"),
            "napierone_normal_avg_kb": calibration.napierone_normal_kb,
            "napierone_ransomware_avg_kb": calibration.napierone_ransomware_kb,
            "size_ratio": calibration.size_ratio,
            "fsl_file_count": calibration.fsl_file_count,
            "fallbacks": calibration.fallbacks,
        },
        "package_version": __version__,
    }


def save_outputs(
    dataset: SyntheticDataset, calibration: Calibration, output_dir: str | Path
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "telemetry": output_dir / TELEMETRY_FILE,
        "nodes": output_dir / NODES_FILE,
        "edges": output_dir / EDGES_FILE,
        "manifest": output_dir / MANIFEST_FILE,
    }

    dataset.events.to_csv(paths["telemetry"], index=False)
    nodes = [{"node_id": n, "kind": d["kind"]} for n, d in dataset.graph.nodes(data=True)]
    pd.DataFrame(nodes, columns=["node_id", "kind"]).to_csv(paths["nodes"], index=False)
    edges = [{"source": u, "target": v} for u, v in dataset.graph.edges()]
    pd.DataFrame(edges, columns=["source", "target"]).to_csv(paths["edges"], index=False)
    with paths["manifest"].open("w", encoding="utf-8") as f:
        json.dump(build_manifest(dataset, calibration), f, indent=2)
    return paths
