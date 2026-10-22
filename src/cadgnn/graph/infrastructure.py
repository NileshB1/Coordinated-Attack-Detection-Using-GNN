"""Typed, indexed view of the infrastructure graph.

Each node type gets its own 0-based index, in the order the nodes appear in the
input. Edges are kept per edge type as (2, E) arrays of those indices, sorted so
the result does not depend on the row order of the edge table
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from cadgnn.graph.schema import (
    EDGE_TYPES,
    NODE_TYPES,
    EdgeType,
    GraphConstructionError,
    edge_type_between,
    reverse,
)

log = logging.getLogger(__name__)


@dataclass
class InfrastructureGraph:
    node_ids: dict[str, list[str]]
    edges: dict[EdgeType, np.ndarray]

    def __post_init__(self) -> None:
        self._locate = {
            node_id: (kind, i)
            for kind, ids in self.node_ids.items()
            for i, node_id in enumerate(ids)
        }

    @classmethod
    def from_frames(cls, nodes: pd.DataFrame, edges: pd.DataFrame) -> InfrastructureGraph:
        _require_columns(nodes, ["node_id", "kind"], "node table")
        _require_columns(edges, ["source", "target"], "edge table")
        node_ids = _index_nodes(nodes)
        kind_of = {n: kind for kind, ids in node_ids.items() for n in ids}
        return cls(node_ids, _index_edges(edges, node_ids, kind_of))

    @classmethod
    def from_networkx(cls, graph: nx.Graph) -> InfrastructureGraph:
        """Build from the generator's networkx graph (nodes carry a ``kind`` attribute)."""
        nodes = pd.DataFrame(
            [{"node_id": n, "kind": d.get("kind")} for n, d in graph.nodes(data=True)],
            columns=["node_id", "kind"],
        )
        edges = pd.DataFrame(list(graph.edges()), columns=["source", "target"])
        return cls.from_frames(nodes, edges)

    def num_nodes(self, kind: str) -> int:
        return len(self.node_ids[kind])

    def locate(self, node_id: str) -> tuple[str, int]:
        """Node type and per-type index of a raw node id."""
        try:
            return self._locate[node_id]
        except KeyError:
            raise GraphConstructionError(f"unknown node {node_id!r}") from None

    def neighbourhood(self, vm_index: np.ndarray) -> dict[str, np.ndarray]:
        """The given VMs plus the images, storage pools and networks they use.

        Other VMs are not pulled in, even if they share a network or storage pool.
        The VM order is kept as given; the other types come back sorted.
        """
        selected = {kind: np.empty(0, dtype=np.int64) for kind in NODE_TYPES}
        selected["vm"] = np.asarray(vm_index, dtype=np.int64)
        # Edges point away from the VM and EDGE_TYPES lists vm -> image before
        # image -> storage, so one pass in that order reaches everything.
        for et in EDGE_TYPES:
            src, _, dst = et
            index = self.edges[et]
            found = index[1, np.isin(index[0], selected[src])]
            selected[dst] = np.union1d(selected[dst], found)
        return selected

    def subgraph_edges(self, selected: dict[str, np.ndarray]) -> dict[EdgeType, np.ndarray]:
        """Edges among the selected nodes, re-indexed to positions within ``selected``."""
        local = {}
        for kind, idx in selected.items():
            lookup = np.full(self.num_nodes(kind), -1, dtype=np.int64)
            lookup[idx] = np.arange(len(idx))
            local[kind] = lookup

        result = {}
        for et in EDGE_TYPES:
            src, _, dst = et
            index = self.edges[et]
            s, d = local[src][index[0]], local[dst][index[1]]
            keep = (s >= 0) & (d >= 0)
            result[et] = np.stack([s[keep], d[keep]])
        return result

    def to_hetero_data(self) -> HeteroData:
        """Structure of the whole infrastructure as HeteroData, without features."""
        data = HeteroData()
        for kind in NODE_TYPES:
            data[kind].num_nodes = self.num_nodes(kind)
        for et, index in self.edges.items():
            edge_index = torch.from_numpy(index)
            data[et].edge_index = edge_index
            data[reverse(et)].edge_index = edge_index.flip(0)
        return data


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise GraphConstructionError(f"{name} is missing column(s): {', '.join(missing)}")


def _index_nodes(nodes: pd.DataFrame) -> dict[str, list[str]]:
    if nodes[["node_id", "kind"]].isna().any().any():
        raise GraphConstructionError("node table has rows with a missing node_id or kind")
    nodes = nodes.astype({"node_id": str, "kind": str})

    unknown = sorted(set(nodes["kind"]) - set(NODE_TYPES))
    if unknown:
        raise GraphConstructionError(f"unknown node kind(s): {', '.join(unknown)}")

    kinds_per_id = nodes.groupby("node_id")["kind"].nunique()
    conflicting = kinds_per_id[kinds_per_id > 1].index.tolist()
    if conflicting:
        raise GraphConstructionError(
            f"node(s) listed with more than one kind: {', '.join(conflicting[:5])}"
        )
    duplicates = int(nodes.duplicated("node_id").sum())
    if duplicates:
        log.warning("dropping %d duplicate node row(s)", duplicates)
        nodes = nodes.drop_duplicates("node_id")

    return {kind: nodes.loc[nodes["kind"] == kind, "node_id"].tolist() for kind in NODE_TYPES}


def _index_edges(
    edges: pd.DataFrame, node_ids: dict[str, list[str]], kind_of: dict[str, str]
) -> dict[EdgeType, np.ndarray]:
    if edges[["source", "target"]].isna().any().any():
        raise GraphConstructionError("edge table has rows with a missing source or target")
    edges = edges.astype({"source": str, "target": str})

    unknown = sorted((set(edges["source"]) | set(edges["target"])) - set(kind_of))
    if unknown:
        raise GraphConstructionError(
            f"{len(unknown)} edge endpoint(s) are not in the node table, e.g. {unknown[:5]}"
        )
    loops = edges["source"] == edges["target"]
    if loops.any():
        raise GraphConstructionError(f"{int(loops.sum())} self-loop edge(s) found")

    position = {kind: {n: i for i, n in enumerate(ids)} for kind, ids in node_ids.items()}
    pairs: dict[EdgeType, set[tuple[int, int]]] = {et: set() for et in EDGE_TYPES}
    duplicates = 0
    for source, target in edges[["source", "target"]].itertuples(index=False):
        match = edge_type_between(kind_of[source], kind_of[target])
        if match is None:
            raise GraphConstructionError(
                f"no relation between {kind_of[source]} and {kind_of[target]} "
                f"(edge {source} - {target})"
            )
        et, swap = match
        if swap:
            source, target = target, source
        pair = (position[et[0]][source], position[et[2]][target])
        if pair in pairs[et]:
            duplicates += 1
        pairs[et].add(pair)
    if duplicates:
        log.warning("dropping %d duplicate edge(s)", duplicates)

    return {
        et: np.array(sorted(found), dtype=np.int64).reshape(-1, 2).T for et, found in pairs.items()
    }
