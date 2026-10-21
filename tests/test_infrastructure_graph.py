import logging

import numpy as np
import pandas as pd
import pytest

from cadgnn.graph import EDGE_TYPES, NODE_TYPES, GraphConstructionError, InfrastructureGraph
from cadgnn.graph.schema import all_edge_types, edge_type_between, reverse


def frames(nodes, edges):
    return (
        pd.DataFrame(nodes, columns=["node_id", "kind"]),
        pd.DataFrame(edges, columns=["source", "target"]),
    )


TINY_NODES = [
    ("vm_0", "vm"), ("vm_1", "vm"),
    ("image_0", "backup_image"), ("image_1", "backup_image"),
    ("storage_0", "storage"), ("storage_1", "storage"),
    ("network_0", "network"),
]  # fmt: skip
TINY_EDGES = [
    ("vm_0", "image_0"), ("vm_1", "image_1"),
    ("image_0", "storage_0"), ("image_1", "storage_0"),
    ("vm_0", "network_0"), ("network_0", "vm_1"),
]  # fmt: skip


def tiny():
    return InfrastructureGraph.from_frames(*frames(TINY_NODES, TINY_EDGES))


def test_schema_relations():
    assert edge_type_between("vm", "network") == (("vm", "attached_to", "network"), False)
    assert edge_type_between("network", "vm") == (("vm", "attached_to", "network"), True)
    assert edge_type_between("vm", "storage") is None
    assert edge_type_between("vm", "vm") is None
    assert reverse(("vm", "has_backup", "backup_image")) == ("backup_image", "backup_of", "vm")
    assert len(set(all_edge_types())) == 2 * len(EDGE_TYPES)


def test_index_per_type_follows_input_order():
    g = tiny()
    assert g.node_ids["vm"] == ["vm_0", "vm_1"]
    assert g.node_ids["storage"] == ["storage_0", "storage_1"]
    assert g.locate("image_1") == ("backup_image", 1)
    with pytest.raises(GraphConstructionError, match="unknown node"):
        g.locate("vm_9")


def test_edges_are_typed_and_oriented_from_the_vm():
    g = tiny()
    # network_0 -> vm_1 was given backwards and is flipped to vm -> network.
    assert g.edges[("vm", "attached_to", "network")].tolist() == [[0, 1], [0, 0]]
    assert g.edges[("backup_image", "stored_on", "storage")].tolist() == [[0, 1], [0, 0]]
    assert g.edges[("vm", "has_backup", "backup_image")].dtype == np.int64


def test_unused_nodes_are_kept():
    # storage_1 has no edges but is still part of the infrastructure.
    assert tiny().num_nodes("storage") == 2


def test_edge_order_does_not_matter():
    a = tiny()
    b = InfrastructureGraph.from_frames(*frames(TINY_NODES, TINY_EDGES[::-1]))
    for et in EDGE_TYPES:
        assert np.array_equal(a.edges[et], b.edges[et])


def test_duplicates_are_dropped(caplog):
    caplog.set_level(logging.WARNING)
    nodes = TINY_NODES + [("vm_0", "vm")]
    edges = TINY_EDGES + [("image_0", "vm_0"), ("vm_0", "image_0")]
    g = InfrastructureGraph.from_frames(*frames(nodes, edges))
    assert g.num_nodes("vm") == 2
    assert g.edges[("vm", "has_backup", "backup_image")].shape == (2, 2)
    assert "duplicate node" in caplog.text
    assert "2 duplicate edge" in caplog.text


@pytest.mark.parametrize(
    "nodes, edges, message",
    [
        (TINY_NODES + [("vm_0", "storage")], TINY_EDGES, "more than one kind"),
        (TINY_NODES + [("router_0", "router")], TINY_EDGES, "unknown node kind"),
        (TINY_NODES + [(None, "vm")], TINY_EDGES, "missing node_id or kind"),
        (TINY_NODES, TINY_EDGES + [("vm_0", "vm_7")], "not in the node table"),
        (TINY_NODES, TINY_EDGES + [("vm_0", "vm_0")], "self-loop"),
        (TINY_NODES, TINY_EDGES + [("vm_0", "vm_1")], "no relation between vm and vm"),
        (TINY_NODES, TINY_EDGES + [("vm_0", "storage_1")], "no relation between vm and storage"),
        (TINY_NODES, TINY_EDGES + [("vm_0", None)], "missing source or target"),
    ],
)
def test_invalid_structure_is_rejected(nodes, edges, message):
    with pytest.raises(GraphConstructionError, match=message):
        InfrastructureGraph.from_frames(*frames(nodes, edges))


def test_missing_columns():
    nodes, edges = frames(TINY_NODES, TINY_EDGES)
    with pytest.raises(GraphConstructionError, match="node table is missing column"):
        InfrastructureGraph.from_frames(nodes.rename(columns={"kind": "type"}), edges)
    with pytest.raises(GraphConstructionError, match="edge table is missing column"):
        InfrastructureGraph.from_frames(nodes, edges.drop(columns="target"))


def test_empty_graph():
    g = InfrastructureGraph.from_frames(*frames([], []))
    assert all(g.num_nodes(k) == 0 for k in NODE_TYPES)
    assert all(g.edges[et].shape == (2, 0) for et in EDGE_TYPES)


def test_from_generator_graph(small_dataset):
    g = InfrastructureGraph.from_networkx(small_dataset.graph)
    assert g.num_nodes("vm") == 60
    assert g.num_nodes("storage") == 4
    assert sum(g.edges[et].shape[1] for et in EDGE_TYPES) == small_dataset.graph.number_of_edges()
    # Every VM has exactly one image and one network.
    for et in (("vm", "has_backup", "backup_image"), ("vm", "attached_to", "network")):
        assert sorted(g.edges[et][0].tolist()) == list(range(60))


def test_neighbourhood_does_not_pull_in_other_vms():
    g = tiny()
    selected = g.neighbourhood(np.array([1]))
    assert selected["vm"].tolist() == [1]
    assert selected["backup_image"].tolist() == [1]
    assert selected["storage"].tolist() == [0]
    # vm_0 shares network_0 but was not selected.
    assert selected["network"].tolist() == [0]


def test_subgraph_edges_are_reindexed():
    g = tiny()
    selected = g.neighbourhood(np.array([1, 0]))
    edges = g.subgraph_edges(selected)
    # vm order is kept as given: local 0 is vm_1, local 1 is vm_0.
    assert edges[("vm", "has_backup", "backup_image")].tolist() == [[1, 0], [0, 1]]
    assert edges[("vm", "attached_to", "network")].tolist() == [[1, 0], [0, 0]]


def test_to_hetero_data():
    data = tiny().to_hetero_data()
    assert data["vm"].num_nodes == 2
    assert set(data.edge_types) == set(all_edge_types())
    forward = data["vm", "attached_to", "network"].edge_index
    assert data["network", "hosts", "vm"].edge_index.tolist() == forward.flip(0).tolist()
