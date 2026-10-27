import copy

import pytest
import torch
from torch_geometric.data import HeteroData

from cadgnn.config import ConfigError, ModelConfig
from cadgnn.graph import (
    EDGE_TYPES,
    InfrastructureGraph,
    all_edge_types,
    build_episode_graphs,
    clean_events,
)
from cadgnn.graph.schema import reverse
from cadgnn.models import DualStreamModel, collate, seed_everything
from cadgnn.spatial import SpatialTopologyStream, build_spatial_stream, node_degrees
from cadgnn.temporal import build_temporal_stream

FEATURES = torch.tensor(
    [[0.6, 0.55, 0.7, 1.1], [0.8, 0.6, 0.75, 0.9], [0.5, 0.7, 0.72, 1.4], [0.9, 0.65, 0.68, 0.5]]
)


def episode(networks, storages, x=FEATURES):
    """Hand-built episode graph; networks[i] and storages[i] are VM i's resources."""
    n = len(networks)
    g = HeteroData()
    g["vm"].x = x[:n].clone()
    g["vm"].timestamp_hours = torch.arange(n, dtype=torch.float32)
    g["vm"].rel_time_hours = torch.arange(n, dtype=torch.float32)
    g["vm"].event_order = torch.arange(n)
    g["vm"].anomaly_score = torch.full((n,), 0.3)
    sizes = {"vm": n, "backup_image": n, "storage": max(storages) + 1, "network": max(networks) + 1}
    for kind, size in sizes.items():
        g[kind].num_nodes = size
        g[kind].global_index = torch.arange(size)
    vms = torch.arange(n)
    edges = {
        ("vm", "has_backup", "backup_image"): torch.stack([vms, vms]),
        ("backup_image", "stored_on", "storage"): torch.stack([vms, torch.tensor(storages)]),
        ("vm", "attached_to", "network"): torch.stack([vms, torch.tensor(networks)]),
    }
    for et, edge_index in edges.items():
        g[et].edge_index = edge_index
        g[reverse(et)].edge_index = edge_index.flip(0)
    g.y = torch.tensor([0])
    g.episode_id = "hand_built"
    return g


SHARED = episode(networks=[0, 0, 0, 0], storages=[0, 1, 2, 3])
SEPARATE = episode(networks=[0, 1, 2, 3], storages=[0, 1, 2, 3])


def make_stream(**kwargs):
    seed_everything(0)
    return SpatialTopologyStream(hidden_dim=16, out_dim=8, **kwargs).eval()


@pytest.fixture(scope="module")
def graphs(small_dataset):
    infra = InfrastructureGraph.from_networkx(small_dataset.graph)
    return build_episode_graphs(infra, clean_events(small_dataset.events, infra))


def test_node_degrees():
    degrees = node_degrees(collate([SHARED, SEPARATE]))
    assert degrees["network"].tolist() == [4, 1, 1, 1, 1]
    assert degrees["vm"].tolist() == [2] * 8
    assert degrees["backup_image"].tolist() == [2] * 8
    assert degrees["storage"].tolist() == [1] * 8


def test_output_shape(graphs):
    out = make_stream()(collate(graphs[:6]))
    assert out.shape == (6, 8)
    assert torch.isfinite(out).all()


def test_shared_network_is_distinguishable():
    stream = make_stream()
    out = stream(collate([SHARED, SEPARATE]))
    assert not torch.allclose(out[0], out[1], atol=1e-4)


def test_two_layers_let_vms_see_each_other():
    stream = make_stream(num_layers=2)
    x = FEATURES.clone()
    x[3] += 5.0
    changed = episode(networks=[0, 0, 0, 0], storages=[0, 1, 2, 3], x=x)
    before = stream.node_embeddings(collate([SHARED]))["vm"][0]
    after = stream.node_embeddings(collate([changed]))["vm"][0]
    # VM 3's telemetry reaches VM 0 through their shared network.
    assert not torch.allclose(before, after)

    separate_changed = episode(networks=[0, 1, 2, 3], storages=[0, 1, 2, 3], x=x)
    before = stream.node_embeddings(collate([SEPARATE]))["vm"][0]
    after = stream.node_embeddings(collate([separate_changed]))["vm"][0]
    # With nothing shared, VM 0 is unaffected.
    assert torch.allclose(before, after)


def test_relabelling_resource_nodes_does_not_matter():
    stream = make_stream()
    a = episode(networks=[0, 0, 1, 1], storages=[0, 1, 1, 2])
    b = episode(networks=[1, 1, 0, 0], storages=[2, 0, 0, 1])
    assert torch.allclose(stream(collate([a])), stream(collate([b])), atol=1e-6)


def test_inputs_outside_the_spatial_view_are_ignored(graphs):
    stream = make_stream()
    g = graphs[0]
    other = copy.copy(g)
    other["vm"].rel_time_hours = g["vm"].rel_time_hours * 50
    other["vm"].timestamp_hours = g["vm"].timestamp_hours + 10
    other.y = 1 - g.y
    other.episode_id = "renamed"
    for kind in other.node_types:
        other[kind].global_index = g[kind].global_index + 1000
    assert torch.equal(stream(collate([g])), stream(collate([other])))


def test_batching_does_not_mix_graphs(graphs):
    stream = make_stream()
    alone = stream(collate([graphs[2]]))
    together = stream(collate(graphs[:5]))[2:3]
    assert torch.allclose(alone, together, atol=1e-6)


def test_missing_relations_in_a_graph():
    g = episode(networks=[0, 1], storages=[0, 0])
    for et in (("backup_image", "stored_on", "storage"),):
        g[et].edge_index = torch.empty(2, 0, dtype=torch.long)
        g[reverse(et)].edge_index = torch.empty(2, 0, dtype=torch.long)
    out = make_stream()(collate([g, SHARED]))
    assert torch.isfinite(out).all()


def test_gradients_reach_every_parameter(graphs):
    stream = SpatialTopologyStream(hidden_dim=16, out_dim=8, num_layers=2, heads=2, dropout=0.1)
    stream(collate(graphs)).sum().backward()
    for name, param in stream.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name


def test_one_attention_layer_per_edge_type():
    stream = make_stream(num_layers=3)
    assert len(stream.convs) == 3
    for conv in stream.convs:
        assert set(conv.convs.keys()) == set(all_edge_types())
    assert len(all_edge_types()) == 2 * len(EDGE_TYPES)


def test_reproducible_with_seed(graphs):
    batch = collate(graphs[:4])
    assert torch.equal(make_stream()(batch), make_stream()(batch))


def test_build_from_config():
    config = ModelConfig(hidden_dim=32, spatial_dim=20, spatial_layers=3, spatial_heads=4)
    stream = build_spatial_stream(config)
    assert stream.out_dim == 20
    assert len(stream.convs) == 3
    gat = next(iter(stream.convs[0].convs.values()))
    assert gat.heads == 4
    for kwargs in ({"spatial_layers": 0}, {"spatial_heads": 0}):
        with pytest.raises(ConfigError):
            ModelConfig(**kwargs)


def test_full_dual_stream_model(graphs):
    config = ModelConfig(hidden_dim=16, temporal_dim=8, spatial_dim=12, dropout=0.0)
    seed_everything(3)
    model = DualStreamModel(config, build_temporal_stream(config), build_spatial_stream(config))
    model.fit_scaler(graphs[:12])
    out = model(collate(graphs))
    assert out.spatial.shape == (len(graphs), 12)
    assert out.temporal.shape == (len(graphs), 8)
    (out.temporal.sum() + out.spatial.sum()).backward()
    assert all(p.grad is not None for p in model.parameters())
