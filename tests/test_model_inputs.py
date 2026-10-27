import copy

import pytest
import torch
from torch_geometric.loader import DataLoader

from cadgnn.graph import InfrastructureGraph, build_episode_graphs, clean_events
from cadgnn.models import FeatureScaler, ModelInputError, NodeEncoder, check_batch, collate


@pytest.fixture(scope="module")
def graphs(small_dataset):
    infra = InfrastructureGraph.from_networkx(small_dataset.graph)
    return build_episode_graphs(infra, clean_events(small_dataset.events, infra))


def test_collate_matches_dataloader(graphs):
    batch = collate(graphs[:5])
    loader_batch = next(iter(DataLoader(graphs[:5], batch_size=5)))
    assert batch.num_graphs == 5
    assert torch.equal(batch["vm"].x, loader_batch["vm"].x)
    check_batch(batch)
    check_batch(loader_batch)


def test_collate_empty():
    with pytest.raises(ModelInputError, match="no graphs"):
        collate([])


def test_single_graph_must_be_batched(graphs):
    with pytest.raises(ModelInputError, match="expected a Batch"):
        check_batch(graphs[0])


def broken(graphs, change):
    batch = copy.copy(collate(graphs[:3]))
    change(batch)
    return batch


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda b: setattr(b["vm"], "x", b["vm"].x[:, :3]), r"shape \[n, 4\]"),
        (lambda b: setattr(b["vm"], "x", b["vm"].x.long()), "floating point"),
        (lambda b: b["vm"].x.__setitem__((0, 0), float("nan")), "finite"),
        (lambda b: delattr(b["vm"], "rel_time_hours"), "rel_time_hours"),
        (lambda b: setattr(b["vm"], "event_order", b["vm"].event_order[:1]), "event_order"),
        (lambda b: delattr(b["network"], "batch"), "network nodes have no batch"),
        (lambda b: setattr(b["vm", "attached_to", "network"], "edge_index",
                           b["vm", "attached_to", "network"].edge_index + 1000), "outside"),
        (lambda b: setattr(b["vm", "attached_to", "network"], "edge_index",
                           b["vm", "attached_to", "network"].edge_index.float()), "long tensor"),
        (lambda b: delattr(b["storage", "stores", "backup_image"], "edge_index"), "edge type"),
    ],
)  # fmt: skip
def test_check_batch_rejects_malformed_input(graphs, change, message):
    batch = broken(graphs, change)
    with pytest.raises(ModelInputError, match=message):
        check_batch(batch)


def test_scaler_uses_training_statistics_only(graphs):
    train, test = graphs[:12], graphs[12:]
    scaler = FeatureScaler().fit(train)
    train_x = torch.cat([g["vm"].x for g in train])
    assert torch.allclose(scaler.mean, train_x.mean(0))
    scaled = scaler(train_x)
    assert torch.allclose(scaled.mean(0), torch.zeros(4), atol=1e-5)
    assert torch.allclose(scaled.std(0, unbiased=False), torch.ones(4), atol=1e-4)
    # Test graphs are transformed with the training statistics, not refitted.
    test_x = torch.cat([g["vm"].x for g in test])
    assert torch.allclose(scaler(test_x), (test_x - scaler.mean) / scaler.std)


def test_scaler_must_be_fitted():
    with pytest.raises(ModelInputError, match="not fitted"):
        FeatureScaler()(torch.zeros(2, 4))
    with pytest.raises(ModelInputError, match="empty"):
        FeatureScaler().fit([])


def test_constant_feature_is_only_centred(graphs):
    g = copy.copy(graphs[0])
    g["vm"].x = g["vm"].x.clone()
    g["vm"].x[:, 2] = 5.0
    scaler = FeatureScaler().fit([g])
    assert scaler.std[2] == 1.0
    assert torch.all(scaler(g["vm"].x)[:, 2] == 0)


def test_scaler_state_is_saved(graphs):
    scaler = FeatureScaler().fit(graphs)
    restored = FeatureScaler()
    restored.load_state_dict(scaler.state_dict())
    assert bool(restored.fitted)
    assert torch.equal(restored.mean, scaler.mean)


def test_node_encoder_shapes(graphs):
    batch = collate(graphs[:4])
    out = NodeEncoder(hidden_dim=16)(batch)
    for kind, h in out.items():
        assert h.shape == (batch[kind].num_nodes, 16)
    # Featureless types share one vector per type.
    assert torch.equal(out["storage"][0], out["storage"][-1])


def test_node_encoder_subset_of_types(graphs):
    encoder = NodeEncoder(hidden_dim=8, node_types=("vm",))
    assert set(encoder(collate(graphs[:2]))) == {"vm"}
    assert len(encoder.type_vectors) == 0
    with pytest.raises(ValueError, match="unknown node type"):
        NodeEncoder(hidden_dim=8, node_types=("router",))
