import pytest
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from cadgnn.config import ConfigError, ModelConfig
from cadgnn.graph import InfrastructureGraph, build_episode_graphs, clean_events
from cadgnn.models import (
    DualStreamModel,
    ModelInputError,
    NodeEncoder,
    collate,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    seed_everything,
)

CONFIG = ModelConfig(hidden_dim=16, temporal_dim=8, spatial_dim=12, dropout=0.0)


class PoolingStream(nn.Module):
    """Stand-in stream for these tests: encode, mean-pool per graph, project."""

    def __init__(self, out_dim, node_types):
        super().__init__()
        self.out_dim = out_dim
        self.node_types = node_types
        self.encoder = NodeEncoder(CONFIG.hidden_dim, node_types=node_types)
        self.head = nn.Linear(CONFIG.hidden_dim, out_dim)

    def forward(self, batch):
        h = self.encoder(batch)
        pooled = sum(
            global_mean_pool(h[k], batch[k].batch, size=batch.num_graphs) for k in self.node_types
        )
        return self.head(pooled)


def make_model(config=CONFIG):
    temporal = PoolingStream(config.temporal_dim, ("vm",))
    spatial = PoolingStream(config.spatial_dim, ("vm", "backup_image", "storage", "network"))
    return DualStreamModel(config, temporal, spatial)


@pytest.fixture(scope="module")
def graphs(small_dataset):
    infra = InfrastructureGraph.from_networkx(small_dataset.graph)
    return build_episode_graphs(infra, clean_events(small_dataset.events, infra))


@pytest.fixture
def model(graphs):
    seed_everything(0)
    return make_model().fit_scaler(graphs[:12])


def test_forward_shapes(model, graphs):
    batch = collate(graphs[:5])
    out = model(batch)
    assert out.temporal.shape == (5, 8)
    assert out.spatial.shape == (5, 12)


def test_forward_does_not_modify_the_batch(model, graphs):
    batch = collate(graphs[:3])
    raw = batch["vm"].x.clone()
    model(batch)
    assert torch.equal(batch["vm"].x, raw)


def test_streams_see_scaled_features(model, graphs):
    seen = {}

    def capture(module, args):
        seen["x"] = args[0]["vm"].x

    model.temporal.register_forward_pre_hook(capture)
    batch = collate(graphs[:3])
    model(batch)
    assert torch.allclose(seen["x"], model.scaler(batch["vm"].x))


def test_gradients_reach_every_trainable_parameter(model, graphs):
    out = model(collate(graphs))
    (out.temporal.sum() + out.spatial.sum()).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    # Scaler statistics are buffers, not parameters.
    assert "scaler.mean" in dict(model.named_buffers())


def test_streams_work_on_their_own(graphs):
    model = make_model().fit_scaler(graphs)
    batch = collate(graphs[:4])
    assert model.temporal(batch).shape == (4, 8)
    assert model.spatial(batch).shape == (4, 12)


def test_unfitted_scaler_is_an_error(graphs):
    with pytest.raises(ModelInputError, match="not fitted"):
        make_model()(collate(graphs[:2]))


def test_stream_out_dim_must_match_config():
    with pytest.raises(ModelInputError, match="temporal stream out_dim is 5"):
        DualStreamModel(CONFIG, PoolingStream(5, ("vm",)), PoolingStream(12, ("vm",)))
    with pytest.raises(ModelInputError, match="spatial stream out_dim is None"):
        DualStreamModel(CONFIG, PoolingStream(8, ("vm",)), nn.Linear(4, 12))


class WrongShape(PoolingStream):
    def forward(self, batch):
        return super().forward(batch)[:1]


class NotFinite(PoolingStream):
    def forward(self, batch):
        return super().forward(batch) * float("nan")


@pytest.mark.parametrize(
    "stream_cls, message", [(WrongShape, "expected"), (NotFinite, "non-finite")]
)
def test_bad_stream_output_is_caught(graphs, stream_cls, message):
    model = DualStreamModel(CONFIG, stream_cls(8, ("vm",)), PoolingStream(12, ("vm",)))
    model.fit_scaler(graphs)
    with pytest.raises(ModelInputError, match=message):
        model(collate(graphs[:3]))


def test_same_seed_same_model(graphs):
    seed_everything(5)
    a = make_model().fit_scaler(graphs)
    seed_everything(5)
    b = make_model().fit_scaler(graphs)
    batch = collate(graphs[:4])
    assert torch.equal(a(batch).spatial, b(batch).spatial)


def test_dropout_is_off_in_eval(graphs):
    config = ModelConfig(hidden_dim=16, temporal_dim=8, spatial_dim=12, dropout=0.5)
    temporal = PoolingStream(8, ("vm",))
    temporal.encoder = NodeEncoder(16, node_types=("vm",), dropout=config.dropout)
    model = DualStreamModel(config, temporal, PoolingStream(12, ("vm",))).fit_scaler(graphs)
    batch = collate(graphs[:4])
    model.eval()
    assert torch.equal(model(batch).temporal, model(batch).temporal)


def test_checkpoint_round_trip(tmp_path, model, graphs):
    path = tmp_path / "ckpt" / "model.pt"
    save_checkpoint(model, path, epoch=3)
    restored = make_model()
    checkpoint = load_checkpoint(restored, path)
    assert checkpoint["epoch"] == 3
    assert bool(restored.scaler.fitted)
    batch = collate(graphs[:4])
    model.eval()
    restored.eval()
    assert torch.equal(model(batch).temporal, restored(batch).temporal)


def test_checkpoint_config_mismatch(tmp_path, model):
    path = tmp_path / "model.pt"
    save_checkpoint(model, path)
    other = make_model(ModelConfig(hidden_dim=16, temporal_dim=8, spatial_dim=12, dropout=0.2))
    with pytest.raises(ModelInputError, match="model config"):
        load_checkpoint(other, path)
    with pytest.raises(ModelInputError, match="not found"):
        load_checkpoint(other, tmp_path / "missing.pt")


def test_device_placement(model, graphs):
    device = resolve_device("cpu")
    out = model.to(device)(collate(graphs[:2]).to(device))
    assert out.temporal.device.type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA error path")
def test_cuda_requested_without_cuda():
    with pytest.raises(ModelInputError, match="CUDA is not available"):
        resolve_device("cuda")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"temporal_dim": 2.5}, "temporal_dim"),
        ({"dropout": 1.0}, "dropout"),
        ({"dropout": -0.1}, "dropout"),
        ({"device": "tpu"}, "device"),
    ],
)
def test_model_config_validation(kwargs, message):
    with pytest.raises(ConfigError, match=message):
        ModelConfig(**kwargs)


def test_a_stream_can_be_switched_off(graphs):
    model = DualStreamModel(CONFIG, None, PoolingStream(12, ("vm",))).fit_scaler(graphs)
    out = model(collate(graphs[:3]))
    assert out.temporal is None
    assert out.spatial.shape == (3, 12)
    with pytest.raises(ModelInputError, match="at least one stream"):
        DualStreamModel(CONFIG, None, None)
