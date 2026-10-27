import json
import sys

import pandas as pd
import pytest

from cadgnn.data.calibration import calibrate
from cadgnn.config import GeneratorConfig
from cadgnn.generator import generate
from conftest import REPO_ROOT

NOTEBOOK = REPO_ROOT / "notebooks" / "random_anomaly_generator_simple.ipynb"
DATASETS = REPO_ROOT / "datasets"

pytestmark = [
    pytest.mark.skipif(not DATASETS.is_dir(), reason="local calibration datasets not present"),
    # The notebook nests same-type quotes inside an f-string.
    pytest.mark.skipif(sys.version_info < (3, 12), reason="notebook needs Python 3.12+"),
]

NOTEBOOK_HEAD = [
    ("vm_4142", 0.002, 0.0710, 0.4429, 11.9395, 5398.8195, 0.2294), ("vm_322", 0.003, 0.0550, 0.4460, 22.8759, 9748.0689, 0.3370),
    ("vm_4684", 0.004, 0.0523, 0.4304, 23.9531, 13672.9032, 0.3532), ("vm_11250", 0.004, 0.0425, 0.4623, 9.4953, 9508.7041, 0.2339),
    ("vm_3010", 0.004, 0.0032, 0.4797, 16.6065, 28397.3921, 0.3851),  ("vm_11428", 0.005, 0.0390, 0.4703, 20.1534, 28313.9496, 0.4194),
    ("vm_1434", 0.007, 0.0587, 0.4918, 23.9178, 8579.2033, 0.3740), ("vm_5710", 0.007, 0.0357, 0.5074, 24.3372, 21297.9515, 0.4440),
    ("vm_3184", 0.007, 0.0607, 0.3677, 19.5320, 8479.6203, 0.2489), ("vm_10208", 0.008, 0.0863, 0.5195, 4.3337, 306.5335, 0.2024)
]
NOTEBOOK_LABEL_COUNTS = {"none": 72000, "coincidental": 465, "coordinated": 288}


def run_notebook(tmp_path, monkeypatch, n_vms, n_storage, n_network):
    pytest.importorskip("matplotlib")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    # Setup, RanSAP, NapierOne, FSL, profile, graph and event cells; analysis, plot and
    # export cells are not needed.
    cells = cells[:7]
    assert "events_df = " in cells[-1], "notebook layout changed"

    source = "\n".join(cells)
    replacements = {
        '"datasets/': f'"{DATASETS.as_posix()}/',
        "N_VMS = 12000": f"N_VMS = {n_vms}",
        "N_STORAGE = 300": f"N_STORAGE = {n_storage}", "N_NETWORK = 200": f"N_NETWORK = {n_network}"
    }
    for old, new in replacements.items():
        assert old in source, f"{old!r} not found in notebook"
        source = source.replace(old, new)

    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "notebook"}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def calibration():
    return calibrate(DATASETS)


def test_calibration_matches_notebook(tmp_path, monkeypatch, calibration):
    ns = run_notebook(tmp_path, monkeypatch, 20, 2, 2)
    for name, feature in calibration.profile.items():
        expected = ns["PARAM_PROFILE"][name]
        for key in ("weight", "benign_low", "benign_high", "infected_low", "infected_high"):
            assert getattr(feature, key) == expected[key], (name, key)
        assert feature.source == expected["source"]


@pytest.mark.parametrize("sizes", [(40, 3, 4), (400, 12, 9)])
def test_graph_and_events_match_notebook(tmp_path, monkeypatch, calibration, sizes):
    n_vms, n_storage, n_network = sizes
    ns = run_notebook(tmp_path, monkeypatch, n_vms, n_storage, n_network)
    config = GeneratorConfig(n_vms=n_vms, n_storage=n_storage, n_network=n_network)
    dataset = generate(config, calibration.profile, seed=ns["RNG_SEED"])

    assert list(dataset.graph.nodes(data="kind")) == list(ns["infrastructure"].nodes(data="kind"))
    assert list(dataset.graph.edges()) == list(ns["infrastructure"].edges())
    pd.testing.assert_frame_equal(dataset.events, ns["events_df"])


def test_full_scale_run_matches_recorded_notebook_output(calibration):
    events = generate(GeneratorConfig(), calibration.profile, seed=42).events
    assert events["label"].value_counts().to_dict() == NOTEBOOK_LABEL_COUNTS
    columns = ["vm", "timestamp_hours", "change_rate", "file_entropy", "backup_size_gb",
               "file_count", "anomaly_score"]
    head = list(events[columns].head(10).itertuples(index=False, name=None))
    assert head == NOTEBOOK_HEAD
