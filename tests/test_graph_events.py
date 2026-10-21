import logging

import numpy as np
import pytest

from cadgnn.generator import save_outputs
from cadgnn.graph import (
    GraphConstructionError,
    InfrastructureGraph,
    anomaly_episodes,
    clean_events,
    load_events,
)


@pytest.fixture
def events(small_dataset):
    return small_dataset.events.copy()


@pytest.fixture
def infra(small_dataset):
    return InfrastructureGraph.from_networkx(small_dataset.graph)


def test_clean_generator_output_passes_unchanged(events, infra):
    cleaned = clean_events(events, infra)
    assert len(cleaned) == len(events)
    assert cleaned["vm"].tolist() == events["vm"].tolist()


def test_round_trip_through_csv(tmp_path, small_dataset, fake_datasets, infra):
    from cadgnn.calibration import calibrate

    paths = save_outputs(small_dataset, calibrate(fake_datasets), tmp_path)
    loaded = clean_events(load_events(paths["telemetry"]), infra)
    assert len(loaded) == len(small_dataset.events)
    assert loaded["episode_id"].isna().sum() == (small_dataset.events["label"] == "none").sum()


def test_missing_file(tmp_path):
    with pytest.raises(GraphConstructionError, match="not found"):
        load_events(tmp_path / "nope.csv")


def test_missing_column(events):
    with pytest.raises(GraphConstructionError, match="missing column.*file_entropy"):
        clean_events(events.drop(columns="file_entropy"))


def test_non_numeric_feature(events):
    events["change_rate"] = events["change_rate"].astype(object)
    events.loc[0, "change_rate"] = "high"
    with pytest.raises(GraphConstructionError, match="non-numeric"):
        clean_events(events)


def test_rows_with_missing_values_are_dropped(events, caplog):
    caplog.set_level(logging.WARNING)
    events.loc[0, "file_count"] = np.nan
    events.loc[1, "timestamp_hours"] = np.inf
    events.loc[2, "vm"] = None
    cleaned = clean_events(events)
    assert len(cleaned) == len(events) - 3
    assert "3 event(s) with missing or non-finite values" in caplog.text


def test_duplicate_rows_are_dropped(events, caplog):
    caplog.set_level(logging.WARNING)
    doubled = events.iloc[[0, 0, 1]]
    assert len(clean_events(doubled)) == 2
    assert "1 duplicate event" in caplog.text


def first_row(events, label):
    return events.index[events["label"] == label][0]


@pytest.mark.parametrize("bad_label", ["normal", None])
def test_unknown_or_missing_label(events, bad_label):
    events.loc[first_row(events, "coordinated"), "label"] = bad_label
    with pytest.raises(GraphConstructionError, match="unknown event label"):
        clean_events(events)


def test_attack_event_needs_episode_id(events):
    events.loc[first_row(events, "coordinated"), "episode_id"] = None
    with pytest.raises(GraphConstructionError, match="without an episode_id"):
        clean_events(events)


def test_benign_event_must_not_have_episode_id(events):
    events.loc[first_row(events, "none"), "episode_id"] = "coordinated_0"
    with pytest.raises(GraphConstructionError, match="benign events with"):
        clean_events(events)


def test_mixed_label_episode(events):
    row = events.index[events["label"] == "coordinated"][0]
    events.loc[row, "label"] = "coincidental"
    with pytest.raises(GraphConstructionError, match="more than one label"):
        clean_events(events)


def test_vm_twice_in_one_episode(events):
    episode = events[events["episode_id"] == "coincidental_0"]
    a, b = episode.index[:2]
    events.loc[b, "vm"] = events.loc[a, "vm"]
    with pytest.raises(GraphConstructionError, match="VM appears twice"):
        clean_events(events)


def test_unknown_vm(events, infra):
    events.loc[0, "vm"] = "vm_999"
    with pytest.raises(GraphConstructionError, match="not VM nodes"):
        clean_events(events, infra)


def test_anomaly_episodes_are_sorted_and_exclude_benign(events):
    episodes = anomaly_episodes(clean_events(events))
    assert list(episodes) == sorted(episodes)
    assert len(episodes) == 18
    for episode_id, ep in episodes.items():
        assert (ep["episode_id"] == episode_id).all()
        assert ep["timestamp_hours"].is_monotonic_increasing
        assert (ep["label"] != "none").all()


def test_episode_ties_are_broken_by_vm(events):
    events = clean_events(events)
    ep_id = events.loc[events["label"] == "coincidental", "episode_id"].iloc[0]
    events.loc[events["episode_id"] == ep_id, "timestamp_hours"] = 5.0
    ep = anomaly_episodes(events)[ep_id]
    assert ep["vm"].tolist() == sorted(ep["vm"])
