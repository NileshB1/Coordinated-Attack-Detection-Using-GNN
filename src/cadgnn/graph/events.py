"""Loading and checking the generated backup events before they go into graphs.

Rules applied by ``clean_events``:

- missing columns, unknown labels, an attack event without an episode id, a
  benign event with one, an episode mixing labels, the same VM twice in one
  episode, and VMs that are not in the infrastructure graph are errors
- rows with a missing or non-finite VM, timestamp or feature value are dropped
  with a warning, since there is no sensible value to fill in;
- exact duplicate rows are dropped with a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cadgnn.generator import EVENT_COLUMNS, LABELS
from cadgnn.graph.infrastructure import InfrastructureGraph
from cadgnn.graph.schema import VM_FEATURES, GraphConstructionError

log = logging.getLogger(__name__)

NUMERIC_COLUMNS = ["timestamp_hours", *VM_FEATURES, "anomaly_score"]


def load_events(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise GraphConstructionError(f"telemetry file not found: {path}")
    try:
        return pd.read_csv(path, dtype={"vm": str, "state": str, "episode_id": str, "label": str})
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise GraphConstructionError(f"could not read {path}: {exc}") from exc


def clean_events(events: pd.DataFrame, infra: InfrastructureGraph | None = None) -> pd.DataFrame:
    missing = [c for c in EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise GraphConstructionError(f"events are missing column(s): {', '.join(missing)}")
    events = events[EVENT_COLUMNS].copy()

    try:
        events[NUMERIC_COLUMNS] = events[NUMERIC_COLUMNS].apply(pd.to_numeric)
    except (ValueError, TypeError) as exc:
        raise GraphConstructionError(f"non-numeric value in event features: {exc}") from exc

    incomplete = events["vm"].isna() | ~np.isfinite(events[NUMERIC_COLUMNS]).all(axis=1)
    if incomplete.any():
        log.warning("dropping %d event(s) with missing or non-finite values", int(incomplete.sum()))
        events = events[~incomplete]

    duplicates = events.duplicated()
    if duplicates.any():
        log.warning("dropping %d duplicate event row(s)", int(duplicates.sum()))
        events = events[~duplicates]

    bad_labels = sorted(set(events["label"].fillna("<missing>")) - set(LABELS))
    if bad_labels:
        raise GraphConstructionError(f"unknown event label(s): {', '.join(bad_labels)}")

    benign = events["label"] == "none"
    if events.loc[~benign, "episode_id"].isna().any():
        raise GraphConstructionError("anomaly events without an episode_id")
    if events.loc[benign, "episode_id"].notna().any():
        raise GraphConstructionError("benign events with an episode_id")

    attacks = events[~benign]
    mixed = attacks.groupby("episode_id")["label"].nunique()
    if (mixed > 1).any():
        raise GraphConstructionError(
            f"episode(s) with more than one label: {', '.join(mixed[mixed > 1].index[:5])}"
        )
    repeated = attacks.duplicated(["episode_id", "vm"])
    if repeated.any():
        examples = attacks.loc[repeated, "episode_id"].unique()[:5]
        raise GraphConstructionError(f"VM appears twice in episode(s): {', '.join(examples)}")

    if infra is not None:
        known = set(infra.node_ids["vm"])
        unknown = sorted(set(events["vm"]) - known)
        if unknown:
            raise GraphConstructionError(
                f"{len(unknown)} event VM(s) are not VM nodes in the graph, e.g. {unknown[:5]}"
            )

    return events.reset_index(drop=True)


def anomaly_episodes(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Flagged events grouped by episode, each sorted by time (ties broken by VM id)."""
    attacks = events[events["label"] != "none"]
    return {
        episode_id: group.sort_values(["timestamp_hours", "vm"], kind="stable")
        for episode_id, group in sorted(attacks.groupby("episode_id"), key=lambda item: item[0])
    }
