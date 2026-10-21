"""Node and edge types of the heterogeneous infrastructure graph.

The types follow what the generator actually produces: every VM has one backup
image, each image lives on a storage pool, and each VM is attached to a network
fabric. Edges are stored pointing away from the VM, and every relation also gets
a reverse edge type so messages can flow both ways.
"""

from __future__ import annotations

from cadgnn.calibration import FEATURES
from cadgnn.generator import NODE_KINDS

NODE_TYPES = NODE_KINDS

EdgeType = tuple[str, str, str]

EDGE_TYPES: tuple[EdgeType, ...] = (
    ("vm", "has_backup", "backup_image"),
    ("backup_image", "stored_on", "storage"),
    ("vm", "attached_to", "network"),
)
REVERSE_RELATIONS = {
    "has_backup": "backup_of",
    "stored_on": "stores",
    "attached_to": "hosts",
}

# The four telemetry features become the VM node features, in this order.
VM_FEATURES = FEATURES

# Only flagged anomaly episodes are classified; benign backups are background.
TARGETS = {"coincidental": 0, "coordinated": 1}


class GraphConstructionError(ValueError):
    pass


def reverse(edge_type: EdgeType) -> EdgeType:
    src, rel, dst = edge_type
    return dst, REVERSE_RELATIONS[rel], src


def all_edge_types() -> list[EdgeType]:
    return [*EDGE_TYPES, *(reverse(et) for et in EDGE_TYPES)]


def edge_type_between(kind_a: str, kind_b: str) -> tuple[EdgeType, bool] | None:
    """Edge type linking two node kinds, and whether (a, b) has to be swapped to match it.

    Returns None when the schema has no relation between the two kinds.
    """
    for et in EDGE_TYPES:
        if (kind_a, kind_b) == (et[0], et[2]):
            return et, False
        if (kind_b, kind_a) == (et[0], et[2]):
            return et, True
    return None
