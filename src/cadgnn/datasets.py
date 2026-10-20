"""Locate and sanity-check the local calibration datasets.

The pipeline only reads datasets from disk; it never downloads them during a
run. Expected layout under the datasets directory:

    ransap/<Trace>.csv   => RanSAP ata_write traces (win7-120gb-hdd)
    napierone/NapierOne-details.md
    filesystems/**/*.hash.anon   => FSL fslhomes snapshot traces
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Trace name -> whether the program is ransomware. Each trace is the ata_write.csv
# of one RanSAP run, saved under the trace name.
RANSAP_TRACES: dict[str, bool] = {
    "Excel": False,  "Firefox": False,
    "Cerber": True, "Darkside": True,
    "GandCrab4": True,
    "Ryuk": True,  "Sodinokibi": True,
    "TeslaCrypt": True,  "WannaCry": True,
}
RANSAP_COLUMNS = (
    "unix_s",  "unix_ns",
    "block_address",
    "bytes_written", "entropy_4kib",
    "entropy_8x256b",
)
NAPIERONE_FILE = "napierone/NapierOne-details.md"
FSL_PATTERN = "filesystems/**/*.hash.anon"
# The snapshot record count is a little-endian uint32 at byte offset 4216.
FSL_HEADER_BYTES = 4220


class DatasetError(Exception):
    pass


@dataclass
class DatasetInventory:
    root: Path
    ransap: dict[str, Path] = field(default_factory=dict)
    napierone: Path | None = None
    fsl_snapshots: list[Path] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        lines = [
            f"datasets root: {self.root} ",
            f"RanSAP traces: {len(self.ransap)}/{len(RANSAP_TRACES)} ",
            f"NapierOne table: {'found' if self.napierone else 'missing'}",
            f"FSL snapshot traces: {len(self.fsl_snapshots)}",
        ]
        lines += [f"  problem: {p}" for p in self.problems]
        return "\n".join(lines)


def inspect_datasets(root: str | Path) -> DatasetInventory:
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"datasets directory not found: {root}")

    inv = DatasetInventory(root=root)

    for name in RANSAP_TRACES:
        path = root / "ransap" / f"{name}.csv"
        if not path.is_file():
            inv.problems.append(f"missing RanSAP trace {path}")
            continue
        problem = _check_ransap_csv(path)
        if problem:
            inv.problems.append(problem)
        else:
            inv.ransap[name] = path

    napier = root / NAPIERONE_FILE
    if not napier.is_file():
        inv.problems.append(f"missing NapierOne table {napier}")
    elif not _has_markdown_table(napier):
        inv.problems.append(f"{napier}: no markdown table rows found")
    else:
        inv.napierone = napier

    for path in sorted(root.glob(FSL_PATTERN)):
        if path.stat().st_size < FSL_HEADER_BYTES:
            inv.problems.append(f"{path}: truncated header ({path.stat().st_size} bytes)")
        else:
            inv.fsl_snapshots.append(path)
    if not inv.fsl_snapshots:
        inv.problems.append(f"no usable FSL .hash.anon files under {root / 'filesystems'}")

    return inv


def require_datasets(root: str | Path) -> DatasetInventory:
    inv = inspect_datasets(root)
    if not inv.ok:
        raise DatasetError("dataset check failed:\n" + "\n".join(inv.problems))
    return inv


def _check_ransap_csv(path: Path) -> str | None:
    # Traces are headerless; checking the first row catches empty files and wrong formats
    # without reading tens of megabytes.
    with path.open(encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
    if not first:
        return f"{path}: file is empty"
    fields = first.split(",")
    if len(fields) != len(RANSAP_COLUMNS):
        return f"{path}: expected {len(RANSAP_COLUMNS)} columns, found {len(fields)}"
    try:
        [float(x) for x in fields]
    except ValueError:
        return f"{path}: first row is not numeric: {first[:80]!r}"
    return None


def _has_markdown_table(path: Path) -> bool:
    with path.open(encoding="utf-8", errors="replace") as f:
        return any(line.startswith("|") for line in f)
