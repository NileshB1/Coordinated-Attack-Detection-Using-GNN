"""Feature ranges for the anomaly generator, calibrated from the local datasets
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cadgnn.datasets import (
    FSL_HEADER_BYTES,
    FSL_PATTERN,
    NAPIERONE_FILE,
    RANSAP_COLUMNS,
    RANSAP_TRACES,
    DatasetError,
)

log = logging.getLogger(__name__)

FEATURE_WEIGHTS = {
    "change_rate": 0.35,
    "file_entropy": 0.30,
    "backup_size_gb": 0.20,
    "file_count": 0.15,
}
FEATURES = tuple(FEATURE_WEIGHTS)
STATES = ("benign", "infected")

NAPIERONE_NORMAL_TYPES = (
    "DOC", "DOCX", "XLS", "XLSX", "PPT", "PPTX", "PDF", "TXT", "CSV", "HTML", "XML", "JSON",
)  # fmt: skip
NAPIERONE_RANSOMWARE_FAMILIES = (
    "NOTPETYA", "SODINOKIBI", "MAZE", "PHOBOS", "NETWALKER", "DHARMA", "RYUK",
)  # fmt: skip
# Column 5 of the NapierOne table is "Average Size (KB)".
NAPIERONE_AVG_SIZE_COLUMN = 5
NAPIERONE_FALLBACK_NORMAL_KB = 1161.0
NAPIERONE_FALLBACK_RANSOMWARE_KB = 844.0
FSL_FALLBACK_FILE_COUNT = 30052

BENIGN_BACKUP_SIZE_GB = (0.5, 25.0)  # estimated, not measured
BENIGN_MIN_FILE_COUNT = 200
INFECTED_FILE_COUNT_MULTIPLIER = (0.40, 1.60)  # assumed
SIZE_RATIO_SPREAD = 0.1  # infected size range is ratio * (1 -/+ spread)


class CalibrationError(Exception):
    pass


@dataclass(frozen=True)
class FeatureProfile:
    weight: float
    benign_low: float
    benign_high: float
    infected_low: float
    infected_high: float
    source: str

    def bounds(self, state: str) -> tuple[float, float]:
        if state == "benign":
            return self.benign_low, self.benign_high
        if state == "infected":
            return self.infected_low, self.infected_high
        raise ValueError(f"unknown state {state!r}, expected one of {STATES}")

    def normalise(self, value: float) -> float:
        lo = min(self.benign_low, self.infected_low)
        hi = max(self.benign_high, self.infected_high)
        return (value - lo) / (hi - lo) if hi > lo else 0.0


ParamProfile = dict[str, FeatureProfile]


@dataclass
class Calibration:
    ransap: pd.DataFrame
    napierone_normal_kb: float
    napierone_ransomware_kb: float
    fsl: pd.DataFrame
    fsl_file_count: int
    profile: ParamProfile
    fallbacks: list[str] = field(default_factory=list)

    @property
    def size_ratio(self) -> float:
        return self.napierone_ransomware_kb / self.napierone_normal_kb


def weighted_score(profile: ParamProfile, values: Mapping[str, float]) -> float:
    return sum(profile[name].weight * profile[name].normalise(v) for name, v in values.items())


def read_ransap(datasets_dir: str | Path) -> pd.DataFrame:
    """Mean 4 KiB write entropy and write throughput (MB/s) of each RanSAP trace."""
    rows = []
    for name, is_ransomware in RANSAP_TRACES.items():
        path = Path(datasets_dir) / "ransap" / f"{name}.csv"
        if not path.is_file():
            raise DatasetError(f"missing RanSAP trace {path}")
        try:
            df = pd.read_csv(path, header=None, names=list(RANSAP_COLUMNS))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            raise DatasetError(f"could not read RanSAP trace {path}: {exc}") from exc
        if df.empty or df.isna().any().any():
            raise DatasetError(f"RanSAP trace {path} is empty or has missing values")

        timestamps = df["unix_s"] + df["unix_ns"] / 1_000_000_000
        duration = float(timestamps.max() - timestamps.min())
        total_mb = float(df["bytes_written"].sum() / 1_000_000)
        mbps = total_mb / duration if duration > 0 else 0.0
        # The notebook rounds to 4 places before deriving the ranges; keep that.
        rows.append(
            {
                "name": name,
                "is_ransomware": is_ransomware,
                "avg_entropy": round(float(df["entropy_4kib"].mean()), 4),
                "avg_throughput_mbps": round(mbps, 4),
            }
        )
    return pd.DataFrame(rows)


def read_napierone(path: str | Path) -> tuple[list[float], list[float]]:
    """Average file sizes (KB) of the normal office types and the ransomware outputs."""
    normal_kb, ransomware_kb = [], []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells[0] in NAPIERONE_NORMAL_TYPES:
                target = normal_kb
            elif any(family in line for family in NAPIERONE_RANSOMWARE_FAMILIES):
                target = ransomware_kb
            else:
                continue
            try:
                target.append(float(cells[NAPIERONE_AVG_SIZE_COLUMN].replace(",", "")))
            except (IndexError, ValueError):
                log.debug("skipping NapierOne row without a numeric size: %s", line.strip())
    return normal_kb, ransomware_kb


def read_fsl_snapshots(datasets_dir: str | Path) -> pd.DataFrame:
    """One row per FSL .hash.anon trace with the file count stored in its header."""
    rows = []
    for path in sorted(Path(datasets_dir).glob(FSL_PATTERN)):
        with path.open("rb") as f:
            header = f.read(FSL_HEADER_BYTES)
        count = (
            struct.unpack("<I", header[FSL_HEADER_BYTES - 4 : FSL_HEADER_BYTES])[0]
            if len(header) >= FSL_HEADER_BYTES
            else 0
        )
        # e.g. fslhomes-user005-2012-03-09.128kb.hash.anon
        parts = path.name.split("-")
        rows.append(
            {
                "filename": path.name,
                "user_id": parts[1] if len(parts) > 1 else "unknown",
                "snapshot_date": (
                    f"{parts[2]}-{parts[3]}-{parts[4].split('.')[0]}"
                    if len(parts) >= 5
                    else "unknown"
                ),
                "chunk_size": path.name.split(".")[-3] if path.name.count(".") >= 2 else "unknown",
                "extracted_file_count": count,
            }
        )
    columns = ["filename", "user_id", "snapshot_date", "chunk_size", "extracted_file_count"]
    return pd.DataFrame(rows, columns=columns)


def build_param_profile(
    ransap: pd.DataFrame, size_ratio: float, fsl_file_count: int
) -> ParamProfile:
    benign = ransap[~ransap["is_ransomware"]]
    infected = ransap[ransap["is_ransomware"]]
    if benign.empty or infected.empty:
        raise CalibrationError("RanSAP calibration needs both benign and ransomware traces")
    max_mbps = float(ransap["avg_throughput_mbps"].max())
    if max_mbps <= 0:
        raise CalibrationError("RanSAP traces have no measurable write throughput")
    if size_ratio <= 0:
        raise CalibrationError(f"NapierOne size ratio must be positive, got {size_ratio}")
    if fsl_file_count <= BENIGN_MIN_FILE_COUNT:
        raise CalibrationError(
            f"FSL file count {fsl_file_count} is not above the benign minimum "
            f"{BENIGN_MIN_FILE_COUNT}"
        )

    # change_rate is write throughput scaled by the fastest trace, so it lies in [0, 1].
    return {
        "change_rate": FeatureProfile(
            weight=FEATURE_WEIGHTS["change_rate"],
            benign_low=0.0,
            benign_high=float(benign["avg_throughput_mbps"].max()) / max_mbps,
            infected_low=float(infected["avg_throughput_mbps"].min()) / max_mbps,
            infected_high=1.0,
            source="real data (RanSAP storage access traces)",
        ),
        "file_entropy": FeatureProfile(
            weight=FEATURE_WEIGHTS["file_entropy"],
            benign_low=float(benign["avg_entropy"].min()),
            benign_high=float(benign["avg_entropy"].max()),
            infected_low=float(infected["avg_entropy"].min()),
            infected_high=float(infected["avg_entropy"].max()),
            source="real data (RanSAP Shannon entropy values)",
        ),
        "backup_size_gb": FeatureProfile(
            weight=FEATURE_WEIGHTS["backup_size_gb"],
            benign_low=BENIGN_BACKUP_SIZE_GB[0],
            benign_high=BENIGN_BACKUP_SIZE_GB[1],
            infected_low=size_ratio * (1 - SIZE_RATIO_SPREAD),
            infected_high=size_ratio * (1 + SIZE_RATIO_SPREAD),
            source="benign range estimated; infected ratio = real data (NapierOne)",
        ),
        "file_count": FeatureProfile(
            weight=FEATURE_WEIGHTS["file_count"],
            benign_low=BENIGN_MIN_FILE_COUNT,
            benign_high=fsl_file_count,
            infected_low=INFECTED_FILE_COUNT_MULTIPLIER[0],
            infected_high=INFECTED_FILE_COUNT_MULTIPLIER[1],
            source=(
                f"real snapshot count ({fsl_file_count} files) from Filesystems Archive .hash.anon"
            ),
        ),
    }


def calibrate(datasets_dir: str | Path) -> Calibration:
    datasets_dir = Path(datasets_dir)
    if not datasets_dir.is_dir():
        raise DatasetError(f"datasets directory not found: {datasets_dir}")
    fallbacks = []

    ransap = read_ransap(datasets_dir)

    napier_path = datasets_dir / NAPIERONE_FILE
    normal_kb, ransomware_kb = read_napierone(napier_path) if napier_path.is_file() else ([], [])
    if normal_kb:
        normal_avg = sum(normal_kb) / len(normal_kb)
    else:
        normal_avg = NAPIERONE_FALLBACK_NORMAL_KB
        fallbacks.append(f"NapierOne normal file size: no rows, using {normal_avg} KB")
    if ransomware_kb:
        ransomware_avg = sum(ransomware_kb) / len(ransomware_kb)
    else:
        ransomware_avg = NAPIERONE_FALLBACK_RANSOMWARE_KB
        fallbacks.append(f"NapierOne ransomware file size: no rows, using {ransomware_avg} KB")

    fsl = read_fsl_snapshots(datasets_dir)
    if fsl.empty:
        fsl_file_count = FSL_FALLBACK_FILE_COUNT
        fallbacks.append(f"FSL file count: no snapshot traces, using {fsl_file_count}")
    else:
        fsl_file_count = int(fsl["extracted_file_count"].max())

    for message in fallbacks:
        log.warning("calibration fallback: %s", message)

    return Calibration(
        ransap=ransap,
        napierone_normal_kb=normal_avg,
        napierone_ransomware_kb=ransomware_avg,
        fsl=fsl,
        fsl_file_count=fsl_file_count,
        profile=build_param_profile(ransap, ransomware_avg / normal_avg, fsl_file_count),
        fallbacks=fallbacks,
    )


def check_score_separation(
    profile: ParamProfile, seed: int, n_samples: int = 300
) -> tuple[float, float]:
    """Mean weighted score of benign and infected samples; infected must be higher.

    Uses its own generator seeded with ``seed`` (as in the notebook), so it does
    not consume draws from the main generation stream.
    """
    rng = np.random.default_rng(seed)
    means = []
    for state in STATES:
        scores = []
        for _ in range(n_samples):
            values = {n: float(rng.uniform(*f.bounds(state))) for n, f in profile.items()}
            scores.append(weighted_score(profile, values))
        means.append(float(np.mean(scores)))
    benign_mean, infected_mean = means
    if infected_mean <= benign_mean:
        raise CalibrationError(
            f"infected mean score {infected_mean:.3f} does not exceed benign {benign_mean:.3f}"
        )
    return benign_mean, infected_mean
