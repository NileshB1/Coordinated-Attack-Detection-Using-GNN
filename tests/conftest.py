import struct
from pathlib import Path

import pytest

from cadgnn.datasets import FSL_HEADER_BYTES, NAPIERONE_FILE, RANSAP_TRACES

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_datasets(tmp_path: Path) -> Path:
    """A small datasets tree with the expected layout that is also enough to calibrate.

    Benign traces write 1 MB/s at entropy 0.4, ransomware traces 3 MB/s at 0.7;
    the NapierOne table gives a 0.75 size ratio and the FSL trace 5,000 files.
    """
    root = tmp_path / "datasets"
    (root / "ransap").mkdir(parents=True)
    for name, is_ransomware in RANSAP_TRACES.items():
        mb, entropy = (3_000_000, 0.7) if is_ransomware else (1_000_000, 0.4)
        rows = [(1596636000, 0, 4096, mb, entropy, 0.5), (1596636001, 0, 8192, mb, entropy, 0.5)]
        (root / "ransap" / f"{name}.csv").write_text(
            "".join(",".join(map(str, r)) + "\n" for r in rows), encoding="utf-8"
        )

    napier = root / NAPIERONE_FILE
    napier.parent.mkdir(parents=True)
    napier.write_text(
        "| File Type | Size (MB) | Files # | Min | Max | Average Size (KB) |\n"
        "|---|---|---|---|---|---|\n"
        "| DOCX | 1 | 5 | 1 | 9 | 1,200 |\n"
        "| EXTRA | 1 | 5 | 1 | 9 | 900 | RANSOMWARE | RYUK |\n",
        encoding="utf-8",
    )

    snapshot = root / "filesystems" / "fslhomes-user001-2015-04-10"
    snapshot.mkdir(parents=True)
    (snapshot / "fslhomes-user001-2015-04-10.8kb.hash.anon").write_bytes(
        b"\0" * (FSL_HEADER_BYTES - 4) + struct.pack("<I", 5000)
    )
    return root


@pytest.fixture(scope="session")
def small_dataset():
    """A small generated scenario (60 VMs, 18 episodes) with a fixed, simple profile."""
    from cadgnn.calibration import FeatureProfile
    from cadgnn.config import GeneratorConfig
    from cadgnn.generator import generate

    profile = {
        "change_rate": FeatureProfile(0.35, 0.0, 0.1, 0.5, 1.0, ""), "file_entropy": FeatureProfile(0.30, 0.3, 0.5, 0.6, 0.8, ""),
        "backup_size_gb": FeatureProfile(0.20, 0.5, 25.0, 0.6, 0.8, ""),  "file_count": FeatureProfile(0.15, 200, 30000, 0.4, 1.6, "")
    }
    config = GeneratorConfig(
        n_vms=60, n_storage=4, n_network=5, n_coincidental_episodes=10, n_coordinated_episodes=8
    )
    return generate(config, profile, seed=21)
