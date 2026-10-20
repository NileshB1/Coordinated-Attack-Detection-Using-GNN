from pathlib import Path

import pytest

from cadgnn.datasets import FSL_HEADER_BYTES, NAPIERONE_FILE, RANSAP_TRACES

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_datasets(tmp_path: Path) -> Path:
    """A minimal datasets tree with the layout inspect_datasets expects."""
    root = tmp_path / "datasets"
    (root / "ransap").mkdir(parents=True)
    for name in RANSAP_TRACES:
        (root / "ransap" / f"{name}.csv").write_text(
            "1596636000,123,4096,4096,0.52,0.48\n", encoding="utf-8"
        )

    napier = root / NAPIERONE_FILE
    napier.parent.mkdir(parents=True)
    napier.write_text("| File Type | Size (MB) |\n|---|---|\n| DOCX | 1,420 |\n", encoding="utf-8")

    snapshot = root / "filesystems" / "fslhomes-user001-2015-04-10"
    snapshot.mkdir(parents=True)
    (snapshot / "fslhomes-user001-2015-04-10.8kb.hash.anon").write_bytes(b"\0" * FSL_HEADER_BYTES)
    return root
