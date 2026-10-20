import pytest

from cadgnn.datasets import DatasetError, inspect_datasets, require_datasets
from conftest import REPO_ROOT


def test_complete_tree_passes(fake_datasets):
    inv = require_datasets(fake_datasets)
    assert inv.ok
    assert len(inv.ransap) == 9
    assert inv.napierone is not None
    assert len(inv.fsl_snapshots) == 1


def test_missing_root(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        inspect_datasets(tmp_path / "nope")


def test_missing_ransap_trace_is_reported(fake_datasets):
    (fake_datasets / "ransap" / "Ryuk.csv").unlink()
    inv = inspect_datasets(fake_datasets)
    assert not inv.ok
    assert "Ryuk" not in inv.ransap
    assert any("Ryuk.csv" in p for p in inv.problems)
    with pytest.raises(DatasetError, match="Ryuk"):
        require_datasets(fake_datasets)


@pytest.mark.parametrize(
    "content, message",
    [
        ("", "empty"),
        ("1,2,3\n", "expected 6 columns"),
        ("unix_s,unix_ns,block_address,bytes_written,entropy_4kib,entropy_8x256b\n", "not numeric"),
    ],
)
def test_malformed_ransap_trace(fake_datasets, content, message):
    (fake_datasets / "ransap" / "Excel.csv").write_text(content, encoding="utf-8")
    inv = inspect_datasets(fake_datasets)
    assert "Excel" not in inv.ransap
    assert any(message in p for p in inv.problems)


def test_napierone_without_table(fake_datasets):
    inv_path = fake_datasets / "napierone" / "NapierOne-details.md"
    inv_path.write_text("# NapierOne\nno table here\n", encoding="utf-8")
    inv = inspect_datasets(fake_datasets)
    assert inv.napierone is None
    assert any("no markdown table" in p for p in inv.problems)


def test_truncated_fsl_header(fake_datasets):
    trace = next(fake_datasets.glob("filesystems/**/*.hash.anon"))
    trace.write_bytes(b"\0" * 100)
    inv = inspect_datasets(fake_datasets)
    assert inv.fsl_snapshots == []
    assert any("truncated" in p for p in inv.problems)
    assert any("no usable FSL" in p for p in inv.problems)


@pytest.mark.skipif(
    not (REPO_ROOT / "datasets").is_dir(), reason="local calibration datasets not present"
)
def test_repository_datasets_are_complete():
    inv = require_datasets(REPO_ROOT / "datasets")
    assert len(inv.ransap) == 9
    # The notebook run recorded 29 snapshot trace files.
    assert len(inv.fsl_snapshots) == 29
