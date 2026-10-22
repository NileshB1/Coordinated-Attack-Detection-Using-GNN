import struct

import pandas as pd
import pytest

from cadgnn.calibration import (
    FSL_FALLBACK_FILE_COUNT, CalibrationError,
    FeatureProfile,  build_param_profile,
    calibrate, check_score_separation,
    read_fsl_snapshots, read_napierone,
    weighted_score
)
from cadgnn.datasets import FSL_HEADER_BYTES, DatasetError
from conftest import REPO_ROOT

REAL_DATASETS = REPO_ROOT / "datasets"
needs_real_datasets = pytest.mark.skipif(
    not REAL_DATASETS.is_dir(), reason="local calibration datasets not present"
)

# Printed by the reference notebook (section 1) for the same traces.
NOTEBOOK_RANSAP = {
    "Excel": (0.5200, 1.1295),  "Firefox": (0.3653, 0.7945),
    "Cerber": (0.6002, 2.9067), "Darkside": (0.5626, 2.8400),
    "GandCrab4": (0.6640, 4.6111),
    "Ryuk": (0.6217, 5.6170), "Sodinokibi": (0.5947, 4.9115),
    "TeslaCrypt": (0.7423, 1.0086), "WannaCry": (0.3261, 12.2243)
}


@pytest.fixture(scope="module")
def real_calibration():
    if not REAL_DATASETS.is_dir():
        pytest.skip("local calibration datasets not present")
    return calibrate(REAL_DATASETS)


def write_ransap(root, name, rows):
    (root / "ransap" / f"{name}.csv").write_text(
        "".join(",".join(map(str, r)) + "\n" for r in rows), encoding="utf-8"
    )


def fsl_trace(count: int) -> bytes:
    return b"\0" * (FSL_HEADER_BYTES - 4) + struct.pack("<I", count)


@needs_real_datasets
def test_ransap_matches_notebook(real_calibration):
    ransap = real_calibration.ransap.set_index("name")
    for name, (entropy, mbps) in NOTEBOOK_RANSAP.items():
        assert ransap.loc[name, "avg_entropy"] == entropy
        assert ransap.loc[name, "avg_throughput_mbps"] == mbps


@needs_real_datasets
def test_napierone_and_fsl_match_notebook(real_calibration):
    # The notebook prints these rounded: 1161 KB, 844 KB, ratio 0.73.
    assert round(real_calibration.napierone_normal_kb) == 1161
    assert round(real_calibration.napierone_ransomware_kb) == 844
    assert round(real_calibration.size_ratio, 2) == 0.73
    assert real_calibration.fsl_file_count == 30052
    assert len(real_calibration.fsl) == 29
    assert real_calibration.fallbacks == []


@needs_real_datasets
def test_profile_from_real_data(real_calibration):
    p = real_calibration.profile
    assert list(p) == ["change_rate", "file_entropy", "backup_size_gb", "file_count"]
    assert sum(f.weight for f in p.values()) == pytest.approx(1.0)
    assert p["change_rate"].benign_high == 1.1295 / 12.2243
    assert p["change_rate"].infected_low == 1.0086 / 12.2243
    assert (p["file_entropy"].infected_low, p["file_entropy"].infected_high) == (0.3261, 0.7423)
    assert p["file_count"].benign_high == 30052


@needs_real_datasets
def test_score_separation_matches_notebook(real_calibration):
    # Notebook section 4 prints "Benign: 0.276 | Infected: 0.332" for seed 42.
    benign, infected = check_score_separation(real_calibration.profile, seed=42)
    assert (round(benign, 3), round(infected, 3)) == (0.276, 0.332)


def test_fake_tree_calibrates_without_fallbacks(fake_datasets):
    calibration = calibrate(fake_datasets)
    assert calibration.fallbacks == []
    ransap = calibration.ransap.set_index("name")
    # Two rows one second apart, each writing the per-row byte count.
    assert ransap.loc["Excel", "avg_throughput_mbps"] == 2.0
    assert ransap.loc["Ryuk", "avg_throughput_mbps"] == 6.0
    assert ransap.loc["Ryuk", "avg_entropy"] == 0.7
    assert calibration.size_ratio == 0.75
    assert calibration.fsl_file_count == 5000
    profile = calibration.profile
    assert profile["change_rate"].benign_high == pytest.approx(1 / 3)
    assert profile["change_rate"].infected_low == 1.0


def test_zero_duration_trace_has_zero_throughput(fake_datasets):
    write_ransap(fake_datasets, "Firefox", [(5, 0, 0, 1_000_000, 0.4, 0)])
    ransap = calibrate(fake_datasets).ransap.set_index("name")
    assert ransap.loc["Firefox", "avg_throughput_mbps"] == 0.0


def test_missing_ransap_trace_is_an_error(fake_datasets):
    (fake_datasets / "ransap" / "WannaCry.csv").unlink()
    with pytest.raises(DatasetError, match="WannaCry"):
        calibrate(fake_datasets)


def test_ransap_trace_with_missing_values_is_an_error(fake_datasets):
    (fake_datasets / "ransap" / "Excel.csv").write_text("1,2,3,4,5,6\n1,2,,4,5,6\n")
    with pytest.raises(DatasetError, match="missing values"):
        calibrate(fake_datasets)


def test_missing_datasets_dir(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        calibrate(tmp_path / "absent")


def test_napierone_parsing(tmp_path):
    table = tmp_path / "napier.md"
    table.write_text(
        "| File Type | Size (MB) | Files # | Min | Max | Average Size (KB) |\n"
        "|---|---|---|---|---|---|\n"
        "| DOCX | 1 | 5 | 1 | 9 | 1,000 |\n"
        "|      | 1 | 5 | 1 | 9 | 5,000 |  DOCX-NOMAGIC |\n"
        "| PDF  | 1 | 5 | 1 | 9 | 3,000 |\n"
        "| EXTRA | 1 | 5 | 1 | 9 | 900 | RANSOMWARE | RYUK |\n"
        "| JPG  | 1 | 5 | 1 | 9 | 7,000 |\n"
        "| XML  | 1 | 5 | 1 | 9 | n/a |\n"
        "| MAZE |\n",
        encoding="utf-8",
    )
    normal, ransomware = read_napierone(table)
    # Sub-rows with an empty first cell and unknown types are ignored; rows without a
    # numeric size are skipped, as in the notebook.
    assert normal == [1000.0, 3000.0]
    assert ransomware == [900.0]


def test_napierone_fallback(fake_datasets, caplog):
    (fake_datasets / "napierone" / "NapierOne-details.md").unlink()
    calibration = calibrate(fake_datasets)
    assert calibration.size_ratio == 844.0 / 1161.0
    assert len(calibration.fallbacks) == 2
    assert "calibration fallback" in caplog.text


def test_fsl_header_parsing(tmp_path):
    snap = tmp_path / "filesystems" / "fslhomes-user015-2015-04-10"
    snap.mkdir(parents=True)
    (snap / "fslhomes-user015-2015-04-10.8kb.hash.anon").write_bytes(fsl_trace(1234))
    (snap / "fslhomes-user015-2015-04-10.64kb.hash.anon").write_bytes(fsl_trace(999) + b"rest")
    (snap / "short.hash.anon").write_bytes(b"\0" * 10)

    fsl = read_fsl_snapshots(tmp_path).set_index("filename")
    assert fsl.loc["fslhomes-user015-2015-04-10.8kb.hash.anon", "extracted_file_count"] == 1234
    row = fsl.loc["fslhomes-user015-2015-04-10.64kb.hash.anon"]
    assert (row.user_id, row.snapshot_date, row.chunk_size) == ("user015", "2015-04-10", "64kb")
    assert fsl.loc["short.hash.anon", "extracted_file_count"] == 0


def test_fsl_fallback_when_no_traces(fake_datasets):
    for path in fake_datasets.glob("filesystems/**/*.hash.anon"):
        path.unlink()
    calibration = calibrate(fake_datasets)
    assert calibration.fsl_file_count == FSL_FALLBACK_FILE_COUNT
    assert calibration.fsl.empty
    assert any("FSL" in f for f in calibration.fallbacks)


def ransap_table(rows):
    return pd.DataFrame(
        rows, columns=["name", "is_ransomware", "avg_entropy", "avg_throughput_mbps"]
    )


def test_profile_needs_both_classes():
    only_benign = ransap_table([("Excel", False, 0.5, 1.0)])
    with pytest.raises(CalibrationError, match="both"):
        build_param_profile(only_benign, 0.7, 30000)


@pytest.mark.parametrize(
    "mbps, ratio, count, message",
    [(0.0, 0.7, 30000, "throughput"), (1.0, 0.0, 30000, "ratio"), (1.0, 0.7, 150, "file count")],
)
def test_profile_rejects_degenerate_inputs(mbps, ratio, count, message):
    table = ransap_table([("Excel", False, 0.5, mbps), ("Ryuk", True, 0.6, mbps)])
    with pytest.raises(CalibrationError, match=message):
        build_param_profile(table, ratio, count)


def test_feature_profile_bounds_and_normalise():
    f = FeatureProfile(
        0.5, benign_low=0, benign_high=10, infected_low=5, infected_high=20, source=""
    )
    assert f.bounds("benign") == (0, 10)
    assert f.bounds("infected") == (5, 20)
    assert f.normalise(0) == 0.0
    assert f.normalise(20) == 1.0
    with pytest.raises(ValueError):
        f.bounds("compromised")
    flat = FeatureProfile(1.0, 3, 3, 3, 3, "")
    assert flat.normalise(3) == 0.0


def test_weighted_score():
    profile = {
        "a": FeatureProfile(0.25, 0, 1, 0, 1, ""),
        "b": FeatureProfile(0.75, 0, 10, 0, 10, ""),
    }
    assert weighted_score(profile, {"a": 1.0, "b": 5.0}) == pytest.approx(0.25 + 0.375)


def test_score_separation_failure():
    # Infected range below the benign range on the only feature.
    profile = {"x": FeatureProfile(1.0, 5, 10, 0, 1, "")}
    with pytest.raises(CalibrationError, match="does not exceed"):
        check_score_separation(profile, seed=1)
