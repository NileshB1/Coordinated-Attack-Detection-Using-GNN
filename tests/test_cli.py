import os
import subprocess
import sys

import pytest

from cadgnn.cli import main
from conftest import REPO_ROOT


def test_check_passes_on_valid_datasets(fake_datasets, capsys):
    assert main(["--datasets-dir", str(fake_datasets), "check"]) == 0
    assert "RanSAP traces: 9/9" in capsys.readouterr().out


def test_check_fails_on_incomplete_datasets(fake_datasets):
    (fake_datasets / "napierone" / "NapierOne-details.md").unlink()
    assert main(["--datasets-dir", str(fake_datasets), "check"]) == 1


def test_check_fails_on_missing_datasets_dir(tmp_path):
    assert main(["--datasets-dir", str(tmp_path / "absent"), "check"]) == 1


def test_bad_config_exits_with_2(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("seed: -3\n", encoding="utf-8")
    assert main(["--config", str(bad), "check"]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_command_is_required():
    with pytest.raises(SystemExit):
        main([])


def test_module_entry_point_runs():
    result = subprocess.run(
        [sys.executable, "-m", "cadgnn", "--version"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert result.returncode == 0
    assert result.stdout.startswith("cadgnn ")


def small_config(tmp_path, datasets):
    path = tmp_path / "small.yaml"
    path.write_text(
        f"seed: 3\n"
        f"paths:\n  datasets_dir: {datasets.as_posix()}\n  output_dir: out\n"
        f"generator:\n  n_vms: 40\n  n_storage: 3\n  n_network: 4\n",
        encoding="utf-8",
    )
    return path


def test_generate_writes_outputs(tmp_path, fake_datasets, capsys):
    assert main(["--config", str(small_config(tmp_path, fake_datasets)), "generate"]) == 0
    out = tmp_path / "out"
    for name in (
        "synthetic_backup_telemetry.csv",  "infrastructure_graph_nodes.csv",
        "infrastructure_graph_edges.csv", "experiment_manifest.json",
    ):
        assert (out / name).is_file()
    assert "none=240" in capsys.readouterr().out


def test_generate_seed_override_changes_output(tmp_path, fake_datasets):
    config = str(small_config(tmp_path, fake_datasets))
    telemetry = tmp_path / "out" / "synthetic_backup_telemetry.csv"
    main(["--config", config, "generate"])
    first = telemetry.read_text()
    main(["--config", config, "generate"])
    assert telemetry.read_text() == first
    main(["--config", config, "--seed", "4", "generate"])
    assert telemetry.read_text() != first


def test_generate_fails_cleanly_on_missing_trace(tmp_path, fake_datasets):
    (fake_datasets / "ransap" / "Cerber.csv").unlink()
    assert main(["--config", str(small_config(tmp_path, fake_datasets)), "generate"]) == 1
    assert not (tmp_path / "out").exists()


def test_run_generator_script(tmp_path, fake_datasets):
    # Run without PYTHONPATH: the script must find the package on its own.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run_generator.py"), "--config",
         str(small_config(tmp_path, fake_datasets))],
        capture_output=True, text=True,
        cwd=tmp_path,  env=env
    )
    assert result.returncode == 0, result.stderr
    assert "none=240" in result.stdout
    assert (tmp_path / "out" / "synthetic_backup_telemetry.csv").is_file()
