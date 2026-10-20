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
