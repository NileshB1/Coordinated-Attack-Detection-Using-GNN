"""Run configuration loaded from YAML.

Relative paths in a config file are resolved against the file's own directory,
so a config behaves the same regardless of the working directory. 
If there is no config file then the defaults are resolved against the working directory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PathsConfig:
    datasets_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class Config:
    seed: int
    log_level: str
    paths: PathsConfig


DEFAULTS: dict[str, Any] = {
    "seed": 8304,
    "log_level": "INFO",
    "paths": {
        "datasets_dir": "datasets",
        "output_dir": "output",
    },
}


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return parse_config({}, base_dir=Path.cwd())

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    return parse_config(raw, base_dir=path.resolve().parent)


def parse_config(raw: dict[str, Any], base_dir: Path) -> Config:
    data = _merge(DEFAULTS, raw, where="config")
    paths = data["paths"]
    return Config(
        seed=_seed(data["seed"]),
        log_level=_log_level(data["log_level"]),
        paths=PathsConfig(
            datasets_dir=_path(paths["datasets_dir"], base_dir, "paths.datasets_dir"),
            output_dir=_path(paths["output_dir"], base_dir, "paths.output_dir"),
        ),
    )


def with_overrides(
    config: Config,
    *,
    seed: int | None = None,  log_level: str | None = None,
    datasets_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Config:
    """Apply command-line overrides; override paths are relative to the working directory."""
    paths = config.paths
    if datasets_dir is not None:
        paths = replace(paths, datasets_dir=_path(datasets_dir, Path.cwd(), "datasets_dir"))
    if output_dir is not None:
        paths = replace(paths, output_dir=_path(output_dir, Path.cwd(), "output_dir"))
    return replace(
        config,
        seed=config.seed if seed is None else _seed(seed),
        log_level=config.log_level if log_level is None else _log_level(log_level),
        paths=paths,
    )


def _merge(defaults: dict[str, Any], raw: dict[str, Any], where: str) -> dict[str, Any]:
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {', '.join(map(str, unknown))}")

    merged = {}
    for key, default in defaults.items():
        value = raw.get(key, default)
        if isinstance(default, dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{key} must be a mapping")
            value = _merge(default, value, where=key)
        merged[key] = value
    return merged


def _seed(value: Any) -> int:
    # bool is a subclass of int, but `seed: true` is almost certainly a mistake.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"seed must be an integer, got {value!r}")
    if value < 0:
        raise ConfigError(f"seed must be non-negative, got {value}")
    return value


def _log_level(value: Any) -> str:
    if not isinstance(value, str) or value.upper() not in LOG_LEVELS:
        raise ConfigError(f"log_level must be one of {', '.join(LOG_LEVELS)}, got {value!r}")
    return value.upper()


def _path(value: Any, base_dir: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigError(f"{name} must be a non-empty path, got {value!r}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def describe(config: Config) -> str:
    return (
        f"seed={config.seed} log_level={config.log_level} "
        f"datasets_dir={config.paths.datasets_dir} output_dir={config.paths.output_dir}"
    )
