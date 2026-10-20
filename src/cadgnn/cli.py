from __future__ import annotations

import argparse
import logging
import sys

from cadgnn import __version__
from cadgnn.config import Config, ConfigError, describe, load_config, with_overrides
from cadgnn.datasets import DatasetError, inspect_datasets
from cadgnn.logging_utils import setup_logging

log = logging.getLogger("cadgnn")


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(prog="cadgnn")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-c", "--config", help="YAML config file (default: built-in defaults)")
    parser.add_argument("--seed", type=int, help="override the random seed")
    parser.add_argument("--datasets-dir", help="override paths.datasets_dir")
    parser.add_argument("--output-dir", help="override paths.output_dir")
    parser.add_argument("--log-level", help="override log_level")
    parser.add_argument("--log-file", help="also write logs to this file")

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="validate the configuration and the local datasets")
    return parser


def cmd_check(config: Config) -> int:
    inventory = inspect_datasets(config.paths.datasets_dir)
    print(inventory.summary())
    if not inventory.ok:
        log.error("dataset check found %d problem(s)", len(inventory.problems))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = with_overrides(
            load_config(args.config),
            seed=args.seed, log_level=args.log_level,
            datasets_dir=args.datasets_dir,
            output_dir=args.output_dir,
        )
    except ConfigError as exc:
        print(f"cadgnn: configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(config.log_level, args.log_file)
    log.info("configuration: %s", describe(config))

    try:
        if args.command == "check":
            return cmd_check(config)
    except DatasetError as exc:
        log.error("%s", exc)
        return 1
    raise AssertionError(f"unhandled command {args.command!r}")
