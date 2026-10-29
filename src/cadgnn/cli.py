from __future__ import annotations

import argparse
import logging
import sys
import warnings

from cadgnn import __version__
from cadgnn.config import Config, ConfigError, describe, load_config, with_overrides
from cadgnn.data.calibration import CalibrationError, calibrate, check_score_separation
from cadgnn.data.datasets import DatasetError, inspect_datasets
from cadgnn.generator import generate, save_outputs
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
    commands.add_parser(
        "generate", help="calibrate from the datasets and write synthetic backup telemetry"
    )
    commands.add_parser(
        "build-graphs", help="turn the generator output into one graph per anomaly episode"
    )
    commands.add_parser("train", help="train the coordination model and pick the threshold")
    commands.add_parser("evaluate", help="score the held-out test episodes and write the report")
    return parser


def cmd_check(config: Config) -> int:
    inventory = inspect_datasets(config.paths.datasets_dir)
    print(inventory.summary())
    if not inventory.ok:
        log.error("dataset check found %d problem(s)", len(inventory.problems))
        return 1
    return 0


def cmd_generate(config: Config) -> int:
    calibration = calibrate(config.paths.datasets_dir)
    benign, infected = check_score_separation(calibration.profile, config.seed)
    log.info("mean weighted score: benign %.3f, infected %.3f", benign, infected)

    dataset = generate(config.generator, calibration.profile, config.seed)
    paths = save_outputs(dataset, calibration, config.paths.output_dir)

    counts = dataset.events["label"].value_counts()
    print(f"{len(dataset.events)} events: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    for path in paths.values():
        print(f"  wrote {path}")
    return 0


def cmd_build_graphs(config: Config) -> int:
    # Imported here so check and generate don't need torch. torch_geometric triggers
    # a torch.jit deprecation warning on import that we can't act on.
    warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated")
    from cadgnn.graph import (
        GraphConstructionError, build_episode_graphs,
        episode_summary, load_generator_output,
        save_episode_graphs
    )
    from cadgnn.graph.episodes import EPISODE_GRAPHS_FILE, EPISODE_SUMMARY_FILE

    output_dir = config.paths.output_dir
    try:
        infra, events = load_generator_output(output_dir)
        graphs = build_episode_graphs(infra, events, config.graph.min_episode_events)
    except GraphConstructionError as exc:
        log.error("%s", exc)
        return 1

    save_episode_graphs(graphs, output_dir / EPISODE_GRAPHS_FILE)
    summary = episode_summary(graphs)
    summary.to_csv(output_dir / EPISODE_SUMMARY_FILE, index=False)

    print(f"{len(graphs)} episode graphs from {len(events)} events")
    if graphs:
        per_label = summary.groupby("y").agg(
            episodes=("episode_id", "size"),
            mean_vms=("n_vm", "mean"),  mean_networks=("n_network", "mean"),
            mean_components=("n_components", "mean"), mean_span_hours=("time_span_hours", "mean")
        )
        per_label = per_label.rename(index={0: "coincidental (0)", 1: "coordinated (1)"})
        print(per_label.round(2).to_string())
    print(f"  wrote {output_dir / EPISODE_GRAPHS_FILE}")
    print(f"  wrote {output_dir / EPISODE_SUMMARY_FILE}")
    return 0


def cmd_train(config: Config) -> int:
    warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated")
    from cadgnn.training import run_training

    try:
        summary = run_training(config)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    sizes, val = summary["split_sizes"], summary["val_metrics"]
    print(f"split: {sizes['train']} train, {sizes['val']} validation, {sizes['test']} test")
    print(f"trained {summary['epochs_run']} epochs, best at epoch {summary['best_epoch']}")
    print(
        f"threshold {summary['threshold']:.4f} (validation F1 {val['f1']:.3f}, "
        f"FPR {val['false_positive_rate']:.3f})"
    )
    print(f"  wrote {summary['checkpoint']}")
    return 0


def cmd_evaluate(config: Config) -> int:
    warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated")
    from cadgnn.evaluation.run import run_evaluation

    try:
        evaluation = run_evaluation(config)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    test = evaluation["results"]["test_metrics"]
    cm = test["confusion_matrix"]

    def fmt(value):
        return "n/a" if value is None else f"{value:.3f}"

    print(f"test episodes: {test['n_episodes']} ({test['n_coordinated']} coordinated)")
    print(
        f"precision {fmt(test['precision'])}  recall {fmt(test['recall'])}  f1 {fmt(test['f1'])}  "
        f"FPR {fmt(test['false_positive_rate'])}  FNR {fmt(test['false_negative_rate'])}"
    )
    print(f"ROC-AUC {fmt(test['roc_auc'])}  PR-AUC {fmt(test['pr_auc'])}")
    print(f"confusion matrix: tp={cm['tp']} fp={cm['fp']} tn={cm['tn']} fn={cm['fn']}")
    for path in evaluation["paths"].values():
        print(f"  wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = with_overrides(
            load_config(args.config),  seed=args.seed, log_level=args.log_level,
            datasets_dir=args.datasets_dir, output_dir=args.output_dir
        )
    except ConfigError as exc:
        print(f"cadgnn: configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(config.log_level, args.log_file)
    log.info("configuration: %s", describe(config))

    try:
        if args.command == "check":
            return cmd_check(config)
        if args.command == "generate":
            return cmd_generate(config)
        if args.command == "build-graphs":
            return cmd_build_graphs(config)
        if args.command == "train":
            return cmd_train(config)
        if args.command == "evaluate":
            return cmd_evaluate(config)
    except (DatasetError, CalibrationError) as exc:
        log.error("%s", exc)
        return 1
    raise AssertionError(f"unhandled command {args.command!r}")
