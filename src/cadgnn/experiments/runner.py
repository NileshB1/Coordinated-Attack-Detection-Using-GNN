"""Run every variant on every scenario and seed, then summarise.

For each scenario and seed the data is generated and turned into episode graphs
once; each variant is then trained and evaluated on those same graphs with its
own split seed, checkpoint and report under ``<out>/<scenario>/seed_<n>/<variant>``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cadgnn.config import Config
from cadgnn.data.calibration import calibrate
from cadgnn.evaluation.run import run_evaluation
from cadgnn.experiments.spec import ExperimentSpec, scenario_config, variant_config
from cadgnn.experiments.summary import summarise, write_summary
from cadgnn.generator import generate, save_outputs
from cadgnn.graph import (
    InfrastructureGraph,
    build_episode_graphs,
    clean_events,
    save_episode_graphs,
)
from cadgnn.graph.episodes import EPISODE_GRAPHS_FILE
from cadgnn.training.run import run_training

log = logging.getLogger(__name__)

RESULTS_FILE = "results.csv"


def run_experiment(base: Config, spec: ExperimentSpec, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    calibration = calibrate(base.paths.datasets_dir)
    rows = []
    for scenario in spec.scenarios:
        for seed in spec.seeds:
            generator_config = scenario_config(base.generator, scenario, seed)
            data_dir = out_dir / scenario.name / f"seed_{seed}"
            dataset = generate(generator_config, calibration.profile, seed)
            save_outputs(dataset, calibration, data_dir)

            infra = InfrastructureGraph.from_networkx(dataset.graph)
            graphs = build_episode_graphs(
                infra, clean_events(dataset.events, infra), base.graph.min_episode_events
            )
            graphs_file = data_dir / EPISODE_GRAPHS_FILE
            save_episode_graphs(graphs, graphs_file)

            for variant in spec.variants:
                log.info("running %s / seed %d / %s", scenario.name, seed, variant.name)
                config = variant_config(base, variant, seed)
                run_dir = data_dir / variant.name
                training = run_training(config, graphs_file, run_dir)
                evaluation = run_evaluation(config, graphs_file, run_dir)["results"]
                names = (scenario.name, seed, variant.name)
                rows.append(result_row(*names, generator_config, training, evaluation))
                # Write as we go so a long run that stops halfway still leaves results.
                pd.DataFrame(rows).to_csv(out_dir / RESULTS_FILE, index=False)

    results = pd.DataFrame(rows)
    summary = summarise(results)
    paths = write_summary(out_dir, results, summary, spec)
    return {"results": results, "summary": summary, "paths": paths}


def result_row(scenario, seed, variant, generator_config, training, evaluation) -> dict:
    test = evaluation["test_metrics"]
    val = training["val_metrics"]
    detection = evaluation["time_to_detect"]
    generator = asdict(generator_config)
    return {
        "scenario": scenario,
        "seed": seed,
        "variant": variant,
        "n_network": generator["n_network"],
        "n_episodes": sum(training["split_sizes"].values()),
        "best_epoch": training["best_epoch"],
        "best_val_loss": training["best_val_loss"],
        "threshold": training["threshold"],
        "val_f1": val["f1"],
        "val_fpr": val["false_positive_rate"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_f1": test["f1"],
        "test_fpr": test["false_positive_rate"],
        "test_fnr": test["false_negative_rate"],
        "test_balanced_accuracy": test["balanced_accuracy"],
        "test_roc_auc": test["roc_auc"],
        "test_pr_auc": test["pr_auc"],
        "detection_rate": detection["detection_rate"],
        "mean_hours_to_detect": detection["mean_hours_to_detect"],
        "generator": json.dumps(generator),
    }
