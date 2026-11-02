"""Averages over seeds and a readable comparison table.

The variant marked as selected for a scenario is the one with the best mean
validation F1 (lowest validation loss breaks ties). Test results are shown for
every variant but play no part in that choice.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cadgnn.experiments.spec import ExperimentSpec

METRICS = [
    "val_f1", "best_val_loss", "test_f1", "test_precision", "test_recall", "test_fpr",
    "test_fnr", "test_roc_auc", "test_pr_auc", "detection_rate", "mean_hours_to_detect",
]  # fmt: skip
SUMMARY_FILE = "summary.csv"
REPORT_FILE = "summary.md"


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        raise ValueError("no experiment results to summarise")
    grouped = results.groupby(["scenario", "variant"], sort=False)
    summary = grouped[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["runs"] = grouped.size()
    summary = summary.reset_index()

    summary["selected"] = False
    for _, part in summary.groupby("scenario", sort=False):
        best = part.sort_values(["val_f1_mean", "best_val_loss_mean"], ascending=[False, True])
        summary.loc[best.index[0], "selected"] = True
    return summary


def _cell(mean, std) -> str:
    if pd.isna(mean):
        return "n/a"
    return f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f} ± {std:.3f}"


def render_summary(summary: pd.DataFrame, spec: ExperimentSpec) -> str:
    lines = [
        "# Experiment summary",
        "",
        f"Seeds: {', '.join(map(str, spec.seeds))}. Values are mean ± standard deviation over "
        "seeds. The selected variant (*) per scenario has the best mean validation F1; test "
        "results did not influence the choice.",
        "",
    ]
    for scenario, part in summary.groupby("scenario", sort=False):
        lines += [
            f"## {scenario}",
            "",
            "| Variant | Val F1 | Test F1 | Precision | Recall | FPR | FNR | ROC-AUC | PR-AUC "
            "| Detected | Hours to detect |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, row in part.iterrows():
            name = f"{row['variant']} *" if row["selected"] else row["variant"]
            cells = [
                _cell(row[f"{m}_mean"], row[f"{m}_std"])
                for m in ("val_f1", "test_f1", "test_precision", "test_recall", "test_fpr",
                          "test_fnr", "test_roc_auc", "test_pr_auc", "detection_rate",
                          "mean_hours_to_detect")
            ]  # fmt: skip
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append(
        "All episodes are synthetic. These numbers compare design choices within the "
        "generator's scenarios and do not measure performance on real attack campaigns."
    )
    return "\n".join(lines) + "\n"


def write_summary(
    out_dir: Path, results: pd.DataFrame, summary: pd.DataFrame, spec: ExperimentSpec
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": out_dir / "results.csv",
        "summary": out_dir / SUMMARY_FILE,
        "report": out_dir / REPORT_FILE,
    }
    results.to_csv(paths["results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["report"].write_text(render_summary(summary, spec), encoding="utf-8")
    return paths
