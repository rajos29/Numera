"""One-command Python benchmark learning suite.

This aggregates several arithmetic experiment configurations into one report.
The suite is still a Python learning prototype, not a native CPU benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
import uuid
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
SUITE_DIR = DATA_DIR / "suites"
DEFAULT_CONFIGS = [
    PROJECT_ROOT / "experiments" / "arithmetic_python.json",
    PROJECT_ROOT / "experiments" / "arithmetic_small_magnitude.json",
    PROJECT_ROOT / "experiments" / "arithmetic_large_magnitude.json",
    PROJECT_ROOT / "experiments" / "arithmetic_cancellation.json",
    PROJECT_ROOT / "experiments" / "arithmetic_denominator_sensitivity.json",
]

sys.path.insert(0, str(ROOT))
import arithmetic_lab


def write_dict_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def error_summary(sample_rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in sample_rows:
        key = (row["input_class"], row["operation"], row["numeric_type"])
        groups.setdefault(key, []).append(row)

    summaries: list[dict] = []
    for (input_class, operation, numeric_type), group in sorted(groups.items()):
        absolute_errors = [float(row["absolute_error"]) for row in group]
        relative_errors = [
            float(row["relative_error"])
            for row in group
            if row["relative_error"] is not None
        ]
        reference_magnitudes = [abs(float(row["reference_result"])) for row in group]
        summaries.append(
            {
                "input_class": input_class,
                "operation": operation,
                "numeric_type": numeric_type,
                "sample_count": len(group),
                "median_reference_magnitude": statistics.median(reference_magnitudes),
                "median_absolute_error": statistics.median(absolute_errors),
                "p95_absolute_error": arithmetic_lab.percentile(absolute_errors, 0.95),
                "median_relative_error": statistics.median(relative_errors),
                "p95_relative_error": arithmetic_lab.percentile(relative_errors, 0.95),
            }
        )
    return summaries


def write_suite_figures(suite_dir: Path, summary_rows: list[dict], sample_rows: list[dict]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ModuleNotFoundError as exc:
        print(f"Skipping suite figures: missing {exc.name}.")
        return []

    figures_dir = suite_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    paths: list[Path] = []

    latency = pd.DataFrame(
        [row for row in summary_rows if row["operation"] != "empty_loop"]
    )
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    sns.barplot(
        data=latency,
        x="input_class",
        y="median_ns",
        hue="operation",
        errorbar=None,
        ax=ax,
    )
    ax.set_title("Median Python Runtime By Experiment Config")
    ax.set_xlabel("Experiment input class")
    ax.set_ylabel("Median runtime (ns / eval)")
    ax.tick_params(axis="x", rotation=24)
    path = figures_dir / "suite_latency_by_config.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    errors = pd.DataFrame(error_summary(sample_rows))
    float32_errors = errors[errors["numeric_type"] == "float32"]
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    sns.barplot(
        data=float32_errors,
        x="input_class",
        y="median_relative_error",
        hue="operation",
        errorbar=None,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("Median Relative Error By Experiment Config, float32")
    ax.set_xlabel("Experiment input class")
    ax.set_ylabel("Median relative error, log scale")
    ax.tick_params(axis="x", rotation=24)
    path = figures_dir / "suite_relative_error_float32.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    samples = pd.DataFrame(
        [
            row
            for row in sample_rows
            if row["numeric_type"] == "float32"
            and row["operation"] in {"add", "subtract", "multiply", "divide"}
        ]
    )
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    sns.scatterplot(
        data=samples,
        x="reference_result",
        y="absolute_error",
        hue="operation",
        style="input_class",
        alpha=0.38,
        s=18,
        ax=ax,
    )
    ax.set_xscale("symlog")
    ax.set_yscale("symlog", linthresh=1e-13)
    ax.set_title("Result Scale vs Absolute Error Across Configs, float32")
    ax.set_xlabel("Reference result, symmetric log scale")
    ax.set_ylabel("Absolute error, symmetric log scale")
    path = figures_dir / "suite_result_scale_vs_absolute_error_float32.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths


def interpretation_for_config(input_class: str) -> str:
    interpretations = {
        "uniform_moderate_magnitude": (
            "This is the baseline sanity check. It shows ordinary moderate inputs "
            "where multiplication tends to have larger absolute error because it "
            "often produces larger result magnitudes."
        ),
        "uniform_small_magnitude": (
            "This checks behavior near zero without intentionally invalid division. "
            "Relative error becomes more important here because absolute values are tiny."
        ),
        "uniform_large_magnitude": (
            "This increases input and result scale. If absolute error grows while "
            "relative error stays similar, that is expected floating-point behavior."
        ),
        "cancellation": (
            "This is designed to challenge addition and subtraction. Nearly equal "
            "large values can produce small results, making relative error fragile."
        ),
        "small_denominator": (
            "This stresses division by using small nonzero denominators. Large quotient "
            "magnitudes can make absolute error grow and may reveal sensitivity."
        ),
    }
    return interpretations.get(input_class, "No specific interpretation has been written for this input class.")


def observed_highlights(error_rows: list[dict]) -> list[str]:
    highlights: list[str] = []
    input_classes = sorted({row["input_class"] for row in error_rows})
    for input_class in input_classes:
        float32_rows = [
            row
            for row in error_rows
            if row["input_class"] == input_class and row["numeric_type"] == "float32"
        ]
        if not float32_rows:
            continue
        largest_abs = max(float32_rows, key=lambda row: float(row["median_absolute_error"]))
        largest_rel = max(float32_rows, key=lambda row: float(row["median_relative_error"]))
        highlights.append(
            f"- `{input_class}`: largest median absolute error is `{largest_abs['operation']}` "
            f"({float(largest_abs['median_absolute_error']):.3e}); largest median relative error is "
            f"`{largest_rel['operation']}` ({float(largest_rel['median_relative_error']):.3e})."
        )
    return highlights


def write_suite_report(
    path: Path,
    configs: list[dict],
    summary_rows: list[dict],
    error_rows: list[dict],
    figure_paths: list[Path],
) -> None:
    backend = str(configs[0].get("backend", "python"))
    if backend == "native":
        timing_note = (
            "This suite used the native C++ backend. Timing values exclude Python "
            "arithmetic-loop overhead, but still need compiler and assembly validation "
            "before strong CPU-level claims."
        )
    else:
        timing_note = (
            "This is still a Python learning prototype. Timing values include Python "
            "interpreter overhead and should not be treated as native CPU arithmetic timings."
        )
    lines = [
        "# Arithmetic Python Learning Suite Report",
        "",
        "This suite aggregates multiple experiment configurations into one cohesive view.",
        "",
        timing_note,
        "",
        "## Configurations",
        "",
    ]
    for config in configs:
        lines.extend(
            [
                f"### {config['input_class']}",
                "",
                f"Question: {config.get('question', 'No question supplied.')}",
                "",
                interpretation_for_config(config["input_class"]),
                "",
                f"Operations: `{', '.join(config['operations'])}`",
                "",
                f"Numeric types: `{', '.join(config['numeric_types'])}`",
                "",
                f"Samples per trial: `{config['samples']}`; passes: `{config.get('passes', 1)}`; trials: `{config['trials']}`; seed: `{config['seed']}`",
                "",
            ]
        )

    lines.extend(["## Observed Highlights", ""])
    lines.extend(observed_highlights(error_rows))
    lines.append("")
    lines.extend(
        [
            "These highlights are descriptive, not claims of native arithmetic cost. They are mainly used to check whether the experiment is revealing the expected floating-point behavior.",
            "",
        ]
    )

    lines.extend(
        [
            "## Runtime Summary",
            "",
            "| input class | operation | type | samples/trial | passes | evaluations/trial | trials | median ns/eval | p05 | p95 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['input_class']} | {row['operation']} | {row['numeric_type']} | "
            f"{row['samples_per_trial']} | {row['passes']} | "
            f"{row['measured_evaluations_per_trial']} | {row['trial_count']} | "
            f"{float(row['median_ns']):.2f} | {float(row['p05_ns']):.2f} | {float(row['p95_ns']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Error Summary",
            "",
            "| input class | operation | type | samples | median abs(reference) | median abs error | p95 abs error | median rel error | p95 rel error |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in error_rows:
        lines.append(
            f"| {row['input_class']} | {row['operation']} | {row['numeric_type']} | "
            f"{row['sample_count']} | {float(row['median_reference_magnitude']):.3e} | "
            f"{float(row['median_absolute_error']):.3e} | {float(row['p95_absolute_error']):.3e} | "
            f"{float(row['median_relative_error']):.3e} | {float(row['p95_relative_error']):.3e} |"
        )

    lines.extend(["", "## Figures", ""])
    if figure_paths:
        for figure_path in figure_paths:
            rel = figure_path.relative_to(path.parent)
            lines.append(f"![{figure_path.stem}]({rel.as_posix()})")
            lines.append("")
    else:
        lines.append("Figures were not generated because plotting dependencies are missing.")
        lines.append("")

    lines.extend(
        [
            "## What This Suite Is Teaching",
            "",
            "- Absolute error grows with result scale, so it must be interpreted with result magnitude.",
            "- Relative error is better for comparing operations whose outputs live on very different scales.",
            "- Cancellation experiments are expected to make addition/subtraction less boring.",
            "- Small-denominator experiments are expected to make division behavior easier to inspect.",
            "- The Python runtime is still too noisy and high-level for native timing conclusions.",
            "",
            "## Next Step",
            "",
            "Inspect whether these configurations reproduce known floating-point behavior clearly. If they do, the same config/report pattern can be reused when the arithmetic kernels move to native C or C++.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--samples", type=int, help="Override samples for all configs.")
    parser.add_argument("--trials", type=int, help="Override trials for all configs.")
    parser.add_argument("--passes", type=int, help="Override native passes for all configs.")
    parser.add_argument("--backend", choices=["python", "native"], default="python")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suite_id = str(uuid.uuid4())
    suite_dir = SUITE_DIR / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    configs: list[dict] = []
    all_observations: list[dict] = []
    all_sample_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    run_outputs: list[dict] = []

    for config_path in args.configs:
        config = arithmetic_lab.load_experiment_config(config_path)
        if args.samples is not None:
            config["samples"] = args.samples
        if args.trials is not None:
            config["trials"] = args.trials
        if args.passes is not None:
            config["passes"] = args.passes
        config["backend"] = args.backend
        print(f"Running {config['input_class']}...")
        try:
            result = arithmetic_lab.run_experiment(config)
        except RuntimeError as exc:
            print(f"Error while running {config['input_class']}: {exc}")
            return 1
        configs.append(config)
        run_outputs.append(
            {
                "input_class": config["input_class"],
                "summary_path": str(result["summary_path"]),
                "sample_metrics_path": str(result["sample_metrics_path"]),
                "report_path": str(result["markdown_report_path"]),
            }
        )
        observation_rows = [asdict(row) for row in result["observations"]]
        sample_rows = [asdict(row) for row in result["sample_metrics"]]
        summary_rows = [dict(row, input_class=config["input_class"]) for row in result["summary_rows"]]
        all_observations.extend(observation_rows)
        all_sample_rows.extend(sample_rows)
        all_summary_rows.extend(summary_rows)

    raw_dir = suite_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_dict_csv(raw_dir / "observations.csv", all_observations)
    write_dict_csv(raw_dir / "sample_metrics.csv", all_sample_rows)
    write_dict_csv(raw_dir / "summary.csv", all_summary_rows)
    error_rows = error_summary(all_sample_rows)
    write_dict_csv(raw_dir / "error_summary.csv", error_rows)
    (suite_dir / "suite_config.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")
    (suite_dir / "run_outputs.json").write_text(json.dumps(run_outputs, indent=2), encoding="utf-8")

    figure_paths = write_suite_figures(suite_dir, all_summary_rows, all_sample_rows)
    report_path = suite_dir / "suite_report.md"
    write_suite_report(report_path, configs, all_summary_rows, error_rows, figure_paths)
    shutil.copyfile(report_path, DATA_DIR / "latest_suite_report.md")

    print("\nSuite complete.")
    print(f"  Suite directory: {suite_dir}")
    print(f"  Report: {report_path}")
    print(f"  Raw aggregate CSVs: {raw_dir}")
    if figure_paths:
        print(f"  Figures: {suite_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
