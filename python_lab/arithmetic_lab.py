"""Python learning prototype for arithmetic experiment design.

This script deliberately favors clarity over benchmark purity. It demonstrates
the research pipeline before low-level kernels move to compiled native code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sqlite3
import statistics
import struct
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Iterable

import native_backend


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_CONFIG_PATH = ROOT.parent / "experiments" / "arithmetic_python.json"
DEFAULT_OPERATIONS = ("empty_loop", "add", "subtract", "multiply", "divide")
DEFAULT_NUMERIC_TYPES = ("float32", "float64")
RELATIVE_ERROR_EPSILON = Decimal("1e-30")
DEFAULT_INPUT_PARAMETERS = {
    "a_low": -1000.0,
    "a_high": 1000.0,
    "abs_b_low": 0.01,
    "abs_b_high": 1000.0,
}


@dataclass(frozen=True)
class Observation:
    experiment_id: str
    run_id: str
    trial_id: int
    timestamp: str
    backend: str
    machine_id: str
    cpu_model: str
    architecture: str
    os: str
    python_version: str
    operation: str
    implementation: str
    numeric_type: str
    input_class: str
    input_parameters: str
    input_set_id: str
    seed: int
    samples_per_trial: int
    passes: int
    measured_evaluations: int
    total_runtime_ns: int
    runtime_per_operation_ns: float
    result_checksum: float
    reference_method: str
    absolute_error: float | None
    relative_error: float | None
    status: str


@dataclass(frozen=True)
class SampleMetric:
    experiment_id: str
    run_id: str
    trial_id: int
    operation: str
    numeric_type: str
    input_class: str
    input_set_id: str
    sample_id: int
    a: float
    b: float
    input_l2_norm: float
    result: float
    reference_result: float
    absolute_error: float
    relative_error: float | None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def machine_profile() -> dict[str, str | int | None]:
    return {
        "timestamp": now_iso(),
        "machine_id": platform.node(),
        "cpu_model": platform.processor() or cpu_name_fallback(),
        "architecture": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "logical_cores": os.cpu_count(),
        "python_version": sys.version.replace("\n", " "),
        "backend": "python_learning_prototype",
        "measurement_warning": (
            "Interpret timing according to the selected backend and methodology."
        ),
    }


def cpu_name_fallback() -> str:
    if platform.system() == "Windows":
        return os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    return "unknown"


def load_experiment_config(path: Path | None) -> dict:
    config = {
        "experiment_id": "arithmetic_python_learning",
        "operations": list(DEFAULT_OPERATIONS),
        "numeric_types": list(DEFAULT_NUMERIC_TYPES),
        "samples": 1_000,
        "trials": 5,
        "passes": 1,
        "seed": 20260912,
        "input_class": "uniform_moderate_magnitude",
        "input_parameters": DEFAULT_INPUT_PARAMETERS,
        "backend": "python",
    }
    if path is not None:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        config.update(loaded)
        if "input_parameters" in loaded:
            merged = dict(DEFAULT_INPUT_PARAMETERS)
            merged.update(loaded["input_parameters"])
            config["input_parameters"] = merged
    return config


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    updated = dict(config)
    if args.samples is not None:
        updated["samples"] = args.samples
    if args.iterations is not None:
        updated["samples"] = args.iterations
    if args.trials is not None:
        updated["trials"] = args.trials
    if args.passes is not None:
        updated["passes"] = args.passes
    if args.seed is not None:
        updated["seed"] = args.seed
    if args.operations is not None:
        updated["operations"] = args.operations
    if args.numeric_types is not None:
        updated["numeric_types"] = args.numeric_types
    if args.backend is not None:
        updated["backend"] = args.backend
    return updated


def to_float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def cast_numeric(value: float, numeric_type: str) -> float:
    if numeric_type == "float32":
        return to_float32(value)
    if numeric_type == "float64":
        return float(value)
    raise ValueError(f"unsupported numeric type: {numeric_type}")


def apply_operation(a: float, b: float, operation: str, numeric_type: str) -> float:
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b
    else:
        raise ValueError(f"unsupported operation: {operation}")
    return cast_numeric(result, numeric_type)


def decimal_reference(a: float, b: float, operation: str) -> Decimal:
    da = Decimal(str(a))
    db = Decimal(str(b))
    if operation == "add":
        return da + db
    if operation == "subtract":
        return da - db
    if operation == "multiply":
        return da * db
    if operation == "divide":
        return da / db
    raise ValueError(f"unsupported operation: {operation}")


def deterministic_input_set(
    numeric_type: str,
    seed: int,
    samples: int,
    input_class: str,
    input_parameters: dict[str, float],
) -> list[tuple[float, float]]:
    rng = random.Random(f"{seed}:shared_arithmetic_inputs:{numeric_type}:{input_class}")
    pairs: list[tuple[float, float]] = []

    for _ in range(samples):
        if input_class == "cancellation":
            base = rng.uniform(input_parameters["base_low"], input_parameters["base_high"])
            if rng.random() < 0.5:
                base = -base
            delta = rng.uniform(input_parameters["delta_low"], input_parameters["delta_high"])
            a = base
            b = -base + delta
        else:
            a = rng.uniform(input_parameters["a_low"], input_parameters["a_high"])
            b = rng.uniform(input_parameters["abs_b_low"], input_parameters["abs_b_high"])
            if rng.random() < 0.5:
                b = -b
        pairs.append((cast_numeric(a, numeric_type), cast_numeric(b, numeric_type)))

    return pairs


def calculate_reference_error(
    checksum: float,
    inputs: list[tuple[float, float]],
    operation: str,
    passes: int = 1,
) -> tuple[float, float | None]:
    getcontext().prec = 50
    reference_sum = Decimal("0")
    for a, b in inputs:
        reference_sum += decimal_reference(a, b, operation)
    reference_sum *= Decimal(passes)

    observed = Decimal(str(checksum))
    absolute = abs(observed - reference_sum)
    if abs(reference_sum) <= RELATIVE_ERROR_EPSILON:
        relative = None
    else:
        relative = absolute / abs(reference_sum)
    return float(absolute), None if relative is None else float(relative)


def calculate_sample_metrics(
    experiment_id: str,
    run_id: str,
    trial_id: int,
    operation: str,
    numeric_type: str,
    input_class: str,
    input_set_id: str,
    inputs: list[tuple[float, float]],
) -> list[SampleMetric]:
    getcontext().prec = 50
    rows: list[SampleMetric] = []
    for sample_id, (a, b) in enumerate(inputs):
        result = apply_operation(a, b, operation, numeric_type)
        reference = decimal_reference(a, b, operation)
        absolute = abs(Decimal(str(result)) - reference)
        if abs(reference) <= RELATIVE_ERROR_EPSILON:
            relative = None
        else:
            relative = absolute / abs(reference)
        rows.append(
            SampleMetric(
                experiment_id=experiment_id,
                run_id=run_id,
                trial_id=trial_id,
                operation=operation,
                numeric_type=numeric_type,
                input_class=input_class,
                input_set_id=input_set_id,
                sample_id=sample_id,
                a=a,
                b=b,
                input_l2_norm=math.hypot(a, b),
                result=result,
                reference_result=float(reference),
                absolute_error=float(absolute),
                relative_error=None if relative is None else float(relative),
            )
        )
    return rows


def run_empty_loop(samples: int) -> tuple[int, float]:
    checksum = 0.0
    start = time.perf_counter_ns()
    for i in range(samples):
        checksum += i & 1
    total_ns = time.perf_counter_ns() - start
    return total_ns, checksum


def run_operation(
    operation: str,
    numeric_type: str,
    inputs: list[tuple[float, float]],
) -> tuple[int, float, float | None, float | None, str]:
    checksum = 0.0

    start = time.perf_counter_ns()
    for a, b in inputs:
        checksum += apply_operation(a, b, operation, numeric_type)
    total_ns = time.perf_counter_ns() - start

    absolute, relative = calculate_reference_error(checksum, inputs, operation)
    return total_ns, checksum, absolute, relative, "decimal_sum_50_digit_precision"


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percent
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize(observations: Iterable[Observation]) -> list[dict[str, float | str | int]]:
    groups: dict[tuple[str, str], list[Observation]] = {}
    for observation in observations:
        key = (observation.operation, observation.numeric_type)
        groups.setdefault(key, []).append(observation)

    rows: list[dict[str, float | str | int]] = []
    for (operation, numeric_type), group in sorted(groups.items()):
        values = [observation.runtime_per_operation_ns for observation in group]
        samples_per_trial = group[0].samples_per_trial
        rows.append(
            {
                "operation": operation,
                "numeric_type": numeric_type,
                "trial_count": len(values),
                "samples_per_trial": samples_per_trial,
                "passes": group[0].passes,
                "measured_evaluations_per_trial": group[0].measured_evaluations,
                "total_measured_evaluations": sum(
                    observation.measured_evaluations for observation in group
                ),
                "mean_ns": statistics.fmean(values),
                "median_ns": statistics.median(values),
                "stdev_ns": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min_ns": min(values),
                "max_ns": max(values),
                "p05_ns": percentile(values, 0.05),
                "p25_ns": percentile(values, 0.25),
                "p75_ns": percentile(values, 0.75),
                "p95_ns": percentile(values, 0.95),
                "p99_ns": percentile(values, 0.99),
            }
        )
    return rows


def init_database(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS observations (
            experiment_id TEXT,
            run_id TEXT,
            trial_id INTEGER,
            timestamp TEXT,
            backend TEXT,
            machine_id TEXT,
            cpu_model TEXT,
            architecture TEXT,
            os TEXT,
            python_version TEXT,
            operation TEXT,
            implementation TEXT,
            numeric_type TEXT,
            input_class TEXT,
            input_parameters TEXT,
            input_set_id TEXT,
            seed INTEGER,
            samples_per_trial INTEGER,
            passes INTEGER,
            measured_evaluations INTEGER,
            total_runtime_ns INTEGER,
            runtime_per_operation_ns REAL,
            result_checksum REAL,
            reference_method TEXT,
            absolute_error REAL,
            relative_error REAL,
            status TEXT
        )
        """
    )
    return conn


def store_observations(conn: sqlite3.Connection, observations: list[Observation]) -> None:
    rows = [asdict(observation) for observation in observations]
    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT INTO observations ({', '.join(columns)}) VALUES ({placeholders})",
        [[row[column] for column in columns] for row in rows],
    )
    conn.commit()


def write_csv(path: Path, observations: list[Observation]) -> None:
    rows = [asdict(observation) for observation in observations]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_sample_metrics_csv(path: Path, sample_metrics: list[SampleMetric]) -> None:
    rows = [asdict(row) for row in sample_metrics]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scaled(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if domain_max == domain_min:
        return (range_min + range_max) / 2.0
    return range_min + ((value - domain_min) / (domain_max - domain_min)) * (range_max - range_min)


def svg_latency_chart(summary_rows: list[dict[str, float | str | int]]) -> str:
    rows = [row for row in summary_rows if row["operation"] != "empty_loop"]
    operations = sorted({str(row["operation"]) for row in rows})
    numeric_types = sorted({str(row["numeric_type"]) for row in rows})
    max_value = max(float(row["median_ns"]) for row in rows)
    width = 900
    height = 360
    left = 74
    right = 24
    top = 34
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / len(operations)
    bar_w = group_w / (len(numeric_types) + 1)
    colors = {"float32": "#4f81bd", "float64": "#c0504d"}

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Median runtime per operation">',
        "<title>Median runtime per operation</title>",
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="currentColor" opacity="0.35"/>',
        f'<text x="{width / 2}" y="18" text-anchor="middle">Median runtime per operation</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 14}" text-anchor="middle">operation</text>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle">ns / sample</text>',
    ]
    for tick in range(5):
        value = max_value * tick / 4
        y = top + plot_h - (value / max_value) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="currentColor" opacity="0.12"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{value:.0f}</text>')
    for op_index, operation in enumerate(operations):
        base_x = left + op_index * group_w
        parts.append(f'<text x="{base_x + group_w / 2:.2f}" y="{height - 42}" text-anchor="middle">{operation}</text>')
        for type_index, numeric_type in enumerate(numeric_types):
            row = next(row for row in rows if row["operation"] == operation and row["numeric_type"] == numeric_type)
            value = float(row["median_ns"])
            bar_h = (value / max_value) * plot_h
            x = base_x + (type_index + 0.5) * bar_w
            y = top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w * 0.72:.2f}" height="{bar_h:.2f}" fill="{colors.get(numeric_type, "#888")}">'
                f'<title>{operation} {numeric_type}: {value:.2f} ns/sample</title></rect>'
            )
            parts.append(f'<text x="{x + bar_w * 0.36:.2f}" y="{y - 4:.2f}" text-anchor="middle">{value:.0f}</text>')
    legend_x = left + plot_w - 160
    for index, numeric_type in enumerate(numeric_types):
        y = 30 + index * 20
        parts.append(f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" fill="{colors.get(numeric_type, "#888")}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{y}">{numeric_type}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_error_scatter(sample_metrics: list[SampleMetric], max_points: int = 900) -> str:
    rows = [
        row
        for row in sample_metrics
        if row.operation != "empty_loop" and row.numeric_type == "float32"
    ]
    if len(rows) > max_points:
        step = math.ceil(len(rows) / max_points)
        rows = rows[::step]
    width = 900
    height = 420
    left = 76
    right = 24
    top = 36
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_x = min(row.input_l2_norm for row in rows)
    max_x = max(row.input_l2_norm for row in rows)
    max_y = max(row.absolute_error for row in rows) or 1.0
    colors = {
        "add": "#4f81bd",
        "subtract": "#9bbb59",
        "multiply": "#8064a2",
        "divide": "#c0504d",
    }
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Input vector size versus absolute error">',
        "<title>Input vector size versus absolute error</title>",
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="currentColor" opacity="0.35"/>',
        f'<text x="{width / 2}" y="18" text-anchor="middle">Input vector size vs absolute error, float32 sample metrics</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 16}" text-anchor="middle">L2 norm of input pair sqrt(a^2 + b^2)</text>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle">absolute error</text>',
    ]
    for tick in range(5):
        y_value = max_y * tick / 4
        y = top + plot_h - (y_value / max_y) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="currentColor" opacity="0.12"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{y_value:.1e}</text>')
    for row in rows:
        x = scaled(row.input_l2_norm, min_x, max_x, left + 6, left + plot_w - 6)
        y = top + plot_h - scaled(row.absolute_error, 0, max_y, 0, plot_h)
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.4" fill="{colors.get(row.operation, "#888")}" opacity="0.55">'
            f'<title>{row.operation} {row.numeric_type}: norm={row.input_l2_norm:.2f}, abs_error={row.absolute_error:.3e}</title></circle>'
        )
    legend_x = left + plot_w - 310
    for index, operation in enumerate(("add", "subtract", "multiply", "divide")):
        x = legend_x + index * 78
        parts.append(f'<circle cx="{x}" cy="32" r="4" fill="{colors[operation]}"/>')
        parts.append(f'<text x="{x + 8}" y="36">{operation}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_relative_error_scatter(sample_metrics: list[SampleMetric], max_points: int = 900) -> str:
    rows = [
        row
        for row in sample_metrics
        if row.operation != "empty_loop"
        and row.numeric_type == "float32"
        and row.relative_error is not None
        and row.relative_error > 0
    ]
    if len(rows) > max_points:
        step = math.ceil(len(rows) / max_points)
        rows = rows[::step]
    width = 900
    height = 420
    left = 76
    right = 24
    top = 36
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_x = min(row.input_l2_norm for row in rows)
    max_x = max(row.input_l2_norm for row in rows)
    min_y = min(row.relative_error for row in rows if row.relative_error is not None)
    max_y = max(row.relative_error for row in rows if row.relative_error is not None)
    min_log_y = math.log10(min_y)
    max_log_y = math.log10(max_y)
    colors = {
        "add": "#4f81bd",
        "subtract": "#9bbb59",
        "multiply": "#8064a2",
        "divide": "#c0504d",
    }
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Input vector size versus relative error">',
        "<title>Input vector size versus relative error</title>",
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="currentColor" opacity="0.35"/>',
        f'<text x="{width / 2}" y="18" text-anchor="middle">Input vector size vs relative error, float32 sample metrics</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 16}" text-anchor="middle">L2 norm of input pair sqrt(a^2 + b^2)</text>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle">relative error, log scale</text>',
    ]
    for tick in range(5):
        log_value = min_log_y + (max_log_y - min_log_y) * tick / 4
        y_value = 10**log_value
        y = top + plot_h - ((log_value - min_log_y) / (max_log_y - min_log_y)) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="currentColor" opacity="0.12"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{y_value:.1e}</text>')
    for row in rows:
        assert row.relative_error is not None
        x = scaled(row.input_l2_norm, min_x, max_x, left + 6, left + plot_w - 6)
        log_value = math.log10(row.relative_error)
        y = top + plot_h - scaled(log_value, min_log_y, max_log_y, 0, plot_h)
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.4" fill="{colors.get(row.operation, "#888")}" opacity="0.55">'
            f'<title>{row.operation} {row.numeric_type}: norm={row.input_l2_norm:.2f}, rel_error={row.relative_error:.3e}</title></circle>'
        )
    legend_x = left + plot_w - 310
    for index, operation in enumerate(("add", "subtract", "multiply", "divide")):
        x = legend_x + index * 78
        parts.append(f'<circle cx="{x}" cy="32" r="4" fill="{colors[operation]}"/>')
        parts.append(f'<text x="{x + 8}" y="36">{operation}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def sample_error_summary(sample_metrics: list[SampleMetric]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    keys = sorted({(row.operation, row.numeric_type) for row in sample_metrics})
    for operation, numeric_type in keys:
        group = [
            row
            for row in sample_metrics
            if row.operation == operation and row.numeric_type == numeric_type
        ]
        absolute_errors = [row.absolute_error for row in group]
        relative_errors = [
            row.relative_error
            for row in group
            if row.relative_error is not None
        ]
        reference_magnitudes = [abs(row.reference_result) for row in group]
        rows.append(
            {
                "operation": operation,
                "numeric_type": numeric_type,
                "sample_count": len(group),
                "median_reference_magnitude": statistics.median(reference_magnitudes),
                "median_absolute_error": statistics.median(absolute_errors),
                "p95_absolute_error": percentile(absolute_errors, 0.95),
                "median_relative_error": statistics.median(relative_errors),
                "p95_relative_error": percentile(relative_errors, 0.95),
            }
        )
    return rows


def html_error_table(sample_metrics: list[SampleMetric]) -> str:
    rows = sample_error_summary(sample_metrics)
    body = "\n".join(
        "<tr>"
        f"<td>{row['operation']}</td>"
        f"<td>{row['numeric_type']}</td>"
        f"<td>{row['sample_count']}</td>"
        f"<td>{row['median_reference_magnitude']:.3e}</td>"
        f"<td>{row['median_absolute_error']:.3e}</td>"
        f"<td>{row['p95_absolute_error']:.3e}</td>"
        f"<td>{row['median_relative_error']:.3e}</td>"
        f"<td>{row['p95_relative_error']:.3e}</td>"
        "</tr>"
        for row in rows
    )
    return f"""
  <h2>Error summary</h2>
  <table>
    <thead>
      <tr>
        <th>operation</th>
        <th>type</th>
        <th>samples</th>
        <th>median |reference|</th>
        <th>median abs error</th>
        <th>p95 abs error</th>
        <th>median rel error</th>
        <th>p95 rel error</th>
      </tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
"""


def write_matplotlib_figures(
    output_dir: Path,
    summary_rows: list[dict[str, float | str | int]],
    sample_metrics: list[SampleMetric],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ModuleNotFoundError as exc:
        missing = exc.name or "matplotlib/seaborn"
        print(
            f"\nSkipping Matplotlib/Seaborn figures: missing {missing}. "
            "Install with: python -m pip install -r python_lab/requirements.txt"
        )
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    figure_paths: list[Path] = []

    latency_rows = pd.DataFrame(
        [row for row in summary_rows if row["operation"] != "empty_loop"]
    )
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.barplot(
        data=latency_rows,
        x="operation",
        y="median_ns",
        hue="numeric_type",
        ax=ax,
    )
    ax.set_title("Median Runtime Per Operation")
    ax.set_xlabel("Operation")
    ax.set_ylabel("Median runtime (ns / eval)")
    path = output_dir / "operation_latency.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path)

    metric_rows = pd.DataFrame(
        [
        asdict(row)
        for row in sample_metrics
        if row.operation != "empty_loop" and row.numeric_type == "float32"
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.scatterplot(
        data=metric_rows,
        x="input_l2_norm",
        y="absolute_error",
        hue="operation",
        alpha=0.45,
        s=18,
        ax=ax,
    )
    ax.set_title("Input Size vs Absolute Error, float32")
    ax.set_xlabel("L2 norm of input pair")
    ax.set_ylabel("Absolute error")
    path = output_dir / "input_norm_vs_absolute_error_float32.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path)

    relative_rows = metric_rows[
        metric_rows["relative_error"].notna() & (metric_rows["relative_error"] > 0)
    ]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.scatterplot(
        data=relative_rows,
        x="input_l2_norm",
        y="relative_error",
        hue="operation",
        alpha=0.45,
        s=18,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("Input Size vs Relative Error, float32")
    ax.set_xlabel("L2 norm of input pair")
    ax.set_ylabel("Relative error, log scale")
    path = output_dir / "input_norm_vs_relative_error_float32.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.scatterplot(
        data=metric_rows,
        x="reference_result",
        y="absolute_error",
        hue="operation",
        alpha=0.45,
        s=18,
        ax=ax,
    )
    ax.set_xscale("symlog")
    ax.set_yscale("symlog", linthresh=1e-13)
    ax.set_title("Result Magnitude vs Absolute Error, float32")
    ax.set_xlabel("Reference result, symmetric log scale")
    ax.set_ylabel("Absolute error, symmetric log scale")
    path = output_dir / "result_magnitude_vs_absolute_error_float32.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figure_paths.append(path)

    return figure_paths


def write_html_report(
    path: Path,
    summary_rows: list[dict[str, float | str | int]],
    sample_metrics: list[SampleMetric],
    profile: dict[str, str | int | None],
    figure_paths: list[Path],
) -> None:
    figure_html = "\n".join(
        f'  <figure><img src="{figure_path.name}" alt="{figure_path.stem}"></figure>'
        for figure_path in figure_paths
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Python Arithmetic Lab Report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; color: #1f2933; }}
    main {{ max-width: 980px; margin: 0 auto; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d9e2ec; }}
    svg {{ width: 100%; height: auto; margin: 20px 0 34px; }}
    text {{ font-size: 13px; fill: currentColor; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 34px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .note {{ color: #52606d; line-height: 1.45; }}
    code {{ background: #eef2f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>Python Arithmetic Lab Report</h1>
  <p class="note">Backend: <code>{profile["backend"]}</code>. {profile["measurement_warning"]}</p>
  {figure_html if figure_html else svg_latency_chart(summary_rows) + svg_error_scatter(sample_metrics) + svg_relative_error_scatter(sample_metrics)}
  {html_error_table(sample_metrics)}
  <p class="note">The scatter plots use per-sample reference calculations. Timing is trial-level and normalized by measured evaluations. Absolute error is magnitude-dependent; relative error is usually the better first comparison across operations with very different result sizes.</p>
</main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_markdown_report(
    path: Path,
    summary_rows: list[dict[str, float | str | int]],
    sample_metrics: list[SampleMetric],
    profile: dict[str, str | int | None],
    figure_paths: list[Path],
) -> None:
    lines = [
        "# Python Arithmetic Lab Report",
        "",
        f"This report describes a `{profile['backend']}` run.",
        "",
        str(profile["measurement_warning"]),
        "",
        "## Machine",
        "",
        f"- machine: `{profile['machine_id']}`",
        f"- CPU: `{profile['cpu_model']}`",
        f"- architecture: `{profile['architecture']}`",
        f"- OS: `{profile['os']}`",
        f"- Python: `{profile['python_version']}`",
        "",
        "## Runtime Summary",
        "",
        "| operation | type | samples/trial | passes | evaluations/trial | trials | median ns/eval | p05 | p95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['operation']} | {row['numeric_type']} | "
            f"{row['samples_per_trial']} | {row['passes']} | "
            f"{row['measured_evaluations_per_trial']} | {row['trial_count']} | "
            f"{float(row['median_ns']):.2f} | {float(row['p05_ns']):.2f} | "
            f"{float(row['p95_ns']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Error Summary",
            "",
            "| operation | type | samples | median abs(reference) | median abs error | p95 abs error | median rel error | p95 rel error |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sample_error_summary(sample_metrics):
        lines.append(
            f"| {row['operation']} | {row['numeric_type']} | "
            f"{row['sample_count']} | "
            f"{float(row['median_reference_magnitude']):.3e} | "
            f"{float(row['median_absolute_error']):.3e} | "
            f"{float(row['p95_absolute_error']):.3e} | "
            f"{float(row['median_relative_error']):.3e} | "
            f"{float(row['p95_relative_error']):.3e} |"
        )

    lines.extend(["", "## Figures", ""])
    if figure_paths:
        for figure_path in figure_paths:
            lines.append(f"![{figure_path.stem}]({figure_path.name})")
            lines.append("")
    else:
        lines.append(
            "Matplotlib/Seaborn figures were not generated because plotting dependencies are missing."
        )
        lines.append("")
        lines.append("Install them with:")
        lines.append("")
        lines.append("```powershell")
        lines.append("python -m pip install -r python_lab/requirements.txt")
        lines.append("```")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Absolute error depends on result magnitude.",
            "- Relative error is the better first comparison across operations with very different output scales.",
            "- Interpret timing values according to the backend and measurement limitations.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(summary_rows: list[dict[str, float | str | int]]) -> None:
    print("\nMedian runtime per operation")
    print("------------------------------------------------------")
    for row in summary_rows:
        print(
            f"{row['operation']:>10} {row['numeric_type']:>7}: "
            f"median={row['median_ns']:.2f} ns/eval, "
            f"p05={row['p05_ns']:.2f}, p95={row['p95_ns']:.2f}, "
            f"samples={row['samples_per_trial']}, "
            f"passes={row['passes']}, "
            f"trials={row['trial_count']}"
        )


def run_experiment(config: dict) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    experiment_id = config["experiment_id"]
    run_id = str(uuid.uuid4())
    profile = machine_profile()
    backend = str(config.get("backend", "python"))
    native_exe: Path | None = None
    native_build_message = None
    native_input_dir = DATA_DIR / "native_inputs" / str(uuid.uuid4())
    if backend == "native":
        native_exe, native_build_message = native_backend.build_native_backend()
        if native_exe is None:
            raise RuntimeError(native_build_message)
        profile["backend"] = "native_cpp"
        profile["native_build"] = native_build_message
        profile["measurement_warning"] = (
            "Native C++ timing excludes Python arithmetic-loop overhead, but still "
            "requires compiler and assembly review before strong CPU-level claims."
        )
    elif backend != "python":
        raise ValueError(f"unsupported backend: {backend}")
    else:
        profile["measurement_warning"] = (
            "Python interpreter overhead dominates primitive arithmetic timings."
        )
    observations: list[Observation] = []
    sample_metrics: list[SampleMetric] = []
    samples = int(config["samples"])
    trials = int(config["trials"])
    passes = int(config["passes"])
    seed = int(config["seed"])
    operations = list(config["operations"])
    numeric_types = list(config["numeric_types"])
    input_class = str(config["input_class"])
    input_parameters = dict(config["input_parameters"])

    for trial_id in range(trials):
        for numeric_type in numeric_types:
            trial_seed = seed + trial_id
            input_set_id = f"seed-{trial_seed}-{numeric_type}-{input_class}"
            inputs = deterministic_input_set(
                numeric_type=numeric_type,
                seed=trial_seed,
                samples=samples,
                input_class=input_class,
                input_parameters=input_parameters,
            )
            native_input_path = native_input_dir / f"{input_set_id}.csv"
            if backend == "native":
                native_backend.write_input_csv(native_input_path, inputs)
            for operation in operations:
                timestamp = now_iso()
                if backend == "native":
                    assert native_exe is not None
                    native_row = native_backend.run_native_trial(
                        exe=native_exe,
                        operation=operation,
                        numeric_type=numeric_type,
                        input_path=native_input_path,
                        experiment_id=experiment_id,
                        input_class=input_class,
                        input_set_id=input_set_id,
                        trial_id=trial_id,
                        seed=trial_seed,
                        passes=passes,
                    )
                    total_ns = int(native_row["total_runtime_ns"])
                    checksum = float(native_row["result_checksum"])
                    reference_method = str(native_row["reference_method"])
                    measured_evaluations = int(native_row["measured_evaluations"])
                    if operation == "empty_loop":
                        absolute_error = None
                        relative_error = None
                    else:
                        absolute_error, relative_error = calculate_reference_error(
                            checksum, inputs, operation, passes=passes
                        )
                else:
                    if operation == "empty_loop":
                        total_ns, checksum = run_empty_loop(samples)
                        measured_evaluations = samples
                        absolute_error = None
                        relative_error = None
                        reference_method = "not_applicable"
                    else:
                        total_ns, checksum, absolute_error, relative_error, reference_method = (
                            run_operation(
                                operation=operation,
                                numeric_type=numeric_type,
                                inputs=inputs,
                            )
                        )
                        measured_evaluations = samples
                if operation != "empty_loop":
                    sample_metrics.extend(
                        calculate_sample_metrics(
                            experiment_id=experiment_id,
                            run_id=run_id,
                            trial_id=trial_id,
                            operation=operation,
                            numeric_type=numeric_type,
                            input_class=input_class,
                            input_set_id=input_set_id,
                            inputs=inputs,
                        )
                    )

                observations.append(
                    Observation(
                        experiment_id=experiment_id,
                        run_id=run_id,
                        trial_id=trial_id,
                        timestamp=timestamp,
                        backend=str(profile["backend"]),
                        machine_id=str(profile["machine_id"]),
                        cpu_model=str(profile["cpu_model"]),
                        architecture=str(profile["architecture"]),
                        os=str(profile["os"]),
                        python_version=str(profile["python_version"]),
                        operation=operation,
                        implementation=(
                            "cpp_builtin_operator"
                            if backend == "native"
                            else "python_builtin_operator"
                        ),
                        numeric_type=numeric_type,
                        input_class=input_class,
                        input_parameters=json.dumps(input_parameters),
                        input_set_id=input_set_id,
                        seed=trial_seed,
                        samples_per_trial=samples,
                        passes=passes if backend == "native" else 1,
                        measured_evaluations=measured_evaluations,
                        total_runtime_ns=total_ns,
                        runtime_per_operation_ns=total_ns / measured_evaluations,
                        result_checksum=checksum,
                        reference_method=reference_method,
                        absolute_error=absolute_error,
                        relative_error=relative_error,
                        status="ok",
                    )
                )

    db_path = DATA_DIR / f"{run_id}.sqlite"
    csv_path = DATA_DIR / f"{run_id}.csv"
    sample_metrics_path = DATA_DIR / f"{run_id}_sample_metrics.csv"
    profile_path = DATA_DIR / f"{run_id}_profile.json"
    summary_path = DATA_DIR / f"{run_id}_summary.json"
    config_snapshot_path = DATA_DIR / f"{run_id}_config.json"
    figures_dir = DATA_DIR / f"{run_id}_figures"
    report_path = figures_dir / "index.html"
    markdown_report_path = figures_dir / "report.md"

    conn = init_database(db_path)
    try:
        store_observations(conn, observations)
    finally:
        conn.close()

    write_csv(csv_path, observations)
    write_sample_metrics_csv(sample_metrics_path, sample_metrics)
    summary_rows = summarize(observations)
    figure_paths = write_matplotlib_figures(figures_dir, summary_rows, sample_metrics)
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    config_snapshot_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_html_report(report_path, summary_rows, sample_metrics, profile, figure_paths)
    write_markdown_report(markdown_report_path, summary_rows, sample_metrics, profile, figure_paths)

    return {
        "config": config,
        "profile": profile,
        "observations": observations,
        "sample_metrics": sample_metrics,
        "summary_rows": summary_rows,
        "db_path": db_path,
        "csv_path": csv_path,
        "sample_metrics_path": sample_metrics_path,
        "profile_path": profile_path,
        "summary_path": summary_path,
        "config_snapshot_path": config_snapshot_path,
        "figures_dir": figures_dir,
        "figure_paths": figure_paths,
        "report_path": report_path,
        "markdown_report_path": markdown_report_path,
        "native_build_message": native_build_message,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a JSON experiment configuration.",
    )
    parser.add_argument("--samples", type=int)
    parser.add_argument(
        "--iterations",
        type=int,
        help="Deprecated alias for --samples.",
    )
    parser.add_argument("--trials", type=int)
    parser.add_argument("--passes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--operations", nargs="+")
    parser.add_argument("--numeric-types", nargs="+")
    parser.add_argument("--backend", choices=["python", "native"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config and args.config.exists() else None
    config = apply_cli_overrides(load_experiment_config(config_path), args)
    try:
        result = run_experiment(config)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    print_summary(result["summary_rows"])
    print("\nWrote:")
    print(f"  SQLite: {result['db_path']}")
    print(f"  CSV:    {result['csv_path']}")
    print(f"  Samples:{result['sample_metrics_path']}")
    print(f"  Profile:{result['profile_path']}")
    print(f"  Stats:  {result['summary_path']}")
    print(f"  Config: {result['config_snapshot_path']}")
    if result["figure_paths"]:
        print(f"  Figures:{result['figures_dir']}")
    print(f"  Report: {result['report_path']}")
    print(f"  Markdown report: {result['markdown_report_path']}")
    print(f"\nReminder: {result['profile']['measurement_warning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
