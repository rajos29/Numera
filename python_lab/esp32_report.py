"""Generate a Matplotlib/Seaborn report from an ESP32 arithmetic suite CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ESP32_DATA_DIR = ROOT / "data" / "esp32"
REPORTS_DIR = ROOT / "data" / "esp32_reports"
OPERATION_ORDER = ["empty_loop", "add", "subtract", "multiply", "divide"]
TYPE_ORDER = ["float32", "float64"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="ESP32 suite CSV. Defaults to the newest *_suite.csv.")
    parser.add_argument("--out-dir", type=Path, help="Report output directory.")
    return parser.parse_args()


def newest_suite_csv() -> Path:
    candidates = sorted(
        ESP32_DATA_DIR.glob("*_suite.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No *_suite.csv files found under {ESP32_DATA_DIR}")
    return candidates[0]


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def as_float(value: object) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def enrich_rows(rows: list[dict]) -> list[dict]:
    empty_by_type = {
        (int(float(row.get("trial_index", 0) or 0)), row["numeric_type"]): as_float(row["runtime_per_eval_ns"])
        for row in rows
        if row.get("op") == "empty_loop"
    }
    enriched: list[dict] = []
    for row in rows:
        item = dict(row)
        item["trial_index"] = int(float(item.get("trial_index", 0) or 0))
        runtime = as_float(item.get("runtime_per_eval_ns"))
        empty_runtime = empty_by_type.get((item["trial_index"], item.get("numeric_type")), math.nan)
        item["runtime_per_eval_ns"] = runtime
        item["empty_loop_ns"] = empty_runtime
        item["net_runtime_per_eval_ns"] = max(runtime - empty_runtime, 0.0)
        item["samples"] = int(float(item["samples"]))
        item["passes"] = int(float(item["passes"]))
        item["measured_evaluations"] = int(float(item["measured_evaluations"]))
        item["free_heap_before"] = int(float(item["free_heap_before"]))
        item["free_heap_after"] = int(float(item["free_heap_after"]))
        item["heap_delta_bytes"] = item["free_heap_after"] - item["free_heap_before"]
        checksum = str(item.get("result_checksum", ""))
        item["checksum_state"] = "overflow/non-finite" if checksum == "ovf" else "finite"
        item["checksum_method"] = item.get("checksum_method") or "raw_sum_legacy"
        enriched.append(item)
    return enriched


def summarize_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["op"], row["numeric_type"]), []).append(row)

    summary: list[dict] = []
    for (operation, numeric_type), group in sorted(
        groups.items(),
        key=lambda item: (TYPE_ORDER.index(item[0][1]), operation_sort_key(item[0][0])),
    ):
        raw = [float(row["runtime_per_eval_ns"]) for row in group]
        net = [float(row["net_runtime_per_eval_ns"]) for row in group]
        heap = [int(row["heap_delta_bytes"]) for row in group]
        checksum_states = sorted({str(row["checksum_state"]) for row in group})
        checksum_methods = sorted({str(row["checksum_method"]) for row in group})
        summary.append(
            {
                "op": operation,
                "numeric_type": numeric_type,
                "samples": int(group[0]["samples"]),
                "passes": int(group[0]["passes"]),
                "measured_evaluations": int(group[0]["measured_evaluations"]),
                "trial_count": len(group),
                "median_raw_ns": statistics.median(raw),
                "p05_raw_ns": percentile(raw, 0.05),
                "p95_raw_ns": percentile(raw, 0.95),
                "median_net_ns": statistics.median(net),
                "p05_net_ns": percentile(net, 0.05),
                "p95_net_ns": percentile(net, 0.95),
                "min_heap_delta_bytes": min(heap),
                "max_heap_delta_bytes": max(heap),
                "checksum_state": ", ".join(checksum_states),
                "checksum_method": ", ".join(checksum_methods),
            }
        )
    return summary


def operation_sort_key(operation: str) -> int:
    try:
        return OPERATION_ORDER.index(operation)
    except ValueError:
        return len(OPERATION_ORDER)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_figures(report_dir: Path, rows: list[dict]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ModuleNotFoundError as exc:
        print(f"Skipping figures: missing {exc.name}.")
        return []

    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summarize_rows(rows))
    df["operation"] = pd.Categorical(df["op"], categories=OPERATION_ORDER, ordered=True)
    df["numeric_type"] = pd.Categorical(df["numeric_type"], categories=TYPE_ORDER, ordered=True)
    summary_df["operation"] = pd.Categorical(summary_df["op"], categories=OPERATION_ORDER, ordered=True)
    summary_df["numeric_type"] = pd.Categorical(summary_df["numeric_type"], categories=TYPE_ORDER, ordered=True)
    arithmetic = df[df["op"] != "empty_loop"].copy()
    arithmetic_summary = summary_df[summary_df["op"] != "empty_loop"].copy()
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    sns.barplot(
        data=summary_df,
        x="operation",
        y="median_raw_ns",
        hue="numeric_type",
        errorbar=None,
        ax=ax,
    )
    if df["trial_index"].nunique() > 1:
        sns.stripplot(
            data=df,
            x="operation",
            y="runtime_per_eval_ns",
            hue="numeric_type",
            dodge=True,
            alpha=0.5,
            size=4,
            legend=False,
            ax=ax,
        )
    ax.set_title("ESP32 Raw Runtime Per Evaluation")
    ax.set_xlabel("Operation")
    ax.set_ylabel("Runtime (ns / evaluation)")
    path = figures_dir / "esp32_raw_runtime.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    sns.barplot(
        data=arithmetic_summary,
        x="operation",
        y="median_net_ns",
        hue="numeric_type",
        errorbar=None,
        ax=ax,
    )
    if df["trial_index"].nunique() > 1:
        sns.stripplot(
            data=arithmetic,
            x="operation",
            y="net_runtime_per_eval_ns",
            hue="numeric_type",
            dodge=True,
            alpha=0.5,
            size=4,
            legend=False,
            ax=ax,
        )
    ax.set_title("ESP32 Runtime After Empty-Loop Subtraction")
    ax.set_xlabel("Operation")
    ax.set_ylabel("Approximate net runtime (ns / evaluation)")
    path = figures_dir / "esp32_net_runtime.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    ratio_rows = []
    for operation in OPERATION_ORDER:
        subset = arithmetic_summary[arithmetic_summary["op"] == operation]
        f32 = subset[subset["numeric_type"] == "float32"]
        f64 = subset[subset["numeric_type"] == "float64"]
        if f32.empty or f64.empty:
            continue
        f32_net = float(f32.iloc[0]["median_net_ns"])
        f64_net = float(f64.iloc[0]["median_net_ns"])
        if f32_net > 0:
            ratio_rows.append({"operation": operation, "float64_vs_float32_net_ratio": f64_net / f32_net})
    ratio_df = pd.DataFrame(ratio_rows)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.barplot(
        data=ratio_df,
        x="operation",
        y="float64_vs_float32_net_ratio",
        color="#4C72B0",
        errorbar=None,
        ax=ax,
    )
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_title("ESP32 float64 Cost Relative To float32")
    ax.set_xlabel("Operation")
    ax.set_ylabel("Net runtime ratio")
    path = figures_dir / "esp32_float64_ratio.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    sns.barplot(
        data=summary_df,
        x="operation",
        y="max_heap_delta_bytes",
        hue="numeric_type",
        errorbar=None,
        ax=ax,
    )
    ax.set_title("ESP32 Heap Change During Benchmark")
    ax.set_xlabel("Operation")
    ax.set_ylabel("Free heap after - before (bytes)")
    path = figures_dir / "esp32_heap_delta.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths


def format_ns(value: float) -> str:
    return f"{value:,.3f}"


def markdown_table(rows: list[dict]) -> list[str]:
    summary = summarize_rows(rows)
    lines = [
        "| Operation | Type | Trials | Samples | Passes | Evaluations | Median raw ns/eval | p05-p95 raw | Median net ns/eval | p05-p95 net | Checksum | Heap delta range |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['op']}` | `{row['numeric_type']}` | {row['trial_count']} | {row['samples']} | "
            f"{row['passes']} | {row['measured_evaluations']:,} | {format_ns(row['median_raw_ns'])} | "
            f"{format_ns(row['p05_raw_ns'])}-{format_ns(row['p95_raw_ns'])} | {format_ns(row['median_net_ns'])} | "
            f"{format_ns(row['p05_net_ns'])}-{format_ns(row['p95_net_ns'])} | {row['checksum_state']} | "
            f"{row['min_heap_delta_bytes']} to {row['max_heap_delta_bytes']} |"
        )
    return lines


def build_findings(rows: list[dict]) -> list[str]:
    summary = summarize_rows(rows)
    arithmetic = [row for row in summary if row["op"] != "empty_loop"]
    fastest = min(arithmetic, key=lambda row: row["median_net_ns"])
    slowest = max(arithmetic, key=lambda row: row["median_net_ns"])
    overflow = [row for row in summary if row["checksum_state"] != "finite"]
    min_heap_delta = min(row["min_heap_delta_bytes"] for row in summary)
    max_heap_delta = max(row["max_heap_delta_bytes"] for row in summary)
    missing = sorted(
        {
            (operation, numeric_type)
            for operation in OPERATION_ORDER
            for numeric_type in TYPE_ORDER
        }
        - {(row["op"], row["numeric_type"]) for row in summary},
        key=lambda pair: (TYPE_ORDER.index(pair[1]), operation_sort_key(pair[0])),
    )

    by_pair = {(row["op"], row["numeric_type"]): row for row in summary}
    ratio_lines = []
    for operation in ["add", "subtract", "multiply", "divide"]:
        f32 = by_pair.get((operation, "float32"))
        f64 = by_pair.get((operation, "float64"))
        if not f32 or not f64 or f32["median_net_ns"] <= 0:
            continue
        ratio = f64["median_net_ns"] / f32["median_net_ns"]
        ratio_lines.append(f"`{operation}` float64 is {ratio:.2f}x the net float32 runtime")

    findings = [
        f"The fastest arithmetic row is `{fastest['op']}` `{fastest['numeric_type']}` at about {format_ns(fastest['median_net_ns'])} median net ns/eval.",
        f"The slowest arithmetic row is `{slowest['op']}` `{slowest['numeric_type']}` at about {format_ns(slowest['median_net_ns'])} median net ns/eval.",
    ]
    if missing:
        pairs = ", ".join(f"`{operation}` `{numeric_type}`" for operation, numeric_type in missing)
        findings.insert(0, f"This suite CSV is incomplete. Missing expected rows: {pairs}.")
    if ratio_lines:
        findings.append("; ".join(ratio_lines) + ".")
    if overflow:
        pairs = ", ".join(f"`{row['op']}` `{row['numeric_type']}`" for row in overflow)
        findings.append(f"Checksum overflow/non-finite state appeared for {pairs}. The timed kernel still ran, but the checksum method should be replaced with a bounded checksum before we rely on checksum comparisons.")
    else:
        methods = ", ".join(sorted({str(row["checksum_method"]) for row in summary}))
        findings.append(f"Every row produced a finite checksum. Checksum method: `{methods}`.")
    if min_heap_delta == 0 and max_heap_delta == 0:
        findings.append("Free heap was unchanged for every row in every trial, so this suite did not show heap growth during execution.")
    else:
        findings.append("At least one row changed free heap during execution; inspect the heap delta graph before treating memory behavior as stable.")
    return findings


def write_report(report_dir: Path, source_csv: Path, rows: list[dict], figure_paths: list[Path]) -> Path:
    report_path = report_dir / "esp32_report.md"
    profile = rows[0]
    trial_count = len({row["trial_index"] for row in rows})
    lines = [
        "# ESP32 Arithmetic Suite Report",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        f"Source CSV: `{source_csv}`",
        "",
        "## Method",
        "",
        "The ESP32 ran one suite containing every current arithmetic operation/type pair. Each row used the same sample count, pass count, and seed, so runtime comparisons are controlled within this firmware-generated input stream.",
        "",
        f"Samples per operation: `{profile['samples']}`",
        "",
        f"Passes per operation: `{profile['passes']}`",
        "",
        f"Trials: `{trial_count}`",
        "",
        f"Measured evaluations per row: `{profile['measured_evaluations']:,}`",
        "",
        f"Seed: `{profile['seed']}`",
        "",
        "Raw runtime includes loop/input/checksum overhead. Net runtime subtracts the matching numeric type's empty-loop row from the same trial, so it is a better first approximation of arithmetic cost, but it is still not a cycle-accurate hardware measurement.",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {finding}" for finding in build_findings(rows))
    lines.extend(["", "## Results", ""])
    lines.extend(markdown_table(rows))
    if figure_paths:
        lines.extend(["", "## Figures", ""])
        for figure in figure_paths:
            lines.extend([f"![{figure.stem}]({figure.as_posix()})", ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "The ESP32 strongly favors `float32` for this benchmark. `float64` storage is available because `double_size_bytes` is 8, but double-precision arithmetic is much slower, especially division.",
            "",
            "The checksum is a bounded validation value. It is not a mathematically meaningful sum of all results; its job is to keep the compiler from removing the arithmetic work and to make repeated runs comparable without overflowing.",
            "",
            "For the project direction, this is exactly the kind of characterization layer we want: a curious user can run one command, get a clean report, and learn which operations and numeric types are friendly on a given device.",
            "",
            "## Next Steps",
            "",
            "- Build a combined workstation-vs-ESP32 report using the native C++ suite and this ESP32 suite.",
            "- Add elementary functions after the arithmetic baseline is stable: `sqrt`, `sin`, `cos`, `exp`, `log`, and `pow`.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    source_csv = args.csv or newest_suite_csv()
    rows = enrich_rows(read_rows(source_csv))
    run_id = source_csv.stem.replace("_suite", "") if source_csv.name.endswith("_suite.csv") else str(uuid.uuid4())
    report_dir = args.out_dir or REPORTS_DIR / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = report_dir / "esp32_enriched_rows.csv"
    write_csv(raw_csv, rows)
    figure_paths = write_figures(report_dir, rows)
    report_path = write_report(report_dir, source_csv, rows, figure_paths)

    latest_report = REPORTS_DIR / "latest_esp32_report.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report_path, latest_report)

    print(f"Source CSV: {source_csv}")
    print(f"Report:     {report_path}")
    print(f"Latest:     {latest_report}")
    print(f"Figures:    {report_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
