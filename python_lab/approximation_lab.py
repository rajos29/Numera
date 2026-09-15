"""Approximation selection lab for elementary numerical methods.

This is a Python learning prototype. It compares approximation parameters
against runtime and error so we can later port promising choices to ESP32.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "approx"
RELATIVE_ERROR_EPSILON = 1.0e-30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--abs-error-threshold", type=float, default=1e-3)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_dict_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def bounded_checksum(checksum: float, value: float) -> float:
    checksum += value
    if checksum > 1_000_000.0:
        checksum -= 2_000_000.0
    elif checksum < -1_000_000.0:
        checksum += 2_000_000.0
    return checksum


def sin_taylor(x: float, order: int) -> float:
    term = x
    total = x
    for n in range(1, (order + 1) // 2):
        term *= -(x * x) / ((2 * n) * (2 * n + 1))
        total += term
    return total


def cos_taylor(x: float, order: int) -> float:
    term = 1.0
    total = 1.0
    for n in range(1, order // 2 + 1):
        term *= -(x * x) / ((2 * n - 1) * (2 * n))
        total += term
    return total


def exp_taylor(x: float, order: int) -> float:
    term = 1.0
    total = 1.0
    for n in range(1, order + 1):
        term *= x / n
        total += term
    return total


def log_atanh_series(x: float, terms: int) -> float:
    z = (x - 1.0) / (x + 1.0)
    z_power = z
    total = 0.0
    for n in range(terms):
        denominator = 2 * n + 1
        total += z_power / denominator
        z_power *= z * z
    return 2.0 * total


def tan_taylor_ratio(x: float, order: int) -> float:
    return sin_taylor(x, order) / cos_taylor(x, order)


def atan_taylor(x: float, order: int) -> float:
    x_power = x
    total = x
    sign = -1.0
    for denominator in range(3, order + 1, 2):
        x_power *= x * x
        total += sign * x_power / denominator
        sign *= -1.0
    return total


def asin_taylor(x: float, order: int) -> float:
    total = x
    term = x
    for n in range(1, (order + 1) // 2):
        term *= ((2 * n - 1) ** 2 * x * x) / ((2 * n) * (2 * n + 1))
        total += term
    return total


def sqrt_newton(value: float, iterations: int) -> float:
    if value == 0.0:
        return 0.0
    _mantissa, exponent = math.frexp(value)
    estimate = math.ldexp(1.0, math.ceil(exponent / 2.0))
    for _ in range(iterations):
        estimate = 0.5 * (estimate + value / estimate)
    return estimate


def reduce_angle_to_half_pi(x: float) -> tuple[float, float]:
    wrapped = (x + math.pi) % (2.0 * math.pi) - math.pi
    sign = 1.0
    if wrapped > math.pi / 2.0:
        wrapped = math.pi - wrapped
    elif wrapped < -math.pi / 2.0:
        wrapped = -math.pi - wrapped
    return wrapped, sign


def reduce_angle_to_half_pi_for_cos(x: float) -> tuple[float, float]:
    wrapped = (x + math.pi) % (2.0 * math.pi) - math.pi
    sign = 1.0
    if wrapped > math.pi / 2.0:
        wrapped = math.pi - wrapped
        sign = -1.0
    elif wrapped < -math.pi / 2.0:
        wrapped = -math.pi - wrapped
        sign = -1.0
    return wrapped, sign


def sin_range_reduced_taylor(x: float, order: int) -> float:
    reduced, sign = reduce_angle_to_half_pi(x)
    return sign * sin_taylor(reduced, order)


def cos_range_reduced_taylor(x: float, order: int) -> float:
    reduced, sign = reduce_angle_to_half_pi_for_cos(x)
    return sign * cos_taylor(reduced, order)


def exp_range_reduced_taylor(x: float, order: int) -> float:
    integer_part = math.floor(x)
    fractional_part = x - integer_part
    return math.exp(integer_part) * exp_taylor(fractional_part, order)


def make_lookup_table(reference, low: float, high: float, table_size: int):
    values = [reference(low + (high - low) * i / (table_size - 1)) for i in range(table_size)]

    def lookup(x: float) -> float:
        if x <= low:
            return values[0]
        if x >= high:
            return values[-1]
        position = (x - low) * (table_size - 1) / (high - low)
        index = int(position)
        fraction = position - index
        return values[index] * (1.0 - fraction) + values[index + 1] * fraction

    return lookup


def sqrt_log_domain(x: float) -> float:
    return math.exp(0.5 * math.log(x))


def sqrt_hybrid_newton_log(x: float, iterations: int) -> float:
    if x <= 0.0:
        return 0.0
    estimate = math.exp(0.5 * math.log(x))
    for _ in range(iterations):
        estimate = 0.5 * (estimate + x / estimate)
    return estimate


def derivative_sin_central(x: float, h: float) -> float:
    return (math.sin(x + h) - math.sin(x - h)) / (2.0 * h)


def domain_label(low: float, high: float) -> str:
    return f"[{format_float(low)}, {format_float(high)}]"


def add_experiment(
    experiments: list[dict],
    function_name: str,
    method: str,
    parameter_name: str,
    parameter_value: float,
    input_low: float,
    input_high: float,
    reference,
    approximation,
    question: str,
    strategy_family: str | None = None,
) -> None:
    experiments.append(
        {
            "function": function_name,
            "method": method,
            "strategy_family": strategy_family or method,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "input_low": input_low,
            "input_high": input_high,
            "input_domain": domain_label(input_low, input_high),
            "reference": reference,
            "approximation": approximation,
            "question": question,
        }
    )


def build_experiments() -> list[dict]:
    experiments: list[dict] = []
    domains = {
        "sin": [(-math.pi / 8.0, math.pi / 8.0), (-math.pi / 4.0, math.pi / 4.0), (-math.pi / 2.0, math.pi / 2.0), (-math.pi, math.pi), (-4.0 * math.pi, 4.0 * math.pi)],
        "cos": [(-math.pi / 8.0, math.pi / 8.0), (-math.pi / 4.0, math.pi / 4.0), (-math.pi / 2.0, math.pi / 2.0), (-math.pi, math.pi), (-4.0 * math.pi, 4.0 * math.pi)],
        "tan": [(-math.pi / 12.0, math.pi / 12.0), (-math.pi / 6.0, math.pi / 6.0), (-math.pi / 4.0, math.pi / 4.0), (-math.pi / 3.0, math.pi / 3.0)],
        "atan": [(-0.25, 0.25), (-0.5, 0.5), (-0.75, 0.75), (-1.0, 1.0)],
        "asin": [(-0.25, 0.25), (-0.5, 0.5), (-0.75, 0.75), (-0.9, 0.9)],
        "exp": [(-0.25, 0.25), (-0.5, 0.5), (-1.0, 1.0), (-2.0, 2.0)],
        "log": [(0.9, 1.1), (0.75, 1.25), (0.5, 2.0), (0.25, 4.0)],
        "sqrt": [(1e-6, 1.0), (1e-3, 100.0), (1e-6, 10_000.0)],
        "d_sin": [(-math.pi, math.pi)],
    }
    references = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "atan": math.atan,
        "asin": math.asin,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "d_sin": math.cos,
    }

    for function_name in ["sin", "cos", "tan", "atan", "asin", "exp", "log", "sqrt"]:
        for low, high in domains[function_name]:
            reference = references[function_name]
            question = f"Builtin {function_name} baseline on {domain_label(low, high)}."
            add_experiment(
                experiments,
                function_name,
                "builtin_math",
                "baseline",
                0.0,
                low,
                high,
                reference,
                reference,
                question,
                "direct_builtin",
            )

    for low, high in domains["sin"]:
        for order in [1, 3, 5, 7, 9, 11, 13]:
            add_experiment(
                experiments,
                "sin",
                "taylor_maclaurin",
                "order",
                order,
                low,
                high,
                math.sin,
                lambda x, order=order: sin_taylor(x, order),
                "How does sine Taylor order interact with input domain?",
                "taylor_approximation",
            )
        for order in [3, 5, 7, 9, 11]:
            add_experiment(
                experiments,
                "sin",
                "range_reduced_taylor",
                "order",
                order,
                low,
                high,
                math.sin,
                lambda x, order=order: sin_range_reduced_taylor(x, order),
                "How much does range reduction help sine Taylor across wider input intervals?",
                "range_reduced_taylor",
            )
        for table_size in [16, 32, 64, 128, 256]:
            add_experiment(
                experiments,
                "sin",
                "lookup_table_linear",
                "table_size",
                table_size,
                low,
                high,
                math.sin,
                make_lookup_table(math.sin, low, high, table_size),
                "How does sine lookup-table size trade memory for runtime and interpolation error?",
                "lookup_table",
            )
    for low, high in domains["cos"]:
        for order in [0, 2, 4, 6, 8, 10, 12, 14]:
            add_experiment(
                experiments,
                "cos",
                "taylor_maclaurin",
                "order",
                order,
                low,
                high,
                math.cos,
                lambda x, order=order: cos_taylor(x, order),
                "How does cosine Taylor order interact with input domain?",
                "taylor_approximation",
            )
        for order in [2, 4, 6, 8, 10, 12]:
            add_experiment(
                experiments,
                "cos",
                "range_reduced_taylor",
                "order",
                order,
                low,
                high,
                math.cos,
                lambda x, order=order: cos_range_reduced_taylor(x, order),
                "How much does range reduction help cosine Taylor across wider input intervals?",
                "range_reduced_taylor",
            )
        for table_size in [16, 32, 64, 128, 256]:
            add_experiment(
                experiments,
                "cos",
                "lookup_table_linear",
                "table_size",
                table_size,
                low,
                high,
                math.cos,
                make_lookup_table(math.cos, low, high, table_size),
                "How does cosine lookup-table size trade memory for runtime and interpolation error?",
                "lookup_table",
            )
    for low, high in domains["tan"]:
        for order in [3, 5, 7, 9, 11, 13]:
            add_experiment(
                experiments,
                "tan",
                "taylor_ratio",
                "order",
                order,
                low,
                high,
                math.tan,
                lambda x, order=order: tan_taylor_ratio(x, order),
                "How does tangent's Taylor-ratio approximation behave as the domain widens?",
                "taylor_approximation",
            )
        for table_size in [16, 32, 64, 128, 256]:
            add_experiment(
                experiments,
                "tan",
                "lookup_table_linear",
                "table_size",
                table_size,
                low,
                high,
                math.tan,
                make_lookup_table(math.tan, low, high, table_size),
                "How does tangent lookup-table size behave as the interval approaches steeper slopes?",
                "lookup_table",
            )
    for low, high in domains["atan"]:
        for order in [1, 3, 5, 7, 9, 11, 13, 15]:
            add_experiment(
                experiments,
                "atan",
                "taylor_maclaurin",
                "order",
                order,
                low,
                high,
                math.atan,
                lambda x, order=order: atan_taylor(x, order),
                "How does arctangent Taylor order interact with distance from |x| = 1?",
                "taylor_approximation",
            )
        for table_size in [16, 32, 64, 128, 256]:
            add_experiment(
                experiments,
                "atan",
                "lookup_table_linear",
                "table_size",
                table_size,
                low,
                high,
                math.atan,
                make_lookup_table(math.atan, low, high, table_size),
                "How does arctangent lookup-table size trade memory for interpolation error?",
                "lookup_table",
            )
    for low, high in domains["asin"]:
        for order in [1, 3, 5, 7, 9, 11, 13, 15]:
            add_experiment(
                experiments,
                "asin",
                "taylor_maclaurin",
                "order",
                order,
                low,
                high,
                math.asin,
                lambda x, order=order: asin_taylor(x, order),
                "How does arcsine Taylor order interact with distance from |x| = 1?",
                "taylor_approximation",
            )
        for table_size in [16, 32, 64, 128, 256]:
            add_experiment(
                experiments,
                "asin",
                "lookup_table_linear",
                "table_size",
                table_size,
                low,
                high,
                math.asin,
                make_lookup_table(math.asin, low, high, table_size),
                "How does arcsine lookup-table size handle steeper behavior near |x| = 1?",
                "lookup_table",
            )
    for low, high in domains["exp"]:
        for order in [1, 2, 3, 4, 5, 6, 8, 10, 12, 14]:
            add_experiment(
                experiments,
                "exp",
                "taylor_maclaurin",
                "order",
                order,
                low,
                high,
                math.exp,
                lambda x, order=order: exp_taylor(x, order),
                "How does exponential Taylor order interact with input domain?",
                "taylor_approximation",
            )
        for order in [2, 3, 4, 5, 6, 8, 10]:
            add_experiment(
                experiments,
                "exp",
                "range_reduced_taylor",
                "order",
                order,
                low,
                high,
                math.exp,
                lambda x, order=order: exp_range_reduced_taylor(x, order),
                "How much does splitting exp(x) into integer and fractional parts reduce Taylor order?",
                "range_reduced_taylor",
            )
        for table_size in [16, 32, 64, 128, 256]:
            add_experiment(
                experiments,
                "exp",
                "lookup_table_linear",
                "table_size",
                table_size,
                low,
                high,
                math.exp,
                make_lookup_table(math.exp, low, high, table_size),
                "How does exponential lookup-table size trade memory for interpolation error?",
                "lookup_table",
            )
    for low, high in domains["log"]:
        for terms in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
            add_experiment(
                experiments,
                "log",
                "atanh_series",
                "terms",
                terms,
                low,
                high,
                math.log,
                lambda x, terms=terms: log_atanh_series(x, terms),
                "How does log atanh-series term count interact with distance from x = 1?",
                "taylor_approximation",
            )
        for table_size in [16, 32, 64, 128, 256]:
            add_experiment(
                experiments,
                "log",
                "lookup_table_linear",
                "table_size",
                table_size,
                low,
                high,
                math.log,
                make_lookup_table(math.log, low, high, table_size),
                "How does logarithm lookup-table size compare with a convergence-sensitive series?",
                "lookup_table",
            )
    for low, high in domains["sqrt"]:
        for iterations in range(1, 9):
            add_experiment(
                experiments,
                "sqrt",
                "newton_raphson",
                "iterations",
                iterations,
                low,
                high,
                math.sqrt,
                lambda x, iterations=iterations: sqrt_newton(x, iterations),
                "How does Newton iteration count interact with square-root input scale?",
                "newton_method",
            )
        add_experiment(
            experiments,
            "sqrt",
            "log_domain_transform",
            "builtin_transform",
            1,
            low,
            high,
            math.sqrt,
            sqrt_log_domain,
            "Can sqrt(x) be computed by transforming to log space, scaling, and transforming back?",
            "log_domain_transform",
        )
        for iterations in [0, 1, 2]:
            add_experiment(
                experiments,
                "sqrt",
                "hybrid_log_newton",
                "iterations_after_log_guess",
                iterations,
                low,
                high,
                math.sqrt,
                lambda x, iterations=iterations: sqrt_hybrid_newton_log(x, iterations),
                "Does a log-domain initial guess plus a few Newton refinements improve robustness?",
                "hybrid_method",
            )
        for table_size in [16, 32, 64, 128, 256]:
            add_experiment(
                experiments,
                "sqrt",
                "lookup_table_linear",
                "table_size",
                table_size,
                low,
                high,
                math.sqrt,
                make_lookup_table(math.sqrt, low, high, table_size),
                "How does square-root lookup-table size behave across different input scales?",
                "lookup_table",
            )
    for low, high in domains["d_sin"]:
        for exponent in range(1, 11):
            h = 10.0 ** (-exponent)
            add_experiment(
                experiments,
                "d_sin",
                "central_difference",
                "h",
                h,
                low,
                high,
                math.cos,
                lambda x, h=h: derivative_sin_central(x, h),
                "Which finite-difference step size balances truncation error and roundoff error?",
                "step_size_method",
            )
    return experiments


def deterministic_inputs(seed: int, function_name: str, samples: int, low: float, high: float) -> list[float]:
    rng = random.Random(f"{seed}:approximation:{function_name}:{low}:{high}")
    return [rng.uniform(low, high) for _ in range(samples)]


def calculate_sample_errors(experiment: dict, inputs: list[float]) -> list[dict]:
    rows: list[dict] = []
    for sample_id, x in enumerate(inputs):
        approx = float(experiment["approximation"](x))
        reference = float(experiment["reference"](x))
        absolute_error = abs(approx - reference)
        relative_error = absolute_error / max(abs(reference), RELATIVE_ERROR_EPSILON)
        rows.append(
            {
                "function": experiment["function"],
                "method": experiment["method"],
                "strategy_family": experiment["strategy_family"],
                "parameter_name": experiment["parameter_name"],
                "parameter_value": experiment["parameter_value"],
                "input_domain": experiment["input_domain"],
                "input_low": experiment["input_low"],
                "input_high": experiment["input_high"],
                "sample_id": sample_id,
                "x": x,
                "approximation": approx,
                "reference": reference,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
            }
        )
    return rows


def time_experiment(experiment: dict, inputs: list[float], passes: int) -> tuple[int, float]:
    checksum = 0.0
    start = time.perf_counter_ns()
    for _ in range(passes):
        for x in inputs:
            checksum = bounded_checksum(checksum, float(experiment["approximation"](x)))
    total_runtime_ns = time.perf_counter_ns() - start
    return total_runtime_ns, checksum


def deterministic_vector(seed: int, label: str, length: int) -> list[float]:
    rng = random.Random(f"{seed}:vector:{label}:{length}")
    return [rng.uniform(-1.0, 1.0) for _ in range(length)]


def direct_convolution(signal: list[float], kernel: list[float]) -> list[float]:
    output_length = len(signal) + len(kernel) - 1
    output = [0.0] * output_length
    for signal_index, signal_value in enumerate(signal):
        for kernel_index, kernel_value in enumerate(kernel):
            output[signal_index + kernel_index] += signal_value * kernel_value
    return output


def fft_convolution(signal: list[float], kernel: list[float]) -> list[float]:
    import numpy as np

    output_length = len(signal) + len(kernel) - 1
    fft_length = 1 << (output_length - 1).bit_length()
    signal_spectrum = np.fft.rfft(np.asarray(signal), fft_length)
    kernel_spectrum = np.fft.rfft(np.asarray(kernel), fft_length)
    output = np.fft.irfft(signal_spectrum * kernel_spectrum, fft_length)[:output_length]
    return output.tolist()


def max_pairwise_error(values: list[float], reference: list[float]) -> float:
    return max(abs(value - reference_value) for value, reference_value in zip(values, reference))


def time_vector_method(method, signal: list[float], kernel: list[float], passes: int) -> tuple[int, float, list[float]]:
    checksum = 0.0
    last_output: list[float] = []
    start = time.perf_counter_ns()
    for _ in range(passes):
        last_output = method(signal, kernel)
        for value in last_output:
            checksum = bounded_checksum(checksum, float(value))
    total_runtime_ns = time.perf_counter_ns() - start
    return total_runtime_ns, checksum, last_output


def run_frequency_domain_experiments(args: argparse.Namespace, run_id: str) -> tuple[list[dict], list[dict]]:
    observations: list[dict] = []
    sample_rows: list[dict] = []
    lengths = [32, 64, 128, 256, 512, 1024]
    kernel_length = 17
    methods = [
        ("direct_convolution", "direct_builtin", "signal_length", 0, direct_convolution),
        ("fft_convolution", "frequency_domain_transform", "signal_length", 0, fft_convolution),
        (
            "hybrid_direct_fft",
            "hybrid_method",
            "fft_threshold",
            256,
            lambda signal, kernel: direct_convolution(signal, kernel) if len(signal) <= 256 else fft_convolution(signal, kernel),
        ),
    ]

    for length in lengths:
        signal = deterministic_vector(args.seed, "convolution_signal", length)
        kernel = deterministic_vector(args.seed, "convolution_kernel", kernel_length)
        reference = direct_convolution(signal, kernel)
        input_domain = f"signal_length={length}, kernel_length={kernel_length}"
        for method_name, strategy_family, parameter_name, parameter_value, method in methods:
            output = method(signal, kernel)
            absolute_error = max_pairwise_error(output, reference)
            relative_error = absolute_error / max(max(abs(value) for value in reference), RELATIVE_ERROR_EPSILON)
            sample_rows.append(
                {
                    "function": "convolution",
                    "method": method_name,
                    "strategy_family": strategy_family,
                    "parameter_name": parameter_name,
                    "parameter_value": parameter_value if parameter_name == "fft_threshold" else length,
                    "input_domain": input_domain,
                    "input_low": 0.0,
                    "input_high": float(length),
                    "sample_id": 0,
                    "x": float(length),
                    "approximation": output[0] if output else 0.0,
                    "reference": reference[0] if reference else 0.0,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                }
            )
            for trial_id in range(args.trials):
                total_runtime_ns, checksum, _last_output = time_vector_method(method, signal, kernel, args.passes)
                observations.append(
                    {
                        "run_id": run_id,
                        "timestamp": now_iso(),
                        "backend": "python_learning_prototype",
                        "trial_id": trial_id,
                        "function": "convolution",
                        "method": method_name,
                        "strategy_family": strategy_family,
                        "parameter_name": parameter_name,
                        "parameter_value": parameter_value if parameter_name == "fft_threshold" else length,
                        "samples": 1,
                        "passes": args.passes,
                        "measured_evaluations": args.passes,
                        "input_domain": input_domain,
                        "input_low": 0.0,
                        "input_high": float(length),
                        "seed": args.seed,
                        "total_runtime_ns": total_runtime_ns,
                        "runtime_per_eval_ns": total_runtime_ns / args.passes,
                        "result_checksum": checksum,
                        "checksum_method": "bounded_sum_v1",
                        "question": "When is it worth transforming a data collection to frequency space to perform convolution?",
                    }
                )
    return observations, sample_rows


def run_experiments(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    observations: list[dict] = []
    sample_rows: list[dict] = []
    experiments = build_experiments()
    run_id = str(uuid.uuid4())

    for experiment in experiments:
        inputs = deterministic_inputs(
            args.seed,
            experiment["function"],
            args.samples,
            experiment["input_low"],
            experiment["input_high"],
        )
        sample_rows.extend(calculate_sample_errors(experiment, inputs))
        for trial_id in range(args.trials):
            total_runtime_ns, checksum = time_experiment(experiment, inputs, args.passes)
            observations.append(
                {
                    "run_id": run_id,
                    "timestamp": now_iso(),
                    "backend": "python_learning_prototype",
                    "trial_id": trial_id,
                    "function": experiment["function"],
                    "method": experiment["method"],
                    "strategy_family": experiment["strategy_family"],
                    "parameter_name": experiment["parameter_name"],
                    "parameter_value": experiment["parameter_value"],
                    "samples": args.samples,
                    "passes": args.passes,
                    "measured_evaluations": args.samples * args.passes,
                    "input_domain": experiment["input_domain"],
                    "input_low": experiment["input_low"],
                    "input_high": experiment["input_high"],
                    "seed": args.seed,
                    "total_runtime_ns": total_runtime_ns,
                    "runtime_per_eval_ns": total_runtime_ns / (args.samples * args.passes),
                    "result_checksum": checksum,
                    "checksum_method": "bounded_sum_v1",
                    "question": experiment["question"],
                }
            )

    vector_observations, vector_sample_rows = run_frequency_domain_experiments(args, run_id)
    observations.extend(vector_observations)
    sample_rows.extend(vector_sample_rows)

    summary = summarize(observations, sample_rows, args.abs_error_threshold)
    return observations, sample_rows, summary


def summarize(observations: list[dict], sample_rows: list[dict], abs_error_threshold: float) -> list[dict]:
    timing_groups: dict[tuple[str, str, str, float, str], list[dict]] = {}
    error_groups: dict[tuple[str, str, str, float, str], list[dict]] = {}
    for row in observations:
        key = (row["function"], row["method"], row["strategy_family"], float(row["parameter_value"]), row["input_domain"])
        timing_groups.setdefault(key, []).append(row)
    for row in sample_rows:
        key = (row["function"], row["method"], row["strategy_family"], float(row["parameter_value"]), row["input_domain"])
        error_groups.setdefault(key, []).append(row)

    summary: list[dict] = []
    for key, timing_group in sorted(timing_groups.items()):
        function_name, method, strategy_family, parameter_value, input_domain = key
        errors = error_groups[key]
        runtimes = [float(row["runtime_per_eval_ns"]) for row in timing_group]
        absolute_errors = [float(row["absolute_error"]) for row in errors]
        relative_errors = [float(row["relative_error"]) for row in errors]
        summary.append(
            {
                "function": function_name,
                "method": method,
                "strategy_family": strategy_family,
                "parameter_name": timing_group[0]["parameter_name"],
                "parameter_value": parameter_value,
                "input_domain": input_domain,
                "input_low": float(timing_group[0]["input_low"]),
                "input_high": float(timing_group[0]["input_high"]),
                "samples": int(timing_group[0]["samples"]),
                "passes": int(timing_group[0]["passes"]),
                "trials": len(timing_group),
                "median_runtime_ns": statistics.median(runtimes),
                "p05_runtime_ns": percentile(runtimes, 0.05),
                "p95_runtime_ns": percentile(runtimes, 0.95),
                "median_absolute_error": statistics.median(absolute_errors),
                "p95_absolute_error": percentile(absolute_errors, 0.95),
                "max_absolute_error": max(absolute_errors),
                "median_relative_error": statistics.median(relative_errors),
                "p95_relative_error": percentile(relative_errors, 0.95),
                "max_relative_error": max(relative_errors),
                "meets_abs_error_threshold": max(absolute_errors) <= abs_error_threshold,
                "question": timing_group[0]["question"],
            }
        )
    return summary


def recommendation_rows(summary: list[dict]) -> list[dict]:
    recommendations: list[dict] = []
    function_domains = sorted({(row["function"], row["input_domain"]) for row in summary})
    for function_name, input_domain in function_domains:
        candidates = [
            row
            for row in summary
            if row["function"] == function_name
            and row["input_domain"] == input_domain
            and row["strategy_family"] != "direct_builtin"
            and row["meets_abs_error_threshold"]
        ]
        if candidates:
            best = min(candidates, key=lambda row: row["median_runtime_ns"])
            status = "meets_threshold"
        else:
            best = min(
                [
                    row
                    for row in summary
                    if row["function"] == function_name
                    and row["input_domain"] == input_domain
                    and row["strategy_family"] != "direct_builtin"
                ],
                key=lambda row: row["max_absolute_error"],
            )
            status = "best_available_but_above_threshold"
        item = dict(best)
        item["recommendation_status"] = status
        recommendations.append(item)
    return recommendations


def write_figures(run_dir: Path, summary: list[dict], sample_rows: list[dict]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ModuleNotFoundError as exc:
        print(f"Skipping figures: missing {exc.name}.")
        return []

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    summary_df = pd.DataFrame(summary)
    sample_df = pd.DataFrame(sample_rows)
    approximation_df = summary_df[summary_df["strategy_family"] != "direct_builtin"].copy()
    paths: list[Path] = []

    tradeoff_specs = []
    for (function_name, method, parameter_name), _group in approximation_df.groupby(["function", "method", "parameter_name"], sort=True):
        if parameter_name in ["baseline", "builtin_transform"]:
            continue
        safe_method = str(method).replace("/", "_").replace(" ", "_")
        tradeoff_specs.append((function_name, method, parameter_name, f"{function_name}_{safe_method}_tradeoff.png"))

    for function_name, method, parameter_name, filename in tradeoff_specs:
        subset = approximation_df[
            (approximation_df["function"] == function_name)
            & (approximation_df["method"] == method)
            & (approximation_df["parameter_name"] == parameter_name)
        ].copy()
        fig, ax_error = plt.subplots(figsize=(10, 6), constrained_layout=True)
        sns.lineplot(
            data=subset,
            x="parameter_value",
            y="max_absolute_error",
            hue="input_domain",
            marker="o",
            ax=ax_error,
            legend=True,
        )
        ax_error.set_yscale("log")
        if parameter_name == "h":
            ax_error.set_xscale("log")
        ax_error.set_xlabel(parameter_name)
        ax_error.set_ylabel("Max absolute error", color="#C44E52")
        ax_error.tick_params(axis="y", labelcolor="#C44E52")

        ax_runtime = ax_error.twinx()
        sns.lineplot(
            data=subset,
            x="parameter_value",
            y="median_runtime_ns",
            hue="input_domain",
            marker="s",
            ax=ax_runtime,
            legend=False,
        )
        if parameter_name == "h":
            ax_runtime.set_xscale("log")
        ax_runtime.set_ylabel("Median runtime (ns / eval)", color="#4C72B0")
        ax_runtime.tick_params(axis="y", labelcolor="#4C72B0")
        ax_error.set_title(f"{function_name}: {method} Tradeoff")
        path = figures_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    function_names = sorted(approximation_df["function"].unique())
    cols = 3
    rows_needed = math.ceil(len(function_names) / cols)
    fig, axes = plt.subplots(rows_needed, cols, figsize=(16, 4.8 * rows_needed), constrained_layout=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, function_name in zip(axes_flat, function_names):
        subset = approximation_df[approximation_df["function"] == function_name].copy()
        sns.scatterplot(
            data=subset,
            x="median_runtime_ns",
            y="max_absolute_error",
            hue="strategy_family",
            style="input_domain",
            s=90,
            legend=False,
            ax=ax,
        )
        for row in subset.itertuples(index=False):
            label = f"{row.method}:{format_float(float(row.parameter_value))}"
            ax.annotate(label, (row.median_runtime_ns, row.max_absolute_error), fontsize=8, xytext=(5, 4), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_title(function_name)
        ax.set_xlabel("Median runtime (ns / eval)")
        ax.set_ylabel("Max absolute error")
    for ax in axes_flat[len(function_names):]:
        ax.set_visible(False)
    fig.suptitle("Runtime vs Worst-Case Error, Split By Approximation Family")
    path = figures_dir / "runtime_vs_error_by_function.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    builtin_df = summary_df[summary_df["strategy_family"] == "direct_builtin"].copy()
    sns.barplot(
        data=builtin_df,
        x="function",
        y="median_runtime_ns",
        color="#55A868",
        errorbar=None,
        ax=ax,
    )
    ax.set_title("Python Builtin Baseline Runtime")
    ax.set_xlabel("Function")
    ax.set_ylabel("Median runtime (ns / eval)")
    path = figures_dir / "builtin_runtime_baseline.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    selected = approximation_df.sort_values(["function", "input_domain", "max_absolute_error"]).groupby(["function", "input_domain"]).head(1)
    selected_keys = {
        (row.function, row.method, row.strategy_family, float(row.parameter_value), row.input_domain)
        for row in selected.itertuples(index=False)
    }
    selected_samples = sample_df[
        sample_df.apply(
            lambda row: (row["function"], row["method"], row["strategy_family"], float(row["parameter_value"]), row["input_domain"]) in selected_keys,
            axis=1,
        )
    ].copy()
    selected_samples["absolute_error_plot"] = selected_samples["absolute_error"].clip(lower=1e-16)
    selected_samples["candidate"] = (
        selected_samples["function"]
        + " "
        + selected_samples["method"]
        + " "
        + selected_samples["parameter_value"].astype(str)
        + " "
        + selected_samples["input_domain"]
    )
    cols = 3
    rows_needed = math.ceil(len(function_names) / cols)
    fig, axes = plt.subplots(rows_needed, cols, figsize=(16, 4.8 * rows_needed), constrained_layout=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, function_name in zip(axes_flat, function_names):
        subset = selected_samples[selected_samples["function"] == function_name]
        sns.scatterplot(
            data=subset,
            x="x",
            y="absolute_error_plot",
            hue="candidate",
            alpha=0.45,
            s=14,
            ax=ax,
            legend=False,
        )
        ax.set_yscale("log")
        if function_name == "sqrt":
            ax.set_xscale("log")
        ax.set_title(function_name)
        ax.set_xlabel("Input x")
        ax.set_ylabel("Absolute error")
    for ax in axes_flat[len(function_names):]:
        ax.set_visible(False)
    fig.suptitle("Input Value vs Absolute Error, Best Candidates")
    path = figures_dir / "input_vs_error_by_function.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    squeeze_rows = []
    for i in range(250):
        x = (math.pi / 2.0) * (i / 249)
        squeeze_rows.extend(
            [
                {"x": x, "curve": "sin(x)", "y": math.sin(x)},
                {"x": x, "curve": "upper: x", "y": x},
                {"x": x, "curve": "lower: x - x^3/6", "y": x - (x**3) / 6.0},
            ]
        )
    squeeze_df = pd.DataFrame(squeeze_rows)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    sns.lineplot(data=squeeze_df, x="x", y="y", hue="curve", ax=ax)
    ax.set_title("Sine Squeeze Bounds On [0, pi/2]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    path = figures_dir / "sin_squeeze_bounds.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths


def format_float(value: float) -> str:
    if abs(value) >= 1000.0 or (0.0 < abs(value) < 0.001):
        return f"{value:.3e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def write_report(
    report_path: Path,
    observations: list[dict],
    sample_rows: list[dict],
    summary: list[dict],
    figure_paths: list[Path],
    abs_error_threshold: float,
) -> None:
    profile = observations[0]
    recommendations = recommendation_rows(summary)
    lines = [
        "# Approximation Selection Lab Report",
        "",
        f"Generated: `{now_iso()}`",
        "",
        "## Purpose",
        "",
        "This run compares approximation strategies by asking how accuracy changes as we spend more computation. It is a Python learning prototype, so the timing numbers are useful for comparing experiment shapes, not for final ESP32 claims.",
        "",
        "## Method",
        "",
        f"Samples per candidate: `{profile['samples']}`",
        "",
        f"Passes per timing trial: `{profile['passes']}`",
        "",
        f"Timing trials per candidate: `{profile['trials'] if 'trials' in profile else len({row['trial_id'] for row in observations})}`",
        "",
        f"Accuracy threshold for recommendations: max absolute error <= `{abs_error_threshold:g}`",
        "",
        "Candidates tested: direct builtin, Taylor approximation, range-reduced Taylor, Newton method, lookup table, log-domain transform, frequency-domain transform, and hybrid method.",
        "",
        "## Strategy Catalog",
        "",
        "| Strategy family | Current role in this run | Backend-selection meaning |",
        "|---|---|---|",
        "| `direct_builtin` | Python `math` or direct reference implementation | Baseline: use vendor/runtime implementation when speed and accuracy are acceptable. |",
        "| `taylor_approximation` | Maclaurin or related series | Good when the input interval is small enough that low order reaches the error target. |",
        "| `range_reduced_taylor` | Reduce inputs before Taylor evaluation | Good when periodic or decomposable functions can be moved into a friendlier interval. |",
        "| `newton_method` | Iterative square root | Good when each extra iteration buys a predictable accuracy gain. |",
        "| `lookup_table` | Linear interpolation over precomputed values | Trades memory for speed; useful when the domain is bounded and known ahead of time. |",
        "| `log_domain_transform` | `sqrt(x) = exp(0.5 log(x))` | Tests whether changing coordinate space simplifies the target operation. |",
        "| `frequency_domain_transform` | FFT convolution | Tests whether transforming a collection makes an expensive operation cheaper. |",
        "| `hybrid_method` | Log guess plus Newton; direct/FFT threshold | Lets a backend choose a path from input size, interval, and accuracy target. |",
        "",
        "## Recommendations",
        "",
        "| Function | Domain | Status | Strategy | Method | Parameter | Median runtime ns/eval | Max abs error | p95 abs error |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in recommendations:
        lines.append(
            f"| `{row['function']}` | `{row['input_domain']}` | `{row['recommendation_status']}` | `{row['strategy_family']}` | `{row['method']}` | "
            f"{format_float(float(row['parameter_value']))} | {format_float(float(row['median_runtime_ns']))} | "
            f"{format_float(float(row['max_absolute_error']))} | {format_float(float(row['p95_absolute_error']))} |"
        )

    lines.extend(
        [
            "",
            "## Result Summary",
            "",
            "| Function | Domain | Strategy | Method | Parameter | Trials | Median runtime ns/eval | p05-p95 runtime | Median abs error | p95 abs error | Max abs error |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| `{row['function']}` | `{row['input_domain']}` | `{row['strategy_family']}` | `{row['method']}` | {format_float(float(row['parameter_value']))} | "
            f"{row['trials']} | {format_float(float(row['median_runtime_ns']))} | "
            f"{format_float(float(row['p05_runtime_ns']))}-{format_float(float(row['p95_runtime_ns']))} | "
            f"{format_float(float(row['median_absolute_error']))} | {format_float(float(row['p95_absolute_error']))} | "
            f"{format_float(float(row['max_absolute_error']))} |"
        )

    if figure_paths:
        lines.extend(["", "## Figures", ""])
        for figure in figure_paths:
            lines.extend([f"![{figure.stem}]({figure.as_posix()})", ""])

    lines.extend(
        [
            "## What To Look For",
            "",
            "- Taylor-style series usually show the classic tradeoff: higher order buys lower error at higher computational cost.",
            "- Range reduction is the first major backend-selection lever: it can make a low-order series behave like a higher-order one by changing the input interval.",
            "- Newton square root often improves quickly at first, then extra iterations become less valuable.",
            "- Lookup tables are memory-for-computation trades; the important question is whether the table resolution is enough for the domain and error target.",
            "- Log-domain transforms and FFT-style transforms are coordinate changes: they can turn one hard operation into a different operation that may be cheaper for a given backend and data shape.",
            "- Logarithm and inverse-trig series are strongly shaped by input domain; moving near convergence boundaries changes the value of extra terms.",
            "- Finite differences expose the step-size trap: very large `h` has truncation error, while very small `h` can suffer roundoff cancellation.",
            "- The sine squeeze plot is included as a math intuition aid: simple bounds can visually trap the true function even before we use high-order approximations.",
            "",
            "## Next Steps",
            "",
            "- Port this broader elementary-function suite to ESP32.",
            "- Compare ESP32 builtin math functions against these approximation families.",
            "- Add domain-specific kernels, such as vector norm and normalized signal transforms.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.samples < 1 or args.trials < 1 or args.passes < 1:
        print("--samples, --trials, and --passes must be at least 1")
        return 1

    observations, sample_rows, summary = run_experiments(args)
    run_id = observations[0]["run_id"]
    run_dir = DATA_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    observations_path = run_dir / "observations.csv"
    samples_path = run_dir / "sample_errors.csv"
    summary_path = run_dir / "summary.csv"
    recommendations_path = run_dir / "recommendations.csv"
    write_dict_csv(observations_path, observations)
    write_dict_csv(samples_path, sample_rows)
    write_dict_csv(summary_path, summary)
    write_dict_csv(recommendations_path, recommendation_rows(summary))

    figure_paths = write_figures(run_dir, summary, sample_rows)
    report_path = run_dir / "approximation_report.md"
    write_report(report_path, observations, sample_rows, summary, figure_paths, args.abs_error_threshold)

    latest_report = DATA_DIR / "latest_approximation_report.md"
    latest_report.parent.mkdir(parents=True, exist_ok=True)
    latest_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Report:          {report_path}")
    print(f"Observations:    {observations_path}")
    print(f"Sample errors:   {samples_path}")
    print(f"Summary:         {summary_path}")
    print(f"Recommendations: {recommendations_path}")
    print(f"Figures:         {run_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
