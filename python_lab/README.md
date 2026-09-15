# Python Arithmetic Characterization Lab

This is a learning prototype for the workstation characterization project.

It is intentionally written in Python so the math, experiment structure, and
data model are easy to inspect before moving the timing-sensitive kernel code
to C or C++.

Important limitation:

Python interpreter overhead dominates these measurements. Treat the results as
evidence about the experiment design, error calculations, and data pipeline,
not as trustworthy low-level CPU arithmetic timings.

## Run

```powershell
python python_lab/arithmetic_lab.py
```

To run the current multi-config learning suite:

```powershell
python python_lab/benchmark_suite.py
```

To use the native C++ backend after installing a compiler:

```powershell
python python_lab/arithmetic_lab.py --backend native
python python_lab/benchmark_suite.py --backend native
```

Useful options:

```powershell
python python_lab/arithmetic_lab.py --config experiments/arithmetic_python.json
python python_lab/arithmetic_lab.py --samples 1000 --trials 7 --seed 1234
python python_lab/arithmetic_lab.py --operations add subtract multiply divide --numeric-types float32 float64
python python_lab/benchmark_suite.py --samples 1000 --trials 5
python python_lab/arithmetic_lab.py --backend native --samples 1000 --trials 5
python python_lab/benchmark_suite.py --backend native
python python_lab/esp32_serial_runner.py --port COM3 --trials 5
python python_lab/esp32_report.py
python python_lab/approximation_lab.py
```

The default run uses:

```text
experiments/arithmetic_python.json
```

Outputs are written to:

```text
python_lab/data/
```

The run creates:

- SQLite raw observations
- CSV raw observations
- JSON system metadata
- JSON summary statistics
- Matplotlib/Seaborn PNG figures when plotting dependencies are installed
- HTML index page for the generated figures

ESP32 suite CSVs are written under `python_lab/data/esp32/`. Use `--trials 5`
to collect a repeated baseline with median and p05-p95 report ranges. Run
`python_lab/esp32_report.py` to create a Markdown report and
Matplotlib/Seaborn figures from the newest ESP32 suite.

Approximation experiments are written under `python_lab/data/approx/`. Run
`python_lab/approximation_lab.py` to compare Taylor, Newton, and finite
difference parameters by runtime and error.

To enable the preferred plotting output:

```powershell
python -m pip install -r python_lab/requirements.txt
```

## What This Teaches

- Why benchmarks repeat an operation many times.
- Why operations should use shared input sets for fair comparison.
- Why multiple trials matter.
- Why raw observations should be preserved.
- How float32 and float64 can behave differently.
- How absolute and relative error are calculated.
- Why relative error becomes awkward near zero.
- Why Python is useful for orchestration but not primitive-operation timing.
