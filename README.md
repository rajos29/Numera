# Numera Research Prototype

This repository is an early research prototype for a small scientific-computing layer aimed at edge devices and lightweight post-processing workflows.

The current goal is not to clone NumPy. The goal is to study how a backend can choose a computation method from the problem constraints:

- operation or function
- input domain
- input size
- target error
- memory budget
- backend hardware

The prototype currently compares direct builtin math, Taylor approximations, range-reduced Taylor approximations, Newton methods, lookup tables, log-domain transforms, frequency-domain transforms, and hybrid methods.

## Layout

- `python_lab/` - Python research scripts, report generation, ESP32 serial runner, and requirements.
- `firmware_esp32/arithmetic_serial/` - Arduino ESP32 benchmark firmware.
- `native/` - Native workstation arithmetic benchmark backend.
- `experiments/` - JSON configs for arithmetic experiment families.
- `docs/` - Architecture notes, benchmarking notes, and ESP32 backend notes.
- `tests/` - Lightweight Python tests.

Generated benchmark outputs are written under `python_lab/data/` and are ignored by Git by default.

## Python Setup

```powershell
pip install -r python_lab\requirements.txt
```

Run the approximation strategy lab:

```powershell
python python_lab\approximation_lab.py
```

Run the workstation arithmetic lab:

```powershell
python python_lab\benchmark_suite.py
```

Run the ESP32 serial suite after uploading the firmware:

```powershell
python python_lab\esp32_serial_runner.py
python python_lab\esp32_report.py
```

## ESP32 Firmware

Open this sketch in Arduino IDE:

```text
firmware_esp32/arithmetic_serial/arithmetic_serial.ino
```

The ESP32 suite emits JSON over serial so the Python runner can collect results into CSV/report files.

## Current Research Direction

The project is moving toward a backend method selector:

```text
Given a function, data shape, tolerance, memory budget, and hardware profile,
choose the cheapest method that satisfies the error target.
```

The interesting part is not only measuring speed. It is learning when a method becomes appropriate:

- Taylor series need suitable domains.
- Range reduction can make wide domains tractable.
- Lookup tables trade memory for runtime.
- Newton methods trade iterations for accuracy.
- Domain transforms can turn one hard operation into an easier equivalent operation.
- Hybrid methods combine strategies based on input size or domain.

## License And Attribution

This project is intended to be released under the Apache License 2.0.

If this project helps your work, please preserve the license and attribution notices required by the license. A visible credit is also appreciated in papers, demos, videos, posts, or products that use the project:

```text
Numera by RAJOS
```

Suggested citation:

```text
RAJOS. Numera: backend-aware scientific computing experiments for edge devices and post-processing workflows.
```

## Push Notes

Do not commit generated build artifacts or generated run data. They are intentionally ignored through `.gitignore`.

If a specific report or figure becomes important enough to preserve, copy a curated snapshot into `docs/` with a short explanation.
