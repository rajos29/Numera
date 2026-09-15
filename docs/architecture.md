# Architecture

The project currently has two layers.

## Python Learning And Orchestration Layer

The Python layer owns:

- experiment configuration
- deterministic input generation
- shared sample sets
- process orchestration
- raw CSV and SQLite-style observations
- summary statistics
- Matplotlib and Seaborn figures
- Markdown reports
- reference/error calculations

The Python backend remains useful for learning the math and report shape, but
its timing values include interpreter overhead.

## Native Arithmetic Backend

The native backend lives in:

```text
native/
```

It contains a small C++ executable that:

- reads Python-generated input CSV files
- runs one arithmetic operation over the shared sample set
- supports `float32` and `float64`
- measures elapsed time with `std::chrono::steady_clock`
- writes one structured JSON observation to stdout
- consumes the result through a process-wide volatile sink to reduce the chance
  of dead-code elimination

Python invokes this executable through:

```text
python_lab/native_backend.py
```

This keeps the important split:

```text
Python experiment suite
        |
        v
native C++ arithmetic executable
        |
        v
CPU
```

## Current Tooling Status

The native backend builds on this machine through Visual Studio Build Tools.
The Python wrapper locates `VsDevCmd.bat` and builds the executable with MSVC.

The runner supports:

```powershell
python python_lab/arithmetic_lab.py --backend native
python python_lab/benchmark_suite.py --backend native
```

Those commands will build and run the native executable once MSVC Build Tools,
MinGW `g++`, or LLVM `clang++` is available on `PATH`.

MSVC assembly output is written to:

```text
native/build/arithmetic_benchmark.asm
```

That listing is used as an early credibility check that floating-point
arithmetic instructions remain in the compiled benchmark.
