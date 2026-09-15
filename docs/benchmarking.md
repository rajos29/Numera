# Benchmarking Methodology

This project currently has a Python learning-prototype backend and an initial
native C++ backend.

The prototype exists to make the experiment shape visible before moving
timing-sensitive arithmetic kernels into compiled native code.

## Current Measurement Meaning

The Python lab measures the cost of executing arithmetic through Python loops,
function calls, object handling, and simulated `float32` conversion.

Therefore the timing values are not native CPU arithmetic timings.

Current timing values are useful for validating:

- deterministic input generation
- shared input sets across operations
- repeated samples
- independent trials
- raw observation storage
- summary statistics
- absolute and relative error calculations
- visual diagnostics

## Samples And Trials

A sample is one deterministic input pair `(a, b)`.

A trial is one full pass over a shared sample set for one operation and one
numeric type.

For fair comparison, arithmetic operations use the same sample set within a
trial and numeric type. For example, `add`, `subtract`, `multiply`, and
`divide` all evaluate sample 17 from the same generated `(a, b)` pair.

## Error Measures

Absolute error is:

```text
abs(observed - reference)
```

Relative error is:

```text
abs(observed - reference) / abs(reference)
```

Relative error is omitted when the reference is zero or too close to zero.

Absolute error must not be compared across operations without considering
result magnitude. Multiplication often produces larger output magnitudes than
division for the current input range, so its absolute error may appear larger
even when its relative error is comparable.

## Known Limitations

- Python interpreter overhead dominates primitive-operation timing.
- `float32` is simulated through packing and unpacking, which adds overhead.
- The current profiler records basic machine context but not compiler details.
- Per-sample timing is intentionally not attempted in Python because timer
  overhead would dominate.
- Current plots show trial-level timing and per-sample error, not per-sample
  runtime.
- The HTML report is only an index for generated figures; PNG figures are the
  preferred report output.

## Native Backend Requirement

Timing conclusions about primitive arithmetic cost require the native backend.

The native backend currently measures repeated execution of one operation over
Python-generated shared samples. It uses `std::chrono::steady_clock` for wall
time and records a checksum through a volatile sink so the compiler cannot
simply discard the loop.

The volatile sink is a first-pass compiler-elimination guard, not the final word
on benchmark validity. Once the backend builds, suspicious results should be
checked by comparing optimization levels and inspecting generated assembly.

Current native verification blocker:

```text
Resolved: MSVC Build Tools are available through `VsDevCmd.bat`.
```

Current native assembly check:

```text
native/build/arithmetic_benchmark.asm
```

The generated listing contains scalar floating-point instructions including
`addss`, `subss`, `mulss`, `divss`, `addsd`, `subsd`, `mulsd`, and `divsd`.
This confirms the measured native loop still contains arithmetic operations.

The native backend now separates:

- `samples`: distinct input pairs
- `passes`: repeated passes over those pairs inside the timed native loop
- `measured_evaluations`: `samples * passes`

Reported runtime is normalized as nanoseconds per measured evaluation.
