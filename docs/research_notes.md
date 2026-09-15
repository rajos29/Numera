# Research Notes

## Current Stage

The project is in a Python learning-prototype stage.

This stage is intentionally not the final workstation benchmark platform. It is
being used to clarify the experiment vocabulary, data model, and numerical
interpretation before implementing native C or C++ kernels.

## Measured Result

The Python prototype can run a reproducible arithmetic experiment with:

- shared input samples
- multiple trials
- `float32` simulation
- `float64` Python floats
- raw CSV output
- SQLite observations
- JSON summary statistics
- machine profile metadata
- generated figures when Matplotlib and Seaborn are installed

## Interpretation

The first visual pass made multiplication appear to have the most error.

After separating absolute error from relative error, the better interpretation
is:

- multiplication has larger absolute error because it produces much larger
  result magnitudes for the current input range
- multiplication and division have similar relative error in the current
  moderate-magnitude `float32` sample set

This is an important early lesson: error must be interpreted together with
operation semantics and result scale.

## Design Decision

The prototype now uses an explicit experiment configuration file:

```text
experiments/arithmetic_python.json
```

This keeps experiment definitions machine-readable and makes figures traceable
to configuration plus raw data.

## Future Idea

Elementary functions such as `sqrt`, `sin`, `cos`, `exp`, and `log` are a
natural future characterization target.

They should not become the official benchmark milestone until baseline
arithmetic and the native backend are credible.

An exploratory Python-only elementary-function lab may still be useful for
learning domain restrictions, input magnitude effects, absolute error, relative
error, and composition effects.

## Native Backend Progress

A first native C++ arithmetic backend has been scaffolded.

It is intended to time arithmetic kernels while Python continues to handle
configuration, shared input generation, data aggregation, plotting, and reports.

Previous blocker:

```text
No C++ compiler was available on PATH, and the attempted conda-forge compiler
install failed because conda could not reach the package repository.
```

Current status:

```text
MSVC Build Tools are installed and the native backend builds through VsDevCmd.
```

The generated assembly listing contains expected scalar floating-point
instructions for the measured arithmetic operations.
