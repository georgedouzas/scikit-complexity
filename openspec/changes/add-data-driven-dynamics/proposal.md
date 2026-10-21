# Proposal

## Why

A complex systems toolkit is used as often to find a model from data as to simulate a known one. Brunton et al. (2021,
arXiv 2102.12086) review Koopman operator methods, dynamic mode decomposition and its extensions, which find linear
dynamics in observables of nonlinear systems, and Cestnik and Martens (2025, arXiv 2509.11338) show that
next-generation reservoir computing forecasts and reconstructs chaotic systems from short series. Sparse identification
of nonlinear dynamics (SINDy) finds symbolic laws. The core simulates systems it is given and trains parameters by
gradients in the pending `add-learning-dynamics`, but it cannot fit a model to trajectories and return a system it can
run.

## What Changes

- Add surrogate fitting: fit a model to one or more trajectories and return a registered-system-compatible element that
  `Simulation` runs.
- Add dynamic mode decomposition and extended DMD with a chosen dictionary, through PyDMD or PyKoopman, with modes,
  eigenvalues, and forecasts.
- Add SINDy through PySINDy, returning symbolic laws the compiler accepts.
- Add reservoir models, echo state networks and next-generation reservoirs, through ReservoirPy.
- Add the probe model `echo_state_network` and fitting examples on existing models.

## Capabilities

### New Capabilities

- `data-driven-dynamics`: fitting runnable surrogate systems to trajectories by DMD, SINDy, or reservoirs.

### Modified Capabilities

- `model-library`: the echo state network model.

## Impact

- `skcomplex.core.analysis` gains `fit_dmd`, `fit_sindy`, and `fit_reservoir`, each returning a fitted surrogate.
- PySINDy, PyDMD or PyKoopman, and ReservoirPy become runtime dependencies if the design's survey confirms them.
