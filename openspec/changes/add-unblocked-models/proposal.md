# Proposal

## Why

Four inventory models wait on primitives of changes that are now archived, and no change builds them:
`szabo_cells` and `active_passive_flocking` in `physics/active_matter`, `koopman_model_predictive_control` in
`engineering/control`, and `next_generation_reservoir` in `computer_science/neural_networks`. The general core now
provides almost everything they need: `RateInteraction` with noise, `PairCouplingInteraction`, `PointElement`,
`neighbors` reductions over a `RadiusNeighborhood` in a periodic `ContinuousSpace`, `MapInteraction` with the increment
mode, states with lags, `CallSpec` values, `CoSimulationSolver` for an ODE partition beside a discrete-time partition,
and `ModeDecompositionAnalysis` and `ReservoirAnalysis` in `skcomplex.analysis`. The one missing piece is a mode
decomposition of a system with control inputs, which the Koopman controller needs, so this change adds it to
`ModeDecompositionAnalysis` as a general operation of the analysis layer.

## What Changes

- Add control inputs to `ModeDecompositionAnalysis`: with `inputs`, it fits dynamic mode decomposition with control
  through PyDMD's `DMDc`, on the variables or on the observables of its dictionary, and its fitted system takes the
  inputs as bounded adaptive parameters.
- Add `szabo_cells` and `active_passive_flocking` to `physics/active_matter`, built on `RateInteraction`,
  `PairCouplingInteraction`, `PointElement`, and `neighbors` reductions over a radius.
- Add `koopman_model_predictive_control` to `engineering/control`: a plant under continuous time co-simulated with a
  discrete-time controller whose `MapInteraction` sets the control from a `CallSpec` that solves the predictive problem
  on the fitted surrogate.
- Add `next_generation_reservoir` to `computer_science/neural_networks`: an element whose variables keep lags, and a
  `MapInteraction` in the increment mode whose function is a ridge readout of polynomial features.
- Give each model a curated gallery example with a review row, and set its inventory entry to implemented with the tags
  of the system it builds.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `data-driven-dynamics`: mode decomposition with control inputs.
- `model-library`: four models built on the general classes.

## Impact

- `skcomplex.analysis.ModeDecompositionAnalysis` gains `inputs` and the fitted input matrix.
- The model library, its tests, the gallery, and the inventory. No class of the core changes.
