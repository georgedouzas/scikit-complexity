# Tasks

## 1. Mode decomposition with control

- [x] 1.1 Add `inputs`, `operator_`, and `input_matrix_` to `ModeDecompositionAnalysis` on PyDMD's `DMDc`, with the
  inputs as bounded adaptive parameters of the fitted system, and verify the linear system with input scenario

## 2. Models

- [x] 2.1 Add `szabo_cells` to `physics/active_matter` on `RateInteraction` and `neighbors` over a radius, and verify
  the collective order scenario
- [x] 2.2 Add `active_passive_flocking` to `physics/active_matter` on `PointElement`, `PairCouplingInteraction`, and
  `RateInteraction`, and verify the flock and the gathering of active agents scenarios
- [x] 2.3 Add `koopman_model_predictive_control` to `engineering/control` with the three plants, the fit, and the
  co-simulated controller, and verify the surrogate predictive control scenario for each plant
- [x] 2.4 Add `next_generation_reservoir` to `computer_science/neural_networks` with lagged states and the ridge
  readout taken from a fitted `ReservoirAnalysis(kind='next_generation')`, and verify the reservoir forecast and the
  agreement with `ReservoirAnalysis` scenarios

## 3. Gallery and gate

- [x] 3.1 Add a curated gallery example per model with a review row, set the four inventory entries to implemented with
  the tags of the systems they build, and verify the inventory check passes
- [ ] 3.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
