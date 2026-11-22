## 1. Prerequisite

- [ ] 1.1 Check that `add-learning-foundation` and `add-neural-particle-simulators` are applied, so observed runs,
  `split_runs`, `forecast_skill`, and `force_matching` exist, and stop if they are not

## 2. Dependencies

- [ ] 2.1 Add `torch>=2.15.0`, `sbi>=0.27.0`, `neuraloperator>=2.0.0`, `chronos-forecasting>=2.3.2`,
  `mace-torch>=0.3.17`, and `deepxde>=1.15.1` to the runtime dependencies of `pyproject.toml`. Verify that `pdm lock`
  resolves with the current numpy, scikit-learn, and Python pins
- [ ] 2.2 Add the `pytorch-cpu` source with `include_packages = ["torch"]` to `pyproject.toml`. Verify that the lock
  file names the CPU wheels of torch for Linux
- [ ] 2.3 Record the new dependencies with their release dates, the dropped torchmd-net, and the exceptions for the stale
  transitive packages with their review date in the Project Profile of `openspec/constitution.md`. Verify with the
  dependency audit of the gate
- [ ] 2.4 Set `HF_HUB_OFFLINE=1` and cap `OMP_NUM_THREADS` and torch threads in the test sessions of `noxfile.py`.
  Verify that `nox -s quick_tests` runs with no network access
- [ ] 2.5 Measure the import time of `skcomplex.core.analysis` before and after the new dependencies and note it for the
  changelog

## 3. Neural calibration

- [ ] 3.1 Add `'npe'`, `'nle'`, and `'nre'` to the method and loss table of
  `src/skcomplex/core/experiments/_calibration.py` and to `CalibrationMethodKind` in `src/skcomplex/core/enums.py`.
  Verify with a test in `tests/core/experiments/test_calibration.py` that an unknown method lists them
- [ ] 3.2 Add the batched summary simulator to `_CalibrationObjective` and the map of SciPy priors to torch
  distributions. Test the map of every supported family and the error for another one
- [ ] 3.3 Train and sample the three estimators with seeded torch state. Test the stochastic SIR and logistic scenarios
  of the spec, marked slow where they pass 10 seconds
- [ ] 3.4 Add the `estimator` field to `CalibrationResult`, reuse with `budget=0`, and the parameter check. Test the
  amortized scenario and the mismatched estimator
- [ ] 3.5 Add `rounds` and the refinement of a pretrained estimator. Test the Brock-Hommes rounds scenario and the
  rejection of a tied estimator for other data
- [ ] 3.6 Add `embedding='mlp'` and `'cnn'`. Test the learned embedding scenario
- [ ] 3.7 Add the losses to `diagnostics`, and one amortized training with TARP and SBC ranks to `posterior_coverage`.
  Test the amortized coverage scenario and that the system runs only the expected number of times in
  `tests/core/experiments/test_coverage.py`
- [ ] 3.8 Document the neural methods in `docs/overview/user_guide/run_experiments.md` and add a gallery example of
  amortized NPE on the stochastic SIR system. Verify with the docs build

## 4. Neural operators

- [ ] 4.1 Write `neural_operator` and `NeuralOperatorResult` in `src/skcomplex/core/analysis/_neural_operator.py`,
  reading trajectories, experiment results, and observed runs of fields, with validation runs held out by
  `split_runs`, and export them. Test the Gray-Scott scenario and the error for a source without fields in
  `tests/core/analysis/test_neural_operator.py`
- [ ] 4.2 Build the surrogate `system` as a private ready-made system with `_OperatorStep` and the model parameter
  `shape`. Test that it prepares and simulates, and the finer-grid Kuramoto-Sivashinsky scenario
- [ ] 4.3 Take swept parameters as input channels and as model parameters of the surrogate system. Test the sweep over
  `F` scenario
- [ ] 4.4 Document it in `docs/overview/user_guide/analyse.md` and add a gallery example of a Gray-Scott surrogate.
  Verify with the docs build

## 5. Pretrained forecasts

- [ ] 5.1 Add a test fixture that saves a tiny Chronos-Bolt model with random weights to a temporary directory
- [ ] 5.2 Write `pretrained_forecast` and `PretrainedForecastResult` in
  `src/skcomplex/core/analysis/_pretrained_forecast.py` and export them. Test the shapes and quantile order on a Lorenz
  series with the tiny model, and the context error, in `tests/core/analysis/test_pretrained_forecast.py`
- [ ] 5.3 Add the comparison with `truth` through `forecast_skill`, `lyapunov`, and `scaling`. Test that the statistics
  are filled and that the skill equals `forecast_skill` of the same forecast and truth
- [ ] 5.4 Raise `AnalysisError` for a missing model offline. Test it with `HF_HUB_OFFLINE=1`, and add one slow test of
  the default model that is skipped unless it is cached
- [ ] 5.5 Document it in `docs/overview/user_guide/analyse.md`, including the first download, and add a gallery example
  that forecasts the Lorenz system zero-shot. Verify with the docs build

## 6. Equivariant potentials

- [ ] 6.1 Write `EquivariantPotentialInteraction` in `src/skcomplex/systems/general/mechanics/_equivariant_potential.py`
  and export it from `skcomplex.systems.general.mechanics`. Test the gradient, rotation, and kinds scenarios with a
  small untrained MACE model in `tests/systems/general/mechanics/test_equivariant_potential.py`
- [ ] 6.2 Read members by the minimum image and record the potential as an observable. Test the boundary and the
  energy drift scenarios
- [ ] 6.3 Test that `LammpsSolver` runs the interaction through its existing `fix external` with no change to
  `src/skcomplex/core/engine/solvers/_lammps.py`, and that LAMMPS and velocity Verlet agree within 1e-6 over 100
  steps, in `tests/core/engine/solvers/test_lammps.py`
- [ ] 6.4 Write `MaceInteraction` in `src/skcomplex/systems/physics/common/particles/_mace.py`. Test the map of symbols
  to kinds with a saved small model in `tests/systems/physics/common/particles/test_mace.py`
- [ ] 6.5 Add `kind='equivariant'` to `force_matching` in `src/skcomplex/core/analysis/_force_matching.py`, whose result
  system calls the private `_EquivariantForces` of that module. Test the Lennard-Jones, dimer, and bead-kinds scenarios
  in `tests/core/analysis/test_force_matching.py`
- [ ] 6.6 Document the interaction in the user guide page on building a custom system

## 7. Physics-informed solver

- [ ] 7.1 Write `PinnSolver` in `src/skcomplex/core/engine/solvers/_pinn.py` with the backend setting and its check.
  Test the error when deepxde was imported with another backend in `tests/core/engine/solvers/test_pinn.py`
- [ ] 7.2 Write `_PinnResidual` with the torch printer of field laws and the support check. Test the noise-law error
  and the derivative of a known expression
- [ ] 7.3 Add the boundaries and Fourier features. Test the heat, Allen-Cahn, and periodic scenarios, marked slow where
  they pass 10 seconds
- [ ] 7.4 Add `method='pinn'` to `calibrate`, which trains a physics-informed network of the field system with the
  calibrated parameters as trainable variables of deepxde and the observed runs as point constraints, the inverse mode
  of the solver, and verify the diffusion-coefficient-from-sparse-samples scenario, marked slow
- [ ] 7.5 Document the solver in `docs/overview/user_guide/simulate.md` and add a gallery example of Gray-Scott with the
  physics-informed and spectral solvers side by side. Verify with the docs build

## 8. Systems

- [ ] 8.1 Write `MaceWaterSystem` in `src/skcomplex/systems/physics/statistical_mechanics/_mace_water.py` and export it
  from the subfield. Test the structure scenario with a saved small model, and the liquid-structure scenario as a slow
  test skipped unless the default model is cached, in `tests/systems/physics/statistical_mechanics/test_mace_water.py`
- [ ] 8.2 Test the MACE water system prepared with `LammpsSolver` and its thermostat, with a saved small model
- [ ] 8.3 Write `CgProteinFoldingSystem` with its native coordinates file, read with `importlib.resources.files`, in
  `src/skcomplex/systems/physics/soft_matter/_cg_proteins.py` and export it. Test the fine folding scenario and
  `chain_shape` on both resolutions in `tests/systems/physics/soft_matter/test_cg_proteins.py`
- [ ] 8.4 Test the coarse model fitted by force matching against the fine model, as a slow test
- [ ] 8.5 Add `mace_water_md` and `cg_protein_folding` to `docs/overview/inventory.toml` with their sources and
  primitives, and a gallery example for each. Verify with the inventory check and the strict gallery coverage

## 9. Integration

- [ ] 9.1 Update the changelog with the new methods, functions, solver, systems, the `CalibrationResult` field, the
  dependencies, the install size, and the CPU-only install steps
- [ ] 9.2 Run `nox` checks, tests, and the strict docs build, and fix every finding
