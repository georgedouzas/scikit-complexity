## 0. Prerequisite

- [ ] 0.1 Check that `add-learning-foundation` is applied, so observed runs, `split_runs`, and `forecast_skill` exist,
  and stop if it is not

## 1. Dependencies

- [ ] 1.1 Add `torch>=2.15.0`, `sbi>=0.27.0`, `neuraloperator>=2.0.0`, `chronos-forecasting>=2.3.2`,
  `mace-torch>=0.3.17`, and `deepxde>=1.15.1` to the runtime dependencies of `pyproject.toml`. Verify that `pdm lock`
  resolves with the current numpy, scikit-learn, and Python pins
- [ ] 1.2 Add the `pytorch-cpu` source with `include_packages = ["torch"]` to `pyproject.toml`. Verify that the lock
  file names the CPU wheels of torch for Linux
- [ ] 1.3 Record the new dependencies with their release dates, the dropped torchmd-net, and the exceptions for the stale
  transitive packages with their review date in the Project Profile of `openspec/constitution.md`. Verify with the
  dependency audit of the gate
- [ ] 1.4 Set `HF_HUB_OFFLINE=1` and cap `OMP_NUM_THREADS` and torch threads in the test sessions of `noxfile.py`.
  Verify that `nox -s quick_tests` runs with no network access
- [ ] 1.5 Measure the import time of `skcomplex.core.analysis` before and after the new dependencies and note it for the
  changelog

## 2. Neural calibration

- [ ] 2.1 Add `'npe'`, `'nle'`, and `'nre'` to the method and loss table of `core/experiments/_calibration.py`. Verify
  with a test that an unknown method lists them
- [ ] 2.2 Add the batched summary simulator to `_CalibrationObjective` and the map of SciPy priors to torch
  distributions. Test the map of every supported family and the error for another one
- [ ] 2.3 Train and sample the three estimators with seeded torch state. Test the stochastic SIR and logistic scenarios
  of the spec, marked slow where they pass 10 seconds
- [ ] 2.4 Add the `estimator` field to `CalibrationResult`, reuse with `budget=0`, and the parameter check. Test the
  amortized scenario and the mismatched estimator
- [ ] 2.5 Add `rounds` and the refinement of a pretrained estimator. Test the Brock-Hommes rounds scenario and the
  rejection of a tied estimator for other data
- [ ] 2.6 Add `embedding='mlp'` and `'cnn'`. Test the learned embedding scenario
- [ ] 2.7 Add the losses to `diagnostics`, and one amortized training with TARP and SBC ranks to `posterior_coverage`.
  Test the amortized coverage scenario and that the system runs only the expected number of times
- [ ] 2.8 Document the neural methods in `docs/overview/user_guide/run_experiments.md` and add a gallery example of
  amortized NPE on the stochastic SIR system. Verify with the docs build

## 3. Neural operators

- [ ] 3.1 Write `neural_operator` and `NeuralOperatorResult` in `core/analysis/_neural_operator.py`, reading
  trajectories, experiment results, and observed runs of fields, with validation runs held out by `split_runs`, and
  export them. Test the Gray-Scott scenario and the error for a source without fields
- [ ] 3.2 Build the surrogate `system` with `_OperatorStep` and a `shape` argument. Test that it validates and runs,
  and the finer-grid Kuramoto-Sivashinsky scenario
- [ ] 3.3 Take swept parameters as input channels and model parameters. Test the sweep over `F` scenario
- [ ] 3.4 Document it in `docs/overview/user_guide/analyse.md` and add a gallery example of a Gray-Scott surrogate.
  Verify with the docs build

## 4. Pretrained forecasts

- [ ] 4.1 Add a test fixture that saves a tiny Chronos-Bolt model with random weights to a temporary directory
- [ ] 4.2 Write `pretrained_forecast` and `PretrainedForecastResult` in `core/analysis/_pretrained_forecast.py` and
  export them. Test the shapes and quantile order on a Lorenz series with the tiny model, and the context error
- [ ] 4.3 Add the comparison with `truth` through `forecast_skill`, `lyapunov`, and `scaling`. Test that the statistics
  are filled and that the skill equals `forecast_skill` of the same forecast and truth
- [ ] 4.4 Raise `AnalysisError` for a missing model offline. Test it with `HF_HUB_OFFLINE=1`, and add one slow test of
  the default model that is skipped unless it is cached
- [ ] 4.5 Document it in `docs/overview/user_guide/analyse.md`, including the first download, and add a gallery example
  that forecasts the Lorenz system zero-shot. Verify with the docs build

## 5. Equivariant potentials

- [ ] 5.1 Write `EquivariantPotentialInteraction` in `core/components/interactions/_equivariant_potential.py` and
  export it. Test the gradient, rotation, and kinds scenarios with a small untrained MACE model
- [ ] 5.2 Read members by the minimum image and record the potential as an observable. Test the boundary and the
  energy drift scenarios
- [ ] 5.3 Run the interaction through `fix external` in `core/engine/solvers/_lammps.py`. Test that LAMMPS and velocity
  Verlet agree within 1e-6 over 100 steps
- [ ] 5.4 Write `MaceInteraction` in `physics/common/particles/_mace.py`. Test the map of symbols to kinds with a saved
  small model
- [ ] 5.5 Add `kind='equivariant'` to `force_matching` in `core/analysis/_force_matching.py`. Test the Lennard-Jones,
  dimer, and bead-kinds scenarios
- [ ] 5.6 Verify that the field-term test of `skcomplex.core.components` passes with the new names, and document the
  interaction in the user guide page on building a custom system

## 6. Physics-informed solver

- [ ] 6.1 Write `PinnSolver` in `core/engine/solvers/_pinn.py` with the backend setting and its check. Test the error
  when deepxde was imported with another backend
- [ ] 6.2 Write `_PinnResidual` with the torch printer of field laws and the support check. Test the noise-law error
  and the derivative of a known expression
- [ ] 6.3 Add the boundaries and Fourier features. Test the heat, Allen-Cahn, and periodic scenarios, marked slow where
  they pass 10 seconds
- [ ] 6.4 Document the solver in `docs/overview/user_guide/simulate.md` and add a gallery example of Gray-Scott with the
  physics-informed and spectral solvers side by side. Verify with the docs build

## 7. Systems

- [ ] 7.1 Write `MaceWaterSystem` in `physics/statistical_mechanics/_mace_water.py` and export it from the subfield.
  Test the structure scenario with a saved small model, and the liquid-structure scenario as a slow test skipped
  unless the default model is cached
- [ ] 7.2 Test the MACE water system in LAMMPS with its thermostat, with a saved small model
- [ ] 7.3 Write `CgProteinFoldingSystem` with its native coordinates file in `physics/soft_matter/_cg_proteins.py` and
  export it. Test the fine folding scenario and `chain_shape` on both resolutions
- [ ] 7.4 Test the coarse model fitted by force matching against the fine model, as a slow test
- [ ] 7.5 Add `mace_water_md` and `cg_protein_folding` to `docs/overview/inventory.toml` with their sources and
  primitives, and a gallery example for each. Verify with the inventory check and the strict gallery coverage

## 8. Integration

- [ ] 8.1 Update the changelog with the new methods, functions, solver, systems, the `CalibrationResult` field, the
  dependencies, the install size, and the CPU-only install steps
- [ ] 8.2 Run `nox` checks, tests, and the strict docs build, and fix every finding
