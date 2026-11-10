## Why

The JAX stack covers point fits, ABC, MCMC, and small learned laws, but the mature tools for neural posteriors, neural
operators, pretrained forecasters, machine-learned interatomic potentials, and physics-informed networks are built on
torch (2508.12939, 2412.10354, 2409.15771, 2401.00096, 2308.08468). The user approved torch as a runtime dependency, so
these methods can wrap the libraries the papers use instead of reimplementing them.

## What Changes

- **Neural calibration.** `calibrate` gains the methods `'npe'`, `'nle'`, and `'nre'` beside the methods of
  `add-bayesian-calibration`. They train a neural posterior, likelihood, or ratio estimator on runs from prior draws
  with sbi. One trained estimator is reused for new observations without new runs (amortized), or refined for one
  observation over sequential rounds (2601.07413). Diagnostics add the expected coverage test (TARP) and the ranks of
  simulation-based calibration, and `posterior_coverage` trains one estimator for every truth.
- **BREAKING: `CalibrationResult` gains `estimator`**, the trained estimator of a neural method, `None` otherwise.
- **Neural operators.** New analysis function `neural_operator(source, ...)` learns a field-to-field map with
  neuraloperator (FNO by default) from runs of a field system, such as the runs of a `sweep` or the observed runs of `add-learning-foundation`,
  holding out validation runs with `split_runs`. Its result holds a
  `system` that steps the learned map under discrete time, on any grid size. Probes: `GrayScottSystem` and
  `KuramotoSivashinskySystem`.
- **Pretrained forecasts.** New analysis function `pretrained_forecast(source, ...)` forecasts series zero-shot with a
  pretrained Chronos model (chronos-forecasting). Given the true continuation, it reports the error by lead
  time and the valid prediction time through `forecast_skill` of `add-learning-foundation`, and the largest Lyapunov
  exponent and correlation dimension of forecast and truth. Weights download
  from the Hugging Face Hub on first use, and tests never touch the network.
- **Equivariant potentials.** New core interaction `EquivariantPotentialInteraction` moves the points of a group by the
  forces of an equivariant many-body network potential (mace-torch), pretrained or trained. It runs with the native
  particle solvers and in LAMMPS through `fix external`.
- **Coarse-grained network potentials.** `force_matching` of `add-neural-particle-simulators` gains
  `kind='equivariant'`, which fits an equivariant network potential plus a prior to mapped forces.
- **Physics-informed solver.** New solver `PinnSolver` (deepxde) solves field systems from their declared rate laws by
  a physics-informed network, with Fourier features and periodic or fixed boundaries (2308.08468). Its inverse mode is
  `calibrate(method='pinn')`, which fits parameters of a field system to sparse observed runs with the same network,
  the main use of physics-informed networks in the review of Cuomo et al. (2201.05624).
- **Systems.** New `MaceWaterSystem` (`mace_water_md`, physics/statistical_mechanics) after 2401.00096, and
  `CgProteinFoldingSystem` (`cg_protein_folding`, physics/soft_matter) after 2310.18278.
- **Dependencies.** torch, sbi, neuraloperator, chronos-forecasting, mace-torch, and deepxde become runtime
  dependencies. torchmd-net is dropped, see design.md. Development and CI install CPU-only torch wheels.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `experiments`: neural posterior, likelihood, and ratio calibration, physics-informed calibration of fields, amortized reuse, sequential refinement, and
  neural coverage diagnostics.
- `data-driven-dynamics`: neural-operator surrogates of field systems.
- `trajectory-analysis`: zero-shot forecasts by pretrained models compared by attractor statistics.
- `general-mechanisms`: the equivariant network potential interaction.
- `external-backends`: LAMMPS runs equivariant network potentials.
- `pde-solvers`: the physics-informed network solver.
- `particle-surrogates`: force matching with an equivariant network potential.
- `model-library`: the MACE water and coarse-grained protein systems, with inventory entries and gallery examples.

## Impact

- Core: `src/skcomplex/core/experiments/_calibration.py` and `_coverage.py` (from `add-bayesian-calibration`), new
  `core/analysis/_neural_operator.py` and `_pretrained_forecast.py`, new
  `core/components/interactions/_equivariant_potential.py`, `core/analysis/_force_matching.py` (from
  `add-neural-particle-simulators`), new `core/engine/solvers/_pinn.py`, and `core/engine/solvers/_lammps.py`.
- Systems: new modules `physics/statistical_mechanics/_mace_water.py` and `physics/soft_matter/_cg_proteins.py`, with
  inventory entries and gallery examples.
- API: `calibrate` gains four methods, the three neural ones and `'pinn'`, and the arguments `estimator`, `rounds`, and `embedding`. `CalibrationResult`
  gains `estimator`.
- Dependencies: torch 2.15.0, sbi 0.27.0, neuraloperator 2.0.0, chronos-forecasting 2.3.2, mace-torch 0.3.17, and
  deepxde 1.15.1, with a CPU-only torch source for the lock file. The install grows by several hundred megabytes on
  CPU and by several gigabytes with the default CUDA wheels on Linux.
- Depends on `add-bayesian-calibration` (methods, priors, result, coverage) and `add-neural-particle-simulators`
  (`force_matching`, particle `coarse_grain`), and on `add-learning-foundation`: `calibrate` and `neural_operator` read
  its `ObservedRunsSpec`, validation runs are held out by `split_runs`, forecasts are scored by `forecast_skill`, and
  results follow the convention `system`, `training_error`, `validation_error`. Torch models train in the loops of
  their own libraries, the one stated exception to the shared Optax loop, which serves JAX models.
