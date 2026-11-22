# Tasks

## 1. Prerequisite and dependencies

- [ ] 1.1 Check that `add-learning-foundation` is applied, so `ObservationSpec`, observed runs, and `forecast_skill`
  exist, and stop if it is not
- [ ] 1.2 Add `filterpy>=2.0.0` and `scoringrules>=0.11.0` to the runtime dependencies in `pyproject.toml` and the
  Project Profile, with their release dates, and verify `pdm install` resolves and both import
- [ ] 1.3 Read the settings of the pendulum and car tracking examples from the companion code of Särkkä (2013), record
  them with their source in the test module docstrings, and stop if they differ from the design

## 2. Noisy pendulum

- [ ] 2.1 Add the model parameter `noise` to `PendulumSystem` and have `PendulumSwingInteraction` state a `NoiseLaw` on
  `bob.theta'` when it is positive, with the docstring citing Särkkä (Example 3.7), and verify the
  variance-of-the-velocity and default-unchanged scenarios in `tests/systems/physics/mechanics/test_mechanics.py`

## 3. Assimilation

- [ ] 3.1 Add `AssimilationResult` and `assimilate` in `src/skcomplex/core/experiments/_assimilation.py`, which clones
  the system, sets the conditions of the run with `set_params`, and prepares the clone once, with the checks of
  method, run, noise, and arguments, export them from `skcomplex.core.experiments`, and verify the several-runs,
  counts-with-a-Gaussian-filter, and prepares-once scenarios in `tests/core/experiments/test_assimilation.py`
- [ ] 3.2 Write the private process model that advances members from one observation time to the next with the solver
  of the prepared clone on its backend, inputs set, and the process noise from the noise laws or `process_noise`, and
  verify it against `system.simulate` for one member
- [ ] 3.3 Write the private measurement model from the readouts of the `ObservationSpec`, with `R` from Gaussian noise,
  logarithms for log-normal noise, and rows skipped for missing values, and verify it on a run with missing values
- [ ] 3.4 Add `'kalman'` with the linearity check and the RTS smoother, and verify the linear-tracking,
  smoothed-tracking, and nonlinear-system-with-the-Kalman-filter scenarios
- [ ] 3.5 Add `'ekf'` with `jax.jacfwd` of the traced step and its RTS smoother, and verify the EKF parts of the
  noisy-pendulum and smoothed-pendulum scenarios
- [ ] 3.6 Add `'ukf'` with Merwe sigma points and its RTS smoother, and verify the UKF parts of both pendulum scenarios
- [ ] 3.7 Add `'enkf'` with perturbed observations and inflation, refuse its smoother, and verify the
  ensemble-filter-on-Lorenz-96 (marked slow), ensemble-smoother, and same-seed scenarios
- [ ] 3.8 Add `'particle'` with the six noise kinds, the resampling schemes, the effective sizes, and the backward
  simulation smoother, and verify the particle parts of both pendulum scenarios, marked slow, and the same-seed
  scenario
- [ ] 3.9 Add `augmented` and `augmented_noise` with the estimates of the result, and verify the gravity-of-the-pendulum
  scenario
- [ ] 3.10 Return in the result's `system` a prepared clone whose initial state is the last estimate and whose
  augmented parameters are set to their estimates, and verify the forecast-from-the-last-estimate scenario
- [ ] 3.11 Check the DAPPER benchmark value of the ensemble filter on Lorenz-96 in the setting of Sakov and Oke (2008),
  record it with its filter variant and inflation, and tighten the bound of the Lorenz-96 scenario to it

## 4. Particle marginal Metropolis-Hastings

- [ ] 4.1 Add `PMMH = 'pmmh'` to `CalibrationMethodKind` in `src/skcomplex/core/enums.py` and `method='pmmh'` to
  `calibrate`, with the particle likelihood of `_assimilation.py` over every run, one prepared clone per proposal, and
  the adaptive random walk, and verify the noise-variance-of-the-pendulum and same-seed scenarios in
  `tests/core/experiments/test_calibration.py`
- [ ] 4.2 Verify the outbreak-with-jump-laws scenario on a stochastic SIR system, marked slow

## 5. Probabilistic scores

- [ ] 5.1 Accept ensemble forecasts in `forecast_skill` with `metric='crps'`, `coverage`, and `interval_scores` through
  scoringrules, and verify the Gaussian-ensemble, calibrated-coverage, and one-member scenarios in
  `tests/core/analysis/test_forecast_skill.py`

## 6. Documentation

- [ ] 6.1 Document `assimilate`, its methods, smoothers, joint estimation, and forecasts from the result, and
  `calibrate(method='pmmh')`, in an assimilation section of the calibrate page of the user guide, with runnable
  examples on the noisy pendulum
- [ ] 6.2 Add the gallery example `docs/examples/physics/mechanics/plot_noisy_pendulum_filters.py`, which tracks the
  noisy pendulum with every filter and smoother and tabulates their errors beside the values of the book, and
  `docs/examples/physics/climate/plot_lorenz96_assimilation.py`, which assimilates Lorenz-96 with the ensemble filter
  and scores its ensemble forecast by CRPS, and verify the gallery build runs both
- [ ] 6.3 Add the inventory primitive `state-estimation` with `change = "add-state-estimation"`, mark `pendulum` with
  it, and verify the inventory check passes

## 7. Checks

- [ ] 7.1 Run `nox` (formatting, ruff, mypy, pydoclint, import contracts, tests of the touched modules with the thread
  limits set to 1, doctests, the inventory check, and the strict docs build) and verify every session passes
