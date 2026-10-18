# Tasks

## 1. Runs

- [x] 1.1 Add model sources and overrides by factory argument and dotted path, and verify a factory and a system give
  the same run for the same overrides
- [x] 1.2 Add seed spawning per run and serial, thread, and process executors, and verify the reproducible and
  independent ensemble scenarios

## 2. Designs and scenarios

- [x] 2.1 Add grid, random, Latin hypercube, and Sobol designs with crossing and concatenation, and verify the orbit
  diagram sweep scenario
- [x] 2.2 Add scenarios with overrides and scheduled changes, and verify the policy-at-a-time scenario

## 3. Results

- [x] 3.1 Add `ExperimentResult` with the tidy table, kept series, warm-up removal, `summary`, and `to_frame`, and
  verify the summary scenario

## 4. Sensitivity and calibration

- [x] 4.1 Add one-at-a-time, Morris, and Sobol sensitivity through SALib, and verify the Ishigami scenario
- [x] 4.2 Add calibration with least squares and likelihood losses, ensembles for stochastic models, and fit
  statistics, and verify the logistic recovery scenario

## 5. Documentation and gate

- [x] 5.1 Add `SALib` to the dependencies, record the new layer in the Project Profile, and document experiments in
  the user guide, and verify the documentation test passes
- [x] 5.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
