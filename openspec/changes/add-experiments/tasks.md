# Tasks

## 1. Runs

- [ ] 1.1 Add model sources and overrides by factory argument and dotted path, and verify a factory and a system give
  the same run for the same overrides
- [ ] 1.2 Add seed spawning per run and serial, thread, and process executors, and verify the reproducible and
  independent ensemble scenarios

## 2. Designs and scenarios

- [ ] 2.1 Add grid, random, Latin hypercube, and Sobol designs with crossing and concatenation, and verify the orbit
  diagram sweep scenario
- [ ] 2.2 Add scenarios with overrides and scheduled changes, and verify the policy-at-a-time scenario

## 3. Results

- [ ] 3.1 Add `ExperimentResult` with the tidy table, kept series, warm-up removal, `summary`, and `to_frame`, and
  verify the summary scenario

## 4. Sensitivity and calibration

- [ ] 4.1 Add one-at-a-time, Morris, and Sobol sensitivity through SALib, and verify the Ishigami scenario
- [ ] 4.2 Add calibration with least squares and likelihood losses, ensembles for stochastic models, and fit
  statistics, and verify the logistic recovery scenario

## 5. Documentation and gate

- [ ] 5.1 Add `SALib` to the dependencies, record the new layer in the Project Profile, and document experiments in
  the user guide, and verify the documentation test passes
- [ ] 5.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
