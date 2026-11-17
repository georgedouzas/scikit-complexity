# Tasks

## 1. Prerequisite

- [ ] 1.1 Check that `add-learning-foundation` is applied, so `ObservationSpec`, `PolicyInteraction`, and
  `train_policy` exist, and stop if it is not
- [ ] 1.2 Check the locally D-optimal design of the Michaelis-Menten rate against Duggleby (1979), fetching the paper
  into `references/papers`, or replace the scenario by an example of 2302.14545, and record the source in the test

## 2. Optimal designs

- [ ] 2.1 Add `DesignResult` and `design_experiment` in `src/skcomplex/core/experiments/_experimental_design.py` with
  the checks of variables, criteria, and sizes, export them, and verify the unknown-variable scenario in
  `tests/core/experiments/test_experimental_design.py`
- [ ] 2.2 Write the Fisher information from `jax.jacfwd` of the traced run, with central differences for systems that
  are not differentiable, and verify it against the analytic information of exponential decay
- [ ] 2.3 Add `'d-optimal'` and `'a-optimal'` with the exchange algorithm over candidates and CMA-ES within bounds, and
  verify the decay-sampling-time and Michaelis-Menten-rates scenarios
- [ ] 2.4 Add `'eig'` with the nested Monte Carlo estimator over a prior, and verify the linear-Gaussian-information-gain
  and same-seed scenarios
- [ ] 2.5 Add `rounds` and `policy` with the private adaptive design system and its sequential information gain reward,
  trained by `train_policy`, and verify the location-finding scenario, marked slow, and the adaptive-experiment-runs
  scenario

## 3. Optimisation

- [ ] 3.1 Add `control` and `switch` to `MinimumTimeControlSystem`, and verify the switch-of-the-Pontryagin-solution and
  feedback-unchanged scenarios in `tests/systems/engineering/control/`
- [ ] 3.2 Add `OptimizationResult` and `optimize` in `src/skcomplex/core/experiments/_optimization.py` with
  `'bayesian'` on Optuna's `TPESampler`, `direction`, `replicates`, `constraints`, and the executor, export them, and
  verify the Bayesian part of the switch-time scenario and the same-seed scenario
- [ ] 3.3 Add `'cma'` on pymoo's `CMAES`, and verify the CMA-ES part of the switch-time scenario
- [ ] 3.4 Add `'nsga2'` on pymoo's `NSGA2` with the front and the point closest to the ideal point, and verify the
  two-objectives and several-objectives-with-one-objective-method scenarios

## 4. Structural identifiability

- [ ] 4.1 Add `method='structural'` to `identifiability` with the Taylor coefficients by Sage, the exact rank at random
  rational points, and the field `rank`, and verify the sum-of-two-rates, predator-and-prey, and
  laws-that-are-not-symbolic scenarios in `tests/core/experiments/test_identifiability.py`

## 5. Documentation

- [ ] 5.1 Document `design_experiment`, adaptive design, `optimize`, and structural identifiability in a design section
  of the experiments page of the user guide, with runnable examples, and verify the code blocks run
- [ ] 5.2 Add the gallery examples `docs/examples/biology/epidemiology/plot_optimal_sampling_times.py`, choosing the
  sampling times of a decay and of an outbreak, and `docs/examples/engineering/control/plot_switch_time_search.py`,
  comparing the three optimisers on the switch time, and verify the gallery build runs both
- [ ] 5.3 Add the inventory primitives `experimental-design` and `optimisation` with
  `change = "add-experimental-design"`, mark `minimum_time_control` with `optimisation`, and verify the inventory check
  passes

## 6. Checks

- [ ] 6.1 Run `nox` (formatting, ruff, mypy, pydoclint, import contracts, tests of the touched modules with the thread
  limits set to 1, doctests, the inventory check, and the strict docs build) and verify every session passes
