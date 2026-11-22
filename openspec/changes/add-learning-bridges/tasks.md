# Tasks

## 1. Dependencies

- [ ] 1.1 Add `gymnasium>=1.4.0`, `pettingzoo>=1.27.0`, `arviz>=1.3.0`, and `statsmodels>=0.15.0` (unless
  `extend-network-and-emergence-measures` added it first) to the runtime dependencies in `pyproject.toml` and the
  Project Profile, with their release dates, and verify `pdm install` resolves and all import

## 2. Model selection

- [ ] 2.1 Check that `add-learning-foundation` is applied, so observed runs, `identify`, `EnvironmentSpec`,
  `PolicyInteraction`, `train_policy`, and `make_arrays` exist, and stop if it is not
- [ ] 2.2 Add the private selection in `src/skcomplex/core/experiments/_selection.py`, which expands `param_grid` with
  `ParameterGrid` or reads a `ParameterSampler`, splits with `cv` and `groups`, reads `scoring`, and builds
  `cv_results`, and verify it in `tests/core/experiments/test_selection.py`
- [ ] 2.3 Add `cv`, `groups`, `scoring`, and `param_grid` to `identify`, refit on all runs with the best setting, add
  `cv_results`, `best_params`, and `best_score` to `IdentificationResult`, and verify the folds-over-runs,
  grid-of-learning-rates, same-seed, and no-selection scenarios in `tests/core/experiments/test_identification.py`
- [ ] 2.4 Add the same arguments and fields to `calibrate` and its result, scoring the point estimate of every method,
  and verify the groups-kept-whole scenario in `tests/core/experiments/test_calibration.py`
- [ ] 2.5 Add `conditions`, `evaluation`, and the same arguments and fields to `train_policy` and
  `PolicyTrainingResult`, and verify the pole-lengths-held-out scenario, marked slow, and the
  selection-without-conditions scenario in `tests/core/experiments/test_policy_training.py`
- [ ] 2.6 Add `split='time'` to the selection of `identify` and `calibrate`, scoring test parts by runs from the start
  of their run, and verify the selection-on-one-run and shuffled-times scenarios
- [ ] 2.7 Add `log_likelihood`, `aic`, and `bic` to `IdentificationResult` and `CalibrationResult`, and verify the
  criteria-of-a-Gaussian-fit scenario
- [ ] 2.8 Record the pointwise log-likelihoods of `calibrate(method='mcmc')`, add `loo` through `arviz_stats.loo` and
  `waic` by its formula, and verify the missing-term scenario
- [ ] 2.9 Add `residuals` to both results, and verify the residuals-of-a-run scenario
- [ ] 2.10 Add `whiteness` and `WhitenessResult` in `src/skcomplex/core/analysis/_whiteness.py` on
  `acorr_ljungbox`, export them with their analysis tags, and verify the white-noise, unmodelled-dynamics, and
  runs-pooled scenarios in `tests/core/analysis/test_whiteness.py`
- [ ] 2.11 Add `replicates` to `identify` with `replicate_systems` and `spread`, and verify the spread-of-a-network
  scenario
- [ ] 2.12 Verify the rival-models-of-the-pelts scenario on the probe of the foundation
- [ ] 2.13 Document model selection in the calibrate and experiments pages of the user guide, with the decision that
  systems are not scikit-learn estimators, a runnable search over learning rates with `GroupKFold`, a selection by
  time on one run, the criteria, LOO and WAIC, the residual tests, replicates, and a pipeline trained on the arrays of
  `make_arrays`, and verify their code blocks run

## 3. Gymnasium

- [ ] 3.1 Add `make_environment` in `src/skcomplex/core/systems/_environments.py`, a module above `_system` in the
  layers contract of `skcomplex.core.systems` in `pyproject.toml`, with the spaces, `reset`, and `step` through the
  private `Simulation` of the runtime and its `extend`, export it from `skcomplex.core.systems`, and verify the
  same-seed scenario in `tests/core/systems/test_environments.py`
- [ ] 3.2 Add the environments of `SIRVaccinationSystem` in
  `src/skcomplex/systems/biology/epidemiology/_vaccination.py` and of `PendulumSwingUpSystem` in
  `src/skcomplex/systems/engineering/control/_control.py` as class attributes, and verify the checker scenario in
  `tests/core/systems/test_environments.py`
- [ ] 3.3 Refuse systems whose policy sets the actions, and verify the policy-already-in-the-system scenario
- [ ] 3.4 Verify the cart-pole agreement scenario against `CartPole-v1` in
  `tests/systems/engineering/control/test_cart_pole.py`

## 4. PettingZoo

- [ ] 4.1 Add `make_parallel_environment` in `src/skcomplex/core/systems/_environments.py` with one agent per policy
  or per member, and verify the API-test scenario in `tests/core/systems/test_environments.py`

## 5. Environments as systems

- [ ] 5.1 Create the subpackage `src/skcomplex/systems/general/environments` with its `__init__.py` and the mirrored
  tests, and add it to the systems pages of the user guide
- [ ] 5.2 Add `GymnasiumSystem` in `src/skcomplex/systems/general/environments/_gymnasium.py` with its element, its
  stepping interaction, and its environment, export it, and verify the policy-trained-on-Gymnasium scenario, marked
  slow, and the not-differentiable scenario in `tests/systems/general/environments/test_gymnasium.py`
- [ ] 5.3 Add `PettingZooSystem` in `src/skcomplex/systems/general/environments/_petting_zoo.py` with one element per
  agent, export it, and verify that a policy bound to its agents runs one episode of a PettingZoo test environment in
  `tests/systems/general/environments/test_petting_zoo.py`

## 6. Documentation

- [ ] 6.1 Document Gymnasium and PettingZoo environments both ways in the experiments page of the user guide, and add
  the gallery example `docs/examples/engineering/control/plot_cart_pole_gymnasium.py` that trains a policy on Gymnasium's
  cart-pole and runs a random Gymnasium agent on `CartPoleSystem`, and verify the gallery build runs it

## 7. Checks

- [ ] 7.1 Run `nox` (formatting, checks, import contracts, tests of the touched modules with the thread limits set to
  1, and the strict docs build) and verify every session passes
