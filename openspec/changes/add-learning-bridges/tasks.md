# Tasks

## 1. Dependencies and approval

- [ ] 1.1 Ask the maintainer to approve the subpackage `skcomplex.systems.general.environments`, and record the answer
  in `design.md` before task 5 starts
- [ ] 1.2 Add `gymnasium>=1.4.0` and `pettingzoo>=1.27.0` to the runtime dependencies in `pyproject.toml` and the
  Project Profile, with their release dates, and verify `pdm install` resolves and both import

## 2. scikit-learn estimators

- [ ] 2.1 Add `IdentificationEstimator` in `src/skcomplex/core/experiments/_estimators.py` with `fit`, `predict`, and
  `score` on lists of runs, export it, and verify the cross-validation and estimator-checks scenarios
- [ ] 2.2 Verify the grid-search scenario with nested system parameters and learning rates
- [ ] 2.3 Add `SurrogateEstimator` on any analysis whose result holds a system, and verify the threshold and
  analysis-without-a-system scenarios
- [ ] 2.4 Document both estimators in the experiments page of the user guide, stating that they are the only classes
  with `fit`, with a runnable cross-validation example

## 3. Gymnasium

- [ ] 3.1 Add `make_environment` in `src/skcomplex/core/engine/runtime/_environments.py` with the spaces, `reset`, and
  `step` through the stepwise runtime, export it, and verify the checker and same-seed scenarios
- [ ] 3.2 Refuse systems whose policy sets the actions, and verify the policy-already-in-the-system scenario
- [ ] 3.3 Verify the cart-pole agreement scenario against `CartPole-v1` in `tests/systems/engineering/control/`

## 4. PettingZoo

- [ ] 4.1 Add `make_parallel_environment` with one agent per policy or per member, and verify the API-test scenario

## 5. Environments as systems

- [ ] 5.1 Add `GymnasiumSystem` in the approved subpackage with its element, update law, and environment, and verify
  the policy-trained-on-Gymnasium scenario, marked slow, and the not-differentiable scenario
- [ ] 5.2 Add `PettingZooSystem` with one element per agent, and verify that a policy bound to its agents runs one
  episode of a PettingZoo test environment

## 6. Documentation

- [ ] 6.1 Document Gymnasium and PettingZoo environments both ways in the experiments page of the user guide, and add
  the gallery example `docs/examples/engineering/control/plot_cart_pole_gymnasium.py` that trains a policy on Gymnasium's
  cart-pole and runs a random Gymnasium agent on `CartPoleSystem`, and verify the gallery build runs it

## 7. Checks

- [ ] 7.1 Run `nox` (formatting, checks, tests of the touched modules with the thread limits set to 1, and the strict
  docs build) and verify every session passes
