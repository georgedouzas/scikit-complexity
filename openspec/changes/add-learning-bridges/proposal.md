## Why

`add-learning-foundation` makes identification, learned laws, and policy training work inside the library. The methods
and agents most users already have live in three ecosystems outside it. scikit-learn holds cross-validation, scorers,
and hyperparameter searches. Gymnasium is the interface of single-agent reinforcement learning, which agents of
stable-baselines3, skrl, and many papers read. PettingZoo is the same interface for many agents. The experiments of the
library should select their settings with the tools of scikit-learn, a system of the library should be usable as a
Gymnasium or PettingZoo environment, and an environment of those ecosystems should be usable as a system of the
library, so its policies train against it with `train_policy`.

Systems are not scikit-learn estimators. A system is the model being simulated, and `prepare` and `run` stay the only
verbs of the classes of the library, so scikit-learn enters inside the experiments, not around the systems.

## What Changes

- **Model selection inside the experiments.** `identify`, `calibrate`, and `train_policy` accept a scikit-learn
  splitter or a number of folds as `cv`, with `groups` for group splitters such as `GroupKFold`, a scorer as
  `scoring`, and a parameter grid as `param_grid`. `identify` and `calibrate` split observed runs, and `train_policy`
  splits the conditions its episodes start from, given by its new argument `conditions`. Every setting of the grid is
  fitted on the training part of every fold and scored on its test part, the result is fitted again on all the data
  with the best setting, and the frozen result reports `cv_results` with the score of every fold, `best_params`, and
  `best_score`.
- New function `make_environment(system, ...)` in `skcomplex.core.engine.runtime`, which returns a Gymnasium
  environment that steps the system through its `EnvironmentSpec`, so external agents run against any system.
- New function `make_parallel_environment(system, ...)` beside it, which returns a PettingZoo parallel environment
  with one agent per policy or per member of a group.
- New systems `GymnasiumSystem` and `PettingZooSystem`, which wrap an external environment as a system whose actions
  are adaptive parameters and whose observations are states, so `PolicyInteraction` and `train_policy` act on it. They
  are recipes of core parts in the new concept subpackage `skcomplex.systems.general.environments`, approved on
  2026-11-10.
- Dependencies: Gymnasium 1.4.0 and PettingZoo 1.27.0 become runtime dependencies.

## Capabilities

### New Capabilities

- `learning-bridges`: systems as Gymnasium and PettingZoo environments, and those environments as systems.

### Modified Capabilities

- `experiments`: cross-validation, scorers, and parameter grids in `identify` and `calibrate`.
- `reinforcement-learning`: cross-validation over conditions, scorers of returns, and parameter grids in
  `train_policy`.
- `model-library`: the cart-pole system agrees step by step with Gymnasium's `CartPole-v1`.

## Impact

- Core: `core/experiments/_identification.py`, `_calibration.py`, and `_policy_training.py` gain the selection, with
  a private module `core/experiments/_selection.py` that the three share, and a new module
  `core/engine/runtime/_environments.py`.
- Systems: the new concept subpackage `skcomplex.systems.general.environments` and a test of
  `engineering/control/_cart_pole.py`.
- Dependencies: `gymnasium>=1.4.0` (1.4.0, released 2026-10-05) and `pettingzoo>=1.27.0` (1.27.0, released
  2026-08-13). scikit-learn is already a runtime dependency.
- Docs: sections of the calibrate and experiments pages of the user guide and a gallery example that trains a policy
  of the library on a Gymnasium environment and runs a Gymnasium agent on a system.
- Depends on `add-learning-foundation`. Before `add-off-policy-learning` and `add-multi-agent-learning`.
