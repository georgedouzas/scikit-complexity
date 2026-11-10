## Why

`add-learning-foundation` makes identification, learned laws, and policy training work inside the library. The methods
and agents most users already have live in three ecosystems outside it. scikit-learn holds pipelines, cross-validation,
and hyperparameter searches. Gymnasium is the interface of single-agent reinforcement learning, which agents of
stable-baselines3, skrl, and many papers read. PettingZoo is the same interface for many agents. A system of the library
should be usable in all three without rewriting it, and an environment of those ecosystems should be usable as a system
of the library, so its policies train against it with `train_policy`.

## What Changes

- New scikit-learn estimators in `skcomplex.core.experiments`:
  - `IdentificationEstimator`, which wraps `identify`, takes lists of observed runs, and sets `system_` after `fit`, so
    `cross_validate`, `GridSearchCV`, and `Pipeline` select and compose identified systems
  - `SurrogateEstimator`, which wraps any analysis whose result holds a `system`, such as `sparse_identification`,
    `mode_decomposition`, and `reservoir`, with the same contract
- New function `make_environment(system, ...)` in `skcomplex.core.engine.runtime`, which returns a Gymnasium
  environment that steps the system through its `EnvironmentSpec`, so external agents run against any system.
- New function `make_parallel_environment(system, ...)` beside it, which returns a PettingZoo parallel environment
  with one agent per policy or per member of a group.
- New systems `GymnasiumSystem` and `PettingZooSystem`, which wrap an external environment as a system whose actions
  are adaptive parameters and whose observations are states, so `PolicyInteraction` and `train_policy` act on it. They
  are recipes of core parts, proposed for a new concept subpackage `skcomplex.systems.general.environments` that needs
  the maintainer's approval.
- Dependencies: Gymnasium 1.4.0 and PettingZoo 1.27.0 become runtime dependencies.

## Capabilities

### New Capabilities

- `learning-bridges`: scikit-learn estimators of identification and surrogates, systems as Gymnasium and PettingZoo
  environments, and those environments as systems.

### Modified Capabilities

- `model-library`: the cart-pole system agrees step by step with Gymnasium's `CartPole-v1`.

## Impact

- Core: new modules `core/experiments/_estimators.py` and `core/engine/runtime/_environments.py`.
- Systems: a new concept subpackage `skcomplex.systems.general.environments`, if approved, and a test of
  `engineering/control/_cart_pole.py`.
- Dependencies: `gymnasium>=1.4.0` (1.4.0, released 2026-10-05) and `pettingzoo>=1.27.0` (1.27.0, released
  2026-08-13).
- Docs: sections of the experiments page of the user guide and a gallery example that trains a policy of the library
  on a Gymnasium environment and runs a Gymnasium agent on a system.
- Depends on `add-learning-foundation`. Before `add-off-policy-learning` and `add-multi-agent-learning`.
