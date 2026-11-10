## Context

`add-learning-foundation` adds `ObservedRunsSpec`, `identify` and its `IdentificationResult`, `EnvironmentSpec`,
`PolicyInteraction`, and `train_policy`. Every function that learns a system returns a result with the fields `system`,
`training_error`, and `validation_error`. `BaseConfig` derives from `sklearn.base.BaseEstimator`, so `get_params` and
`set_params` of systems already reach nested parameters with double underscores. The runtime runs a system step by
step through `simulations` and its clocks. `CartPoleSystem` uses the constants and Euler steps of Gymnasium's
`CartPole-v1`.

## Goals / Non-Goals

**Goals:**
- Identified and surrogate systems in scikit-learn model selection and pipelines, with no change to `identify` or the
  analyses.
- Any system with an environment as a Gymnasium or PettingZoo environment, and any such environment as a system.

**Non-Goals:**
- Wrapping agent libraries such as stable-baselines3 or skrl. Their agents read the Gymnasium environment directly.
- Vectorised Gymnasium environments and rendering. A later change can add them.
- JAX environments of gymnax or brax.

## Decisions

### Placement

| Part | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| `IdentificationEstimator`, `SurrogateEstimator` | 2, experiments over any system, in the model selection layer | `skcomplex.core.experiments` | every field with data |
| `make_environment`, `make_parallel_environment` | 2, the runtime steps a system through its `EnvironmentSpec` | `skcomplex.core.engine.runtime` | engineering, epidemiology, ecology, economics, social systems, computer science |
| `GymnasiumSystem`, `PettingZooSystem` | 3, recipes of an element and an update law with a `CallSpec`, used by several fields | `skcomplex.systems.general.environments` (needs approval) | as above |

### Dependencies

Checked on PyPI on 2026-11-10. Gymnasium 1.4.0, released 2026-10-05, needs numpy, cloudpickle, typing-extensions, and
farama-notifications. PettingZoo 1.27.0, released 2026-08-13, needs numpy, gymnasium, and typing-extensions, and
supports Python up to 3.14. Both are runtime dependencies, since the runtime of the core imports them. Their extras for
Atari, Box2D, and MuJoCo are not installed.

### scikit-learn estimators

- The constitution asks public objects to conform to the framework they plug into. These two classes plug into
  scikit-learn, so they speak `fit`, `predict`, and `score`, and they derive from `BaseEstimator` alone, not from
  `BaseConfig`. They are the only classes of the library with `fit`, which the user guide states.
- `IdentificationEstimator(system, trained, method='gradient', loss=None, optimizer='adam', learning_rate=1e-2,
  epochs=500, validation=0.0, random_state=None)`. `fit(X, y=None)` takes a list of `ObservedRunSpec`, which
  scikit-learn can index for splits, or an `ObservedRunsSpec`, calls `identify`, and sets `system_`, `estimates_`, and
  `result_`. `predict(X)` simulates `system_` with the conditions and inputs of every run and returns its observed
  paths at the times of the run. `score(X)` returns minus the mean loss of the runs.
- Parameters of the system are reached as `system__r`, so `GridSearchCV` searches them as starting values together
  with `learning_rate` or `epochs`.
- `SurrogateEstimator(analysis, settings=None)` calls the analysis on the runs with the settings and reads the
  `system` of its result. A search over settings passes a list of dictionaries for `settings`.
- `Pipeline` composes them with transformers of runs, such as a `FunctionTransformer` that smooths every run.

### Gymnasium

- `make_environment(system, *, environment=None, until=None, solver=None, backend=None, random_state=None)` takes the
  environment of the system, its class attribute or the argument. The observation space is a `Box` of the observation
  paths, bounded by the domains of their states. The action space is a `Box` of the continuous actions or a `Discrete`
  of the choices, and several choices give a `MultiDiscrete`.
- `reset(seed=...)` prepares a fresh run from the initial states. `step(action)` writes the actions as the policy
  would, advances one interval through the stepwise runtime, and returns the observation with its noise, the reward,
  `terminated` from the terminal observable, and `truncated` at `until`.
- A system that holds a `PolicyInteraction` for those actions is refused, since an external agent replaces it.

### PettingZoo

- `make_parallel_environment(system, *, environments=None, ...)` makes one agent per `PolicyInteraction` of the system,
  or one agent per member when a policy is bound to a group, named by the policy label and the member index. Each agent
  has the spaces of its environment and its own reward. The external agents replace the policies of the system.

### Environments as systems

- `GymnasiumSystem(environment, label='system')` takes an environment or its identifier. It holds one element with the
  observation as a vector state, the last reward, and the terminated flag, and one discrete-time update law with a
  `CallSpec` that steps the environment with the current action. The action is a bounded adaptive parameter, or a
  choice of the indices of a `Discrete` space, so a `PolicyInteraction` sets it. Its class attribute `environment`
  reads the observation, the reward, and the flag.
- `PettingZooSystem(environment, label='system')` holds one element per agent with the same states and one update law
  that steps every agent at once.
- They are recipes of core parts used by several fields. `systems.general.environments` is a new concept subpackage,
  so it needs the maintainer's approval. Until it is given, the tasks that write them wait.

## Risks / Trade-offs

- [Gymnasium seeds and the run's random stream differ] → `reset(seed=...)` seeds the run's stream, and the test checks
  that two resets with one seed give equal episodes.
- [External environments are not differentiable] → `GymnasiumSystem` and `PettingZooSystem` declare stochastic,
  non-differentiable tags, so `train_policy(method='pathwise')` refuses them before running.
- [The scikit-learn verbs differ from those of the library] → The two adapters are documented as the bridge, and no
  other class gains `fit`.

## Open Questions

- Should the subpackage be named `environments` or `external`? The design proposes `environments`, named by the
  concept of reinforcement learning it bridges.
