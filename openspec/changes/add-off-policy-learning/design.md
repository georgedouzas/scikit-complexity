## Context

`add-learning-foundation` adds `EnvironmentSpec`, `PolicyInteraction`, and `train_policy` with `'reinforce'`, `'ppo'`,
and `'pathwise'`, on the private `_OptaxLoop` of `GradientDescentSolver`. Rollouts run through the runner of `ensemble`
with recorded draws, and `observe` reads them as observed runs. Its design names the off-policy estimators as the work
of this change. `PendulumSwingUpSystem` in `engineering/control/_control.py` swings a pendulum up with a bang-bang
pivot acceleration of size `mu`.

## Goals / Non-Goals

**Goals:**
- Off-policy training for choices and for continuous actions, and offline training from logged runs, as methods of
  `train_policy`.
- One store of transitions, read from recorded runs or from observed runs alike.

**Non-Goals:**
- Prioritised replay, distributional Q-learning, and model-based methods. They can be added as options later.
- Off-policy evaluation with importance weights.

## Decisions

### Placement

| Component | Placement step | Where | Domains that use it |
| --- | --- | --- | --- |
| The methods `'dqn'`, `'sac'`, and `'fqi'` | 2, estimators of an experiment over any system | `core/experiments/_policy_training.py` | engineering (control), epidemiology and medicine (treatment logs), ecology (harvest logs), economics (logged decisions) |
| The temporal-difference gradient | 2, an estimator of the shared loop | `core/engine/solvers/_gradient_descent.py` | as above |
| The environment of the pendulum | 3, one system | `engineering/control/_control.py` | engineering |

### Transitions

- A transition is the observation at a decision, the recorded action draw, the reward over the interval, the
  observation at the next decision, and whether it was terminal. A private `_TransitionStore` of the module holds them
  as arrays with a ring buffer of size `buffer`, filled from rollouts with recorded draws or from `data`.
- Logged runs in `data` must observe the paths of the environment, the actions, and the reward, and the store raises
  `ValidationError` naming any missing path.

### Methods

- `'dqn'`: a categorical policy whose logits are the Q-values of `q_function`, acting epsilon-greedily with a linearly
  decaying epsilon during training. The loss is the Huber loss of the double Q target, with the target network updated
  by Polyak averaging. Optax runs the updates on batches drawn from the store.
- `'sac'`: the normal policy with a tanh squashing to the bounds, two Q-functions, and the entropy temperature tuned
  towards minus the action dimension, as in arXiv 1812.05905.
- `'fqi'`: iterations of fitting the Q-function to `r + discount * max Q(next)` on the logged transitions. A
  scikit-learn regressor refits with `fit`, an Equinox module trains for a few epochs of the shared loop per iteration.
  The policy of the result is greedy in the Q-function.
- Every method returns the trained system with `q_function` in the result, and the policy of the system reads it.

### Pendulum environment

- The environment of `PendulumSwingUpSystem` observes the cosine and sine of the angle and the angular velocity, acts
  on the pivot acceleration within `(-mu, mu)`, rewards `-(theta - pi)^2 - 0.1 * theta'^2 - 0.001 * u^2` wrapped to the
  nearest upright angle, and decides every 0.05. The energy-pumping controller of the system is the reference.

## Risks / Trade-offs

- [Off-policy methods are sensitive to their settings] → The defaults follow the papers, tests check improvement on
  short runs, and the thresholds of the scenarios are slow tests.
- [Logged data that never visits a state cannot teach its value] → The user guide states the coverage needed by
  fitted Q iteration.

## Open Questions

- Should the store accept trajectories recorded by `GymnasiumSystem` of the bridges? It does by construction, since
  they record draws like any system.
