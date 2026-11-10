## Why

`add-learning-foundation` trains policies on-policy: every update needs fresh rollouts of the current policy. Many
systems are costly to run, and many problems come with data logged under another policy, such as recorded
interventions, harvests, or treatments. Off-policy methods learn from stored transitions, reuse every run many times,
and learn from logs alone. The foundation already designs them as one more estimator of its shared loop, reading
transitions from recorded runs, so this change adds the estimators and no concept.

## What Changes

- `train_policy` gains the methods `'dqn'`, Q-learning with a replay store, a target network, and double Q-learning
  (Mnih et al. 2015, van Hasselt et al. 2016), for choices, and `'sac'`, soft actor-critic with two Q-functions and a
  tuned temperature (Haarnoja et al. 2018, arXiv 1812.05905), for continuous actions.
- `train_policy` gains `data`, an `ObservedRunsSpec` of logged runs with their actions and rewards, and `'fqi'`,
  fitted Q iteration from that data alone (Ernst, Geurts, and Wehenkel 2005), with a scikit-learn regressor or an
  Equinox module as the Q-function.
- `train_policy` gains `q_function`, the learned function of the action values, and `buffer`, the size of the store.
- `PolicyTrainingResult` gains `q_function`, the trained action values, `None` for on-policy methods.
- Probes on existing systems: `CartPoleSystem` with `'dqn'` and `PendulumSwingUpSystem` with `'sac'`, compared with the
  energy-pumping controller of Åström and Furuta (2000) that the system already runs.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `reinforcement-learning`: off-policy and offline training of policies.
- `model-library`: the pendulum swing-up system as an environment with a continuous pivot acceleration.

## Impact

- Core: `core/experiments/_policy_training.py`, the store of transitions as a private class of that module, and the
  temporal-difference gradient on the shared loop of `core/engine/solvers/_gradient_descent.py`.
- Systems: `engineering/control/_control.py` gains an environment of `PendulumSwingUpSystem`.
- No new dependency.
- Docs: the experiments page of the user guide and a gallery example of the pendulum swung up by soft actor-critic.
- Depends on `add-learning-foundation`. After `add-learning-bridges`, whose Gymnasium environments it does not need but
  whose agreement test of the cart-pole it reuses as a reference.
