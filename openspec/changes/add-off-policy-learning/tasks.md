# Tasks

## 1. Transitions

- [ ] 1.1 Add the private store of transitions in `src/skcomplex/core/experiments/_policy_training.py`, filled from
  rollouts with recorded draws and from observed runs, and verify the missing-reward scenario and a round trip of
  recorded transitions
- [ ] 1.2 Add the temporal-difference gradient to the shared loop of `GradientDescentSolver`, and verify it on a
  two-state chain with known action values

## 2. Methods

- [ ] 2.1 Add `'dqn'` with epsilon-greedy acting, the target network, and double Q targets, and verify the
  continuous-actions-refused scenario and the cart-pole scenario, marked slow
- [ ] 2.2 Add `'sac'` with the squashed normal policy, two Q-functions, and the tuned temperature
- [ ] 2.3 Add `data` and `'fqi'` with scikit-learn regressors and Equinox modules, and verify the logs-of-a-random-policy
  scenario
- [ ] 2.4 Add `q_function` to `PolicyTrainingResult`, and verify it is `None` for on-policy methods

## 3. Pendulum

- [ ] 3.1 Add the environment of `PendulumSwingUpSystem`, and verify the reference-controller scenario in
  `tests/systems/engineering/control/`
- [ ] 3.2 Verify the pendulum-swung-up scenario with `'sac'`, marked slow, and add the gallery example
  `docs/examples/engineering/control/plot_pendulum_soft_actor_critic.py` that compares the learned policy with the
  energy-pumping controller, and verify the gallery build runs it

## 4. Documentation and checks

- [ ] 4.1 Document off-policy and offline training in the experiments page of the user guide with a runnable example of
  fitted Q iteration
- [ ] 4.2 Run `nox` (formatting, checks, tests of the touched modules with the thread limits set to 1, and the strict
  docs build) and verify every session passes
