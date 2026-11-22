# Spec Delta

## ADDED Requirements

### Requirement: Q-learning trains policies of choices
`train_policy` with `'dqn'` SHALL train a Q-function of the observation from a store of transitions with a target
network and double Q targets, act epsilon-greedily during training and greedily after, and return the Q-function in the
result.

#### Scenario: Cart-pole by Q-learning
- **WHEN** `train_policy` with `'dqn'` trains `CartPoleSystem` with `random_state=0` for 50000 decisions
- **THEN** the mean return over 100 evaluation episodes is at least 475

#### Scenario: Continuous actions refused
- **WHEN** `'dqn'` is called on a system whose environment has a continuous action
- **THEN** it raises `ValidationError` that says Q-learning needs choices and names `'sac'`

### Requirement: Soft actor-critic trains continuous policies
`train_policy` with `'sac'` SHALL train a squashed normal policy and two Q-functions with a tuned entropy temperature
from a store of transitions.

#### Scenario: Pendulum swung up
- **WHEN** `train_policy` with `'sac'` trains `PendulumSwingUpSystem` with `random_state=0`
- **THEN** the trained policy holds the pendulum within 0.2 radians of upright over the last half of a 20-second
  evaluation run, as the energy-pumping controller of the system does

### Requirement: Policies are trained from logged runs alone
`train_policy` with `'fqi'` and `data` SHALL train a Q-function by fitted Q iteration on the transitions of logged runs
without running the system, with a scikit-learn regressor or an Equinox module, and SHALL refuse logged runs that miss
a path of the environment, the actions, or the reward.

#### Scenario: Logs of a random policy
- **WHEN** fitted Q iteration with extremely randomised trees learns from 200 logged cart-pole episodes of a random
  policy
- **THEN** the greedy policy of the result has a mean return above 150 over 100 evaluation episodes

#### Scenario: Missing reward
- **WHEN** the logged runs do not observe the reward
- **THEN** it raises `ValidationError` that names the reward observable
