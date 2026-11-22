# Spec Delta

## Purpose

Lets a user read any system as an environment, put a learned policy inside the system as one of its components, and
train it by an experiment, so control, intervention, harvesting, and learning agents of every field share one form.

## ADDED Requirements

### Requirement: Systems are read as environments
`EnvironmentSpec(observation, actions, reward, terminal, interval, discount)` SHALL name the observation model of a
system, its actions as parameters with bounds or with a tuple of choices, its reward and terminal observables, the time
between decisions, and the discount. Bounds SHALL default to the allowed values of the parameter. Preparing a system
whose policy holds it SHALL refuse an unknown action, bounds outside the allowed values, and an interval that is not
positive.

#### Scenario: Unknown action
- **WHEN** an environment names as action the path `push.forse` and the system has no such parameter
- **THEN** preparing the system raises `ValidationError` that names the path and lists the parameters of the system

#### Scenario: Bounds outside the allowed values
- **WHEN** an environment gives the action `push.force` the bounds `(-20, 20)` and the value allows `[-10, 10]`
- **THEN** preparing the system raises `ValidationError` that names the action and both ranges

### Requirement: Policies are interactions of the system
`PolicyInteraction` SHALL bind through its role `target` the part it observes and controls, and SHALL state one update
law per action whose value is a learned function of the observation: a named categorical draw for choices, a named
normal draw clipped to the bounds for continuous actions, or the mean for a deterministic policy. Its weights SHALL be
an array parameter. A run SHALL apply these laws every `interval` and hold the actions in between. Every random action
SHALL be a named draw that `record_draws` keeps and `replay` forces.

#### Scenario: Actions held between decisions
- **WHEN** a cart-pole system with a policy deciding every 0.02 is prepared with a solver step of 0.005 and simulated
- **THEN** the recorded force is constant within every interval between decisions and takes only the values -10 and 10

#### Scenario: Replay of actions
- **WHEN** a run with recorded draws is replayed with another seed
- **THEN** the actions and the states equal those of the recorded run

#### Scenario: Shared weights over a group
- **WHEN** a policy is bound to a group of 4 members with different observations
- **THEN** each member gets its own action from the same weights, and the actions differ

#### Scenario: Deterministic policy is differentiable
- **WHEN** a pendulum with a deterministic policy is prepared with the JAX backend and simulated
- **THEN** the gradient of the sum of its rewards in the policy weights agrees with central finite differences within
  1e-6 relative error

### Requirement: Training a policy is an experiment
`train_policy(system, method=...)` SHALL clone the system, prepare the clone once, and train the weights of its every
`PolicyInteraction`, with `'reinforce'` and `'ppo'` by the score of the recorded actions of rollouts cut at the
terminal observable, and with `'pathwise'` through the differentiable run. It SHALL return a frozen
`PolicyTrainingResult` whose `system` holds the trained weights and whose `returns` hold the mean return of every
epoch.

#### Scenario: Cart-pole balanced
- **WHEN** `train_policy` with `'ppo'`, a two-layer policy, and a two-layer critic trains `CartPoleSystem` for 200 epochs
  of 16 episodes with `random_state=0`
- **THEN** the mean return over 100 evaluation episodes of the trained system is at least 475, the threshold of
  Gymnasium's `CartPole-v1`

#### Scenario: Learning improves the return
- **WHEN** `train_policy` with `'reinforce'` trains `CartPoleSystem` for 20 epochs
- **THEN** the mean return of the last 5 epochs exceeds that of the first 5

#### Scenario: Same seed
- **WHEN** `train_policy` runs twice with the same seed and the serial executor
- **THEN** both results hold equal returns and equal weights

#### Scenario: No policy
- **WHEN** `train_policy` is called on a system without a `PolicyInteraction`
- **THEN** it raises `ValidationError` that says the system has no policy and names `PolicyInteraction`

#### Scenario: Pathwise needs a differentiable system
- **WHEN** `train_policy` with `'pathwise'` is called on a system with jump laws
- **THEN** it raises `ValidationError` that names the tag `differentiable` and the methods that do not need it

### Requirement: The training of a policy is a trajectory
`PolicyTrainingResult` SHALL hold, for every method, a `trajectory` whose times are the epochs and whose series are the
trained weights, the mean return, and the loss.

#### Scenario: Training as a trajectory
- **WHEN** `train_policy` with `'reinforce'`, `'ppo'`, or `'pathwise'` trains a policy for 20 epochs
- **THEN** the `trajectory` of the result has 21 frames, its weights at the last frame equal the weights of the
  trained system, and its mean returns equal `returns`
