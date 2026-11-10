# Spec Delta

## Purpose

Lets a user read any system as an environment, put a learned policy inside the system as one of its parts, and train
it by an experiment, so control, intervention, harvesting, and learning agents of every field share one form.

## ADDED Requirements

### Requirement: Systems are read as environments
`EnvironmentSpec(observation, actions, reward, terminal, interval, discount)` SHALL name the observation model of a
system, its actions as bounded adaptive parameters with their bounds or as parameters with a tuple of choices, its
reward and terminal observables, the time between decisions, and the discount. It SHALL refuse an action that is not an
adaptive parameter of the system, bounds outside the bounds of the parameter, and an interval that is not positive.

#### Scenario: Action not adaptive
- **WHEN** an environment names as action a parameter not declared adaptive
- **THEN** `prepare` raises `ValidationError` that names the parameter and says actions are bounded adaptive parameters

### Requirement: Policies are parts of the system
`PolicyInteraction` SHALL bind through its role `target` the part it observes and controls, read the observation of its
environment every `interval`, evaluate a learned function of it, and set the actions until the next decision, by a
categorical draw for choices, a normal draw clipped to the bounds for continuous actions, or the mean for a
deterministic policy. Every stochastic action SHALL be a named draw that `record_draws` keeps and replay forces.

#### Scenario: Actions held between decisions
- **WHEN** a cart-pole system with a policy deciding every 0.02 is simulated with a solver step of 0.005
- **THEN** the recorded force is constant within every interval between decisions and takes only the values -10 and 10

#### Scenario: Replay of actions
- **WHEN** a run with recorded draws is replayed with another seed
- **THEN** the actions and the states equal those of the recorded run

#### Scenario: Shared weights over a group
- **WHEN** a policy is bound to a group of 4 members with different observations
- **THEN** each member gets its own action from the same weights, and the actions differ

#### Scenario: Deterministic policy is differentiable
- **WHEN** a pendulum with a deterministic policy runs on the JAX backend
- **THEN** the gradient of the sum of its rewards in the policy weights agrees with central finite differences within
  1e-6 relative error

### Requirement: Training a policy is an experiment
`train_policy(system, method=...)` SHALL train the weights of every `PolicyInteraction` of a system, with `'reinforce'`
and `'ppo'` by the score of the recorded actions of rollouts cut at the terminal observable, and with `'pathwise'` through
the differentiable run of the system, and SHALL return a frozen `PolicyTrainingResult` whose `system` holds the trained
weights and whose `returns` hold the mean return of every epoch.

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
