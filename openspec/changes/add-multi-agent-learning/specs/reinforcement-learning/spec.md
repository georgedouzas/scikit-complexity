# Spec Delta

## ADDED Requirements

### Requirement: Several policies learn independently
`train_policy` SHALL train every policy of a system on the returns of its own reward from shared rollouts, keep one
optimizer state per policy, and report the returns of every policy by its label.

#### Scenario: Naive learners defect
- **WHEN** `train_policy` with `'reinforce'` trains both players of `LearningPrisonersDilemmaSystem` for 500 epochs
  with `random_state=0`
- **THEN** the mean reward per step of each player over the last 50 epochs is below -1.5, towards the mutual defection
  that Foerster et al. (2018) report for naive learners

#### Scenario: Returns by policy
- **WHEN** a system holds two policies with different rewards
- **THEN** the result holds one series of returns per policy label

### Requirement: Critics can read the joint observation
`train_policy` with `'ppo'` and `centralised=True` SHALL give every policy a critic of the concatenated observations of
all policies, while each policy acts on its own observation.

#### Scenario: Critic inputs
- **WHEN** two policies observe 3 and 4 values and train with `centralised=True`
- **THEN** each critic is called with 7 inputs and each policy with its own

### Requirement: Policies play frozen copies
`train_policy` SHALL take `opponents`, a mapping from a policy label to the label of the policy whose past weights it
plays, refreshed from a league of past weights.

#### Scenario: Frozen opponent
- **WHEN** a player trains against a frozen copy of the other
- **THEN** the weights of the frozen copy stay equal between refreshes

### Requirement: Policies learn during a run
`PolicyInteraction` with `online=True` SHALL apply one gradient step to its weights after the reward of every decision
inside `simulate`, by one-step actor-critic with a critic or by the reward against a running baseline without one, and
the trajectory SHALL record the weights at every frame. `train_policy` SHALL refuse a system with an online policy.

#### Scenario: Gradient bandit with a baseline
- **WHEN** online policies with and without a baseline run 1000 steps of `MultiArmedBanditSystem(mean=4.0)` over 200
  seeded tasks
- **THEN** the share of optimal actions in the last 100 steps is higher with the baseline, as in Figure 2.5 of Sutton
  and Barto

#### Scenario: Weights as a trajectory
- **WHEN** two online policies play a random game of 10 moves for 5000 decisions
- **THEN** the trajectory holds their weights at every frame, and `lyapunov` reads them as states

#### Scenario: Online policy in training
- **WHEN** `train_policy` is called on a system whose policy has `online=True`
- **THEN** it raises `ValidationError` that names the policy and `online`
