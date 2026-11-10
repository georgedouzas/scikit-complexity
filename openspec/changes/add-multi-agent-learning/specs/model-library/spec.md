# Spec Delta

## ADDED Requirements

### Requirement: Learning prisoner's dilemma
The library SHALL provide `learning_prisoners_dilemma` in `social/cooperation`, two learners of the iterated prisoner's
dilemma with a memory of the last joint action and the payoffs of Foerster et al. (2018), with a gallery example and an
inventory entry.

#### Scenario: Payoffs of the paper
- **WHEN** the players always cooperate, then always defect
- **THEN** each player receives -1 per step, then -2 per step

### Requirement: Multi-armed bandit
The library SHALL provide `multi_armed_bandit` in `biology/neuroscience`, the 10-armed testbed of Sutton and Barto (2018,
sect. 2.3), with a learner by temporal differences and choice or by an online policy, a gallery example, and an
inventory entry.

#### Scenario: Greedy and epsilon-greedy learners
- **WHEN** sample average learners with epsilon 0, 0.01, and 0.1 run 2000 tasks of 1000 steps, as a slow test
- **THEN** the greedy learner earns about 1 per step and picks the best arm in about one third of the tasks at the end,
  within 0.1 and 0.05, the best possible reward is about 1.54 within 0.05, and the learner with epsilon 0.1 picks the
  best arm at most 91% of the time plus the sampling error

### Requirement: Cliff walking
The library SHALL provide `cliff_walking` in `biology/neuroscience`, the 4 by 12 gridworld of Sutton and Barto (2018,
Example 6.6), with a SARSA or Q-learning learner, a gallery example, and an inventory entry.

#### Scenario: Safe and optimal paths
- **WHEN** SARSA and Q-learning learners with epsilon 0.1 learn for 500 episodes with the same seed
- **THEN** the greedy path of Q-learning runs along the edge of the cliff in 13 moves, the greedy path of SARSA is
  longer and keeps away from the edge, and SARSA earns more per episode over the last 100 episodes
