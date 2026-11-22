# Spec Delta

## ADDED Requirements

### Requirement: Members learn by temporal differences
`skcomplex.systems.general.agents` SHALL provide `TemporalDifferenceInteraction`, which moves the value of the last
observation and action of every member towards the target of Q-learning, SARSA, expected SARSA, or TD(0) after every
step, as one `UpdateLaw` of the values of a group without a `schedule`, and as one rule of the whole system that walks
the agents with a `schedule`, as `ChoiceInteraction` does.

#### Scenario: Q-learning target
- **WHEN** a member with values `[[0, 0], [1, 3]]` acts 0 in state 0, receives 1, and moves to state 1, with rate 0.5
  and discount 0.9
- **THEN** its value of state 0 and action 0 becomes `0.5 * (1 + 0.9 * 3)`, 1.85, within 1e-12

#### Scenario: SARSA target
- **WHEN** a member with the same values acts 0 in state 0, receives 1, moves to state 1, and chooses action 0 there
  under `'sarsa'`
- **THEN** its value of state 0 and action 0 becomes `0.5 * (1 + 0.9 * 1)`, 0.95, within 1e-12

#### Scenario: Sample averages
- **WHEN** a stateless learner with `rate='average'` receives the rewards 1, 2, and 3 for one arm
- **THEN** the value of that arm is 2 within 1e-12

#### Scenario: Same values with and without a schedule
- **WHEN** a group of 10 Q-learners runs 50 steps with the same seed once without a schedule and once with the schedule
  `'synchronous'`
- **THEN** both runs hold the same tables of values within 1e-12

### Requirement: Members choose epsilon-greedily
`ChoiceInteraction` SHALL take the rule `'epsilon-greedy'`, which picks an option of largest utility with probability
`1 - epsilon` and a uniform option otherwise.

#### Scenario: Share of the best option
- **WHEN** 10000 members choose among 10 options with distinct utilities and epsilon 0.1
- **THEN** the share that picks the best option is within 0.01 of 0.91
