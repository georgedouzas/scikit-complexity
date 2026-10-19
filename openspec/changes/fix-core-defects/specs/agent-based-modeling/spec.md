## MODIFIED Requirements

### Requirement: Agent systems run on Mesa
The Mesa backend SHALL build a Mesa model from an agent-based system, advance it one Mesa step per unit of time, and
activate agents in random order by default, with a seed that makes runs reproducible. Each step SHALL advance Mesa's
clock, so the model time that rules read during the step that produces frame `k` is `k`.

#### Scenario: Reproducible run
- **WHEN** the same agent-based system is simulated twice with random state 0
- **THEN** the two trajectories are equal

#### Scenario: Rules read the model time
- **WHEN** a rule stores `model.time` in a state of its agent at every step of a run until time 3
- **THEN** the state is 1, 2, and 3 in the frames at times 1, 2, and 3, not 0

## ADDED Requirements

### Requirement: Agent states do not shadow Mesa attributes
An agent type SHALL raise a named label error when it is defined with a state named after an
attribute that Mesa or the Mesa backend reads from agents, which are `position`, `pos`, `cell`, `unique_id`, and
`model`. The error SHALL name the attribute and the agent type.

#### Scenario: State named position
- **WHEN** a user defines an agent type with a scalar state named `position`
- **THEN** a named label error is raised when the type is defined, which names `position`, and no run fails later
  with a `TypeError`
