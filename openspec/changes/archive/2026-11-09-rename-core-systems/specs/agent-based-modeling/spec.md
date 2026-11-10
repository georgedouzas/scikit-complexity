## MODIFIED Requirements

### Requirement: Agent systems run on Mesa
An agent system SHALL be an ordinary system: elements with rule laws, in fixed or variable groups, under discrete time,
on a space. Mesa SHALL be the backend that executes discrete-time systems with rule laws, selected and validated like
any other backend. It SHALL advance one Mesa step per unit of time, activate agents by the schedule of the system's
time with a seed that makes runs reproducible, and advance Mesa's clock so that rules read the time `k` during the
step that produces frame `k`.

#### Scenario: Reproducible run
- **WHEN** the same agent system is simulated twice on the Mesa backend with random state 0
- **THEN** the two trajectories are equal

#### Scenario: Rules read the system time
- **WHEN** a rule stores `system.time` in a state of its agent at every step of a run until time 3
- **THEN** the state is 1, 2, and 3 in the frames at times 1, 2, and 3, not 0

#### Scenario: No agent-specific system class
- **WHEN** a user builds an agent system
- **THEN** they use `CompositeSystem` with `DiscreteTime`, and no `MesaSystem` or `AgentType` exists in the core
