## ADDED Requirements

### Requirement: Experiments prepare and simulate systems
Every experiment function SHALL take a system that it clones, sets parameters on, prepares, and simulates, in the way
`cross_val_score` fits and predicts. An experiment that varies only initial states or random states SHALL prepare once.

#### Scenario: An ensemble prepares once
- **WHEN** a user calls `ensemble(system, n_runs=100, random_state=0)`
- **THEN** the system is prepared once and simulated 100 times with independent streams

#### Scenario: A sweep prepares per point
- **WHEN** a user calls `sweep(system, {'gravity__g': [1.62, 3.71, 9.81]}, until=1.4)`
- **THEN** each point clones the system, sets the nested parameter, prepares, and simulates, and the user's system is
  unchanged
