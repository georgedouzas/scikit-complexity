# Spec Delta

## ADDED Requirements

### Requirement: Simulations take observables
A simulation SHALL take the observables to record as an argument. A catalog entry SHALL declare the standard observables
of its model, which `create` and `run` pass to the simulation. Trajectories SHALL record observables as before.

#### Scenario: Observables given to a simulation
- **WHEN** a user simulates a system with `Simulation(system, observables=[total_population])`
- **THEN** the trajectory records `total_population` at every frame

#### Scenario: Catalog observables
- **WHEN** a user runs a catalog model with `run`
- **THEN** the trajectory records the observables declared by its catalog entry

### Requirement: Solvers and backends are chosen separately from the system
A simulation SHALL take an optional solver and backend. With `'auto'`, it SHALL choose among those whose declared support
covers the validated system. The system SHALL carry no execution choice.

#### Scenario: Automatic backend for agents
- **WHEN** an agent system is simulated with the default solver and backend
- **THEN** the Mesa backend is chosen because its declared support covers rule laws with variable groups
