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

### Requirement: Simulations take initial states
A simulation SHALL take initial values that replace those of the components, by dotted state name. The whole state of
a quantum system SHALL be named `<system label>.state` and take a vector, a density matrix, or `'ground'`. A catalog
entry SHALL be able to declare the initial values its runs use, as a function of the factory's arguments. Limits of a
solver, such as the largest dimension of a Hilbert space, SHALL be arguments of the solver, not of the system.

#### Scenario: Ground state of a chain
- **WHEN** a spin chain is simulated with `initial_state={'chain.state': 'ground'}`
- **THEN** it starts from the lowest eigenstate of its Hamiltonian, and every expectation stays constant

#### Scenario: Unknown state name
- **WHEN** a simulation is given an initial value of a state the system does not have
- **THEN** it raises a validation error that names the state

#### Scenario: Dimension limit of the solver
- **WHEN** a quantum system above the `max_dimension` of its solver is compiled for that solver
- **THEN** it raises an error that names the subsystems, the dimension, and the limit

### Requirement: Solvers and backends are chosen separately from the system
A simulation SHALL take an optional solver and backend. With `'auto'`, it SHALL choose among those whose declared support
covers the validated system. The system SHALL carry no execution choice.

#### Scenario: Automatic backend for agents
- **WHEN** an agent system is simulated with the default solver and backend
- **THEN** the Mesa backend is chosen because its declared support covers rule laws with variable groups
