## ADDED Requirements

### Requirement: One prepared system simulates many times
A prepared system SHALL simulate any number of times with different end times, initial states, random states, and
observers, without compiling again.

#### Scenario: Two runs from one preparation
- **WHEN** a user prepares a system once and calls `simulate(until=1.4)` and
  `simulate(until=2.0, initial_state={'ball.x': 20.0})`
- **THEN** both return trajectories, and the problem is compiled once

### Requirement: The solver follows from the problem unless the system names one
`prepare` SHALL infer the problem from the kinds of law present and SHALL choose the default solver of that problem
when the system's `solver` is `'auto'`. A solver or backend that does not support the problem SHALL be rejected when
preparing, with an error that names the problem and the solvers that support it.

#### Scenario: Same reactions, two solvers
- **WHEN** a reaction network is prepared once with `solver='auto'` and once with `GillespieSolver()`
- **THEN** the first runs as ODEs, the second as a jump process, and the model is the same object

#### Scenario: An unsuitable solver
- **WHEN** a system of update laws is prepared with an ODE solver
- **THEN** preparing raises `ValidationError`, which names the map problem and the solvers that support it

## REMOVED Requirements

### Requirement: Simulating is a function
**Reason**: A system is an estimator that is prepared once and simulated many times, so simulating is its method.
**Migration**: Replace `simulate(system, until=10, solver=s)` by
`system.set_params(solver=s).prepare().simulate(until=10)`.
