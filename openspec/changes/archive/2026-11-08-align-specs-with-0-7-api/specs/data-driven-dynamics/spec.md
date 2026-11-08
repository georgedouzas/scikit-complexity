## MODIFIED Requirements

### Requirement: Linear surrogates by mode decomposition
`mode_decomposition(source)` SHALL learn dynamic mode decomposition and extended DMD from trajectories and SHALL return a frozen result that holds the modes, the eigenvalues, a `system` that `simulate` runs, and a `forecast` of `horizon` samples.

#### Scenario: Linear system recovered
- **WHEN** `mode_decomposition` runs on a trajectory of a known linear ODE
- **THEN** the continuous eigenvalues of its result equal those of the system matrix within 1e-6

### Requirement: Sparse symbolic surrogates
`sparse_identification(source)` SHALL learn sparse symbolic laws from trajectories that the compiler accepts, and SHALL return them in a frozen result.

#### Scenario: Lorenz rediscovered
- **WHEN** `sparse_identification` runs on a noise-free Lorenz trajectory with `degree=2`
- **THEN** it recovers the seven nonzero terms of the Lorenz system with coefficients within 1%

### Requirement: Surrogates are systems
The result of every surrogate function SHALL hold `system`, a `System` of one element with ordinary laws and a declared
time, continuous for mode decomposition and sparse identification and discrete for reservoirs, and that system SHALL
validate and run through `simulate` like a hand-written system.

#### Scenario: Surrogate validated
- **WHEN** the `system` of the result of `sparse_identification` on a Lorenz trajectory is validated
- **THEN** the report has no error, its time is continuous, and the automatic solver is an ODE solver

#### Scenario: Reservoir under discrete time
- **WHEN** the `system` of the result of `reservoir` is simulated
- **THEN** it declares discrete time, and each step applies the readout once

### Requirement: Reservoir surrogates
`reservoir(source)` SHALL learn echo state networks and next-generation reservoirs from trajectories and SHALL return a frozen result whose `system` runs the autonomous forecast and whose `forecast` holds `horizon` samples.

#### Scenario: Chaotic forecast
- **WHEN** a next-generation reservoir is fitted on 40 Lyapunov times of a Lorenz trajectory
- **THEN** its forecast stays within 10% of the attractor size for at least one Lyapunov time
