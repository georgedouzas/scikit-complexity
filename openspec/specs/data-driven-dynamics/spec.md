# data-driven-dynamics Specification

## Purpose
Lets a user fit runnable surrogate systems to trajectories, by mode decomposition, sparse identification of symbolic laws, or reservoir computing, and run them through the same interface as hand-written systems.

## Requirements

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
The result of every surrogate function SHALL hold `system`, a `CompositeSystem` of one element with ordinary laws and a declared
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

### Requirement: Mode decomposition with control inputs
`mode_decomposition` SHALL take `inputs`, the names of the variables that are control inputs, and SHALL then fit
dynamic mode decomposition with control, `g(t + dt) = A g(t) + B u(t)`, on the variables or on the observables of its
dictionary. Its result SHALL set `operator` and `input_matrix`, and its `system` SHALL take every input as a bounded adaptive
parameter that its rate laws read.

#### Scenario: Linear system with input
- **WHEN** the analysis with one input runs on runs of a known linear system driven by a random
  piecewise-constant input
- **THEN** `operator` and `input_matrix` equal the matrices of the system's one-sample map within 1e-6

#### Scenario: Input as a control
- **WHEN** the `system` of the analysis result is simulated with a schedule of its input parameter
- **THEN** the trajectory records the input at every frame, and the states follow the fitted map
