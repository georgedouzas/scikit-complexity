# data-driven-dynamics Specification

## Purpose
Lets a user fit runnable surrogate systems to trajectories, by mode decomposition, sparse identification of symbolic laws, or reservoir computing, and run them through the same interface as hand-written systems.

## Requirements

### Requirement: Linear surrogates by mode decomposition
`DMDSurrogate` SHALL learn dynamic mode decomposition and extended DMD from trajectories with `prepare(series)`, deriving modes, eigenvalues, and a system that `Simulation` runs, and SHALL forecast with `run(steps)`.

#### Scenario: Linear system recovered
- **WHEN** DMD is prepared on a trajectory of a known linear ODE
- **THEN** its continuous eigenvalues equal those of the system matrix within 1e-6

### Requirement: Sparse symbolic surrogates
`SINDySurrogate` SHALL learn sparse symbolic laws from trajectories with `prepare(series)` that the compiler accepts.

#### Scenario: Lorenz rediscovered
- **WHEN** SINDy is prepared on a noise-free `lorenz` trajectory with a polynomial library of degree 2
- **THEN** it recovers the seven nonzero terms of the Lorenz system with coefficients within 1%

### Requirement: Surrogates are systems
Every prepared surrogate SHALL set `system_`, a `System` of one element with ordinary laws and a declared time, continuous
for mode decomposition and SINDy and discrete for reservoirs, and that system SHALL validate and run through
`Simulation` like a hand-written system.

#### Scenario: Surrogate validated
- **WHEN** the `system_` of a SINDy surrogate prepared on `lorenz` is validated
- **THEN** the report has no error, its time is continuous, and the automatic solver is an ODE solver

#### Scenario: Reservoir under discrete time
- **WHEN** a reservoir surrogate is simulated
- **THEN** its system declares discrete time, and each step applies the readout once

### Requirement: Reservoir surrogates
`ReservoirSurrogate` SHALL learn echo state networks and next-generation reservoirs from trajectories with `prepare(series)` and forecast autonomously with `run(steps)`.

#### Scenario: Chaotic forecast
- **WHEN** a next-generation reservoir is prepared on 40 Lyapunov times of `lorenz`
- **THEN** its forecast from `run` stays within 10% of the attractor size for at least one Lyapunov time

### Requirement: Mode decomposition with control inputs
`ModeDecompositionAnalysis` SHALL take `inputs`, the names of the variables that are control inputs, and SHALL then fit
dynamic mode decomposition with control, `g(t + dt) = A g(t) + B u(t)`, on the variables or on the observables of its
dictionary. It SHALL set `operator_` and `input_matrix_`, and its `system_` SHALL take every input as a bounded adaptive
parameter that its rate laws read.

#### Scenario: Linear system with input
- **WHEN** the analysis with one input is prepared on runs of a known linear system driven by a random
  piecewise-constant input
- **THEN** `operator_` and `input_matrix_` equal the matrices of the system's one-sample map within 1e-6

#### Scenario: Input as a control
- **WHEN** the `system_` of the analysis is simulated with a schedule of its input parameter
- **THEN** the trajectory records the input at every frame, and the states follow the fitted map
