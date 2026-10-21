# Spec Delta

## Purpose

Lets a user fit runnable surrogate systems to trajectories, by mode decomposition, sparse identification of symbolic laws, or reservoir computing, and run them through the same interface as hand-written models.

## ADDED Requirements

### Requirement: Linear surrogates by mode decomposition
The analysis SHALL fit dynamic mode decomposition and extended DMD to trajectories, returning modes, eigenvalues, forecasts, and an element that `Simulation` runs.

#### Scenario: Linear system recovered
- **WHEN** DMD is fitted to a trajectory of a known linear ODE
- **THEN** its continuous eigenvalues equal those of the system matrix within 1e-6

### Requirement: Sparse symbolic surrogates
The analysis SHALL fit SINDy to trajectories and return symbolic laws the compiler accepts.

#### Scenario: Lorenz rediscovered
- **WHEN** SINDy is fitted to a noise-free `lorenz` trajectory with a polynomial library of degree 2
- **THEN** it recovers the seven nonzero terms of the Lorenz system with coefficients within 1%

### Requirement: Reservoir surrogates
The analysis SHALL fit echo state networks and next-generation reservoirs to trajectories and return surrogates that forecast autonomously.

#### Scenario: Chaotic forecast
- **WHEN** a next-generation reservoir is fitted to 40 Lyapunov times of `lorenz`
- **THEN** its autonomous forecast stays within 10% of the attractor size for at least one Lyapunov time
