# Spec Delta

## ADDED Requirements

### Requirement: Fixed points and their stability are derived
The analysis layer SHALL find the fixed points of ODE systems and maps, derive their Jacobians, and classify each fixed
point's stability from its eigenvalues.

#### Scenario: Lotka-Volterra centre
- **WHEN** a user analyses prey with exponential growth rate `r`, predators with mortality `d`, and predation with
  rates `a` for the prey and `b` for the predators
- **THEN** the interior fixed point has prey `d / b` and predators `r / a`, and its eigenvalues are purely imaginary,
  classified as a centre

#### Scenario: Logistic map period one
- **WHEN** a user analyses the logistic map at r = 2.5
- **THEN** the fixed point `1 - 1 / r` is classified as stable

### Requirement: Vector-field divergence is derived
The analysis layer SHALL derive the divergence of a system's vector field.

#### Scenario: Volume preservation
- **WHEN** a user derives the divergence of a frictionless harmonic oscillator
- **THEN** the divergence is 0
