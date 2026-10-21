# Spec Delta

## ADDED Requirements

### Requirement: Repeated Lyapunov spectra reuse their compiled tangent dynamics
`lyapunov_spectrum` SHALL compile the tangent dynamics of a model structure once, with the parameters as an argument,
so repeated calls on models of the same structure do not compile again.

#### Scenario: A sweep of a one-dimensional map
- **WHEN** a user computes the spectrum of the logistic map for 10 values of `r` with 2000 iterations each, after one
  call
- **THEN** the 10 calls take less than 1 second together, and each exponent equals the exponent of a fresh call

### Requirement: Orbit diagrams and basins run as vectorised ensembles
`orbit_diagram` and `basin_stability` SHALL accept `vectorized=True` to run their sweep as one simulation of a
replicated system, with the same results as the sweep of separate runs.

#### Scenario: Vectorised period doubling
- **WHEN** a user builds the orbit diagram of the logistic map for 200 values of r in [2.8, 4] with `vectorized=True`
- **THEN** its points equal those of the serial sweep within 1e-12

#### Scenario: Vectorised basins
- **WHEN** a user estimates the basin stability of a bistable system over 200 initial conditions with
  `vectorized=True`
- **THEN** the fractions equal those of the serial runs with the same seed

### Requirement: Lyapunov spectra of laws with absolute values
`lyapunov_spectrum` SHALL differentiate `abs(x)` as `sign(x)`, so laws with absolute values have a spectrum.

#### Scenario: Sprott jerk system
- **WHEN** a user computes the spectrum of `sprott_jerk` until 200
- **THEN** it returns three finite exponents whose sum is negative
