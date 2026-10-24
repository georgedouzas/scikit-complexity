# Spec Delta

## MODIFIED Requirements

### Requirement: Stochastic runs are reproducible
A simulation SHALL accept a random state, and two runs with the same random state SHALL produce the same trajectory.
Runs of the same system with the same random state, solver, and step SHALL produce the same trajectory on the NumPy and
the Numba backends, within floating-point rounding, from any start time to any end time.

#### Scenario: Same seed
- **WHEN** a stochastic epidemic is simulated twice with random state 0
- **THEN** the two trajectories are equal

#### Scenario: Same seed on both backends
- **WHEN** geometric Brownian motion runs with Euler-Maruyama, a step of 0.1, and random state 0 from time 0.1 to time
  0.4 on the NumPy backend and on the Numba backend
- **THEN** both runs take the same number of steps and their final states agree to within 1e-12 relative error
