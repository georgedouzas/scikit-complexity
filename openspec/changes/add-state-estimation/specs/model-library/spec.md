# Spec Delta

## ADDED Requirements

### Requirement: Noisy pendulum
`PendulumSystem` SHALL take `noise`, the spectral density of a white noise on the angular acceleration, the noisy
pendulum of Särkkä (2013, Example 3.7), and SHALL stay deterministic with `noise=0`, the default.

#### Scenario: Variance of the velocity
- **WHEN** the pendulum with `g=0` and `noise=0.01` runs from rest for 1 time unit in 2000 seeded runs
- **THEN** the variance of the angular velocity at the end is within 5% of 0.01

#### Scenario: Default unchanged
- **WHEN** the pendulum runs with its defaults
- **THEN** its run equals the run of the system before this change within 1e-12
