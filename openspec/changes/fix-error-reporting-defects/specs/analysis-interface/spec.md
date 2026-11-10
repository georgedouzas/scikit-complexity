## ADDED Requirements

### Requirement: An analysis of a trajectory reports times in its units

An analysis that reports times or lags SHALL read the step between samples from the times of a trajectory it is
given. A step given as an argument SHALL apply only to arrays, which carry no times.

#### Scenario: Correlation of a trajectory with a step of a tenth

- **WHEN** `correlation` is run on a trajectory sampled every 0.1 time units
- **THEN** its lags are 0, 0.1, 0.2, and so on, and its correlation time is in time units

#### Scenario: Correlation of an array

- **WHEN** `correlation` is run on an array with `dt=0.5`
- **THEN** its lags are 0, 0.5, 1.0, and so on
