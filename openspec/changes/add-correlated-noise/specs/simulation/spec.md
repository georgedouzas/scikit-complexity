# Spec Delta

## ADDED Requirements

### Requirement: SDE steppers draw shared and correlated increments
The SDE steppers SHALL draw the increments of shared and correlated Wiener processes from the run's random stream, in
Python and in compiled code. Two runs with the same random state SHALL be equal, and a system that names no process
SHALL reproduce the trajectory it had before shared processes existed. A backend that cannot apply a shared process
SHALL reject it.

#### Scenario: Same seed with correlated noise
- **WHEN** a system with a correlated Wiener element runs twice with random state 0 on the Numba backend
- **THEN** the two trajectories are equal

#### Scenario: Unchanged seeded runs
- **WHEN** geometric Brownian motion runs with random state 0 and `EulerMaruyama`
- **THEN** its trajectory equals the one stored before this change

#### Scenario: Shared process on both backends
- **WHEN** one process drives two states with unit noise on the NumPy and on the Numba backend
- **THEN** on each backend the two states keep their initial difference at every frame

#### Scenario: Backend without shared noise
- **WHEN** a system with a shared process is given to the LAMMPS backend
- **THEN** it raises a named solver error that names the process
