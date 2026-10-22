# Spec Delta

## ADDED Requirements

### Requirement: Canonical and variational interactions generate laws
An interaction SHALL declare a function over canonical pairs of coordinates and momenta, from which the framework
derives Hamilton's equations as rate laws, or a function over coordinates of order 2 and their velocities, from which
it derives the Euler-Lagrange equations as equation laws, symbolically. Both interactions SHALL require continuous time
and real or angle coordinates, and validation SHALL report a coordinate without a momentum or a coordinate of order 1
in a variational interaction.

#### Scenario: Pendulum from a variational function
- **WHEN** a pendulum declares `L = m l² ω² / 2 + m g l cos θ` with an angle coordinate `θ`
- **THEN** the derived equation is `dω/dt = -(g / l) sin θ`

#### Scenario: Double pendulum chaos
- **WHEN** a double pendulum declared by its variational function runs from two initial angles that differ by 1e-9
- **THEN** the trajectories separate exponentially, and the energy stays constant within 1e-6 relative drift with the
  Yoshida solver

#### Scenario: Coordinate without a momentum
- **WHEN** a canonical interaction lists two coordinates and one momentum
- **THEN** `validate()` reports one error whose path is the interaction and whose message names the unpaired coordinate

#### Scenario: Coordinate of order 1
- **WHEN** a variational interaction names a coordinate declared with order 1
- **THEN** `validate()` reports the state, and its fix is `state(order=2)`

#### Scenario: Non-separable function
- **WHEN** a system whose canonical function couples coordinates and momenta in one term is validated with the solver
  `Yoshida4`
- **THEN** the report says that the solver needs a separable function and lists solvers that do not
