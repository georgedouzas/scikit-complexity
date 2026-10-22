# Spec Delta

## ADDED Requirements

### Requirement: Sweeps step a parameter as the frame axis
A simulation SHALL accept a sweep, a parameter and a sequence of values, in place of a time span; the system SHALL
carry no sweep. Each frame SHALL set the parameter to the next value and solve or advance the system once. The
trajectory SHALL name its axis after the parameter and hold the values in place of times. A sweep of an adaptive
parameter or of a parameter the system does not have SHALL be rejected with a named error.

#### Scenario: Dose sweep
- **WHEN** a static survival model runs with a sweep of the dose over 101 values from 0 to 10
- **THEN** the trajectory has 101 frames, its axis is the dose, and its axis values are the swept doses

### Requirement: Iterative procedures advance one step per time step
Systems with elements advanced by an external algorithm SHALL run under discrete time, compile to the procedure form,
and the procedure solver SHALL call one step of every such element per time step, in declaration order, with the run's
random stream. The procedure solver SHALL declare its support for discrete time and procedure laws, and validation
SHALL report a procedure law with a solver that does not support it.

#### Scenario: Procedure frames
- **WHEN** a search runs with `until=100`
- **THEN** the trajectory has 101 frames, the first holding the initial population

#### Scenario: Unsupported solver
- **WHEN** a system with a search is validated with the solver `'rk4'`
- **THEN** the report says that the solver does not support procedure laws and lists the procedure solver
