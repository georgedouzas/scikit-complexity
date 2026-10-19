# Spec Delta

## ADDED Requirements

### Requirement: Sweeps step a parameter as the frame axis
A run SHALL accept a sweep, a parameter and a sequence of values, in place of a time span. Each frame SHALL set the
parameter to the next value and solve or advance the system once. The trajectory SHALL name its axis after the
parameter and hold the values in place of times.

#### Scenario: Dose sweep
- **WHEN** a static survival model runs with a sweep of the dose over 101 values from 0 to 10
- **THEN** the trajectory has 101 frames, its axis is the dose, and its axis values are the swept doses

### Requirement: Iterative procedures advance one step per frame
Systems with elements advanced by an external algorithm SHALL compile to the procedure form, and its stepper SHALL
call one step of every such element per frame, in declaration order, with the run's random stream.

#### Scenario: Procedure frames
- **WHEN** a search runs with `until=100`
- **THEN** the trajectory has 101 frames, the first holding the initial population
