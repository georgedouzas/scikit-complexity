# Spec Delta

## ADDED Requirements

### Requirement: Fixed-step steppers run compiled
The fixed-step steppers SHALL run all internal steps of a frame in compiled code, and SHALL include a fourth-order
Runge-Kutta stepper for ODE systems.

#### Scenario: RK4 accuracy
- **WHEN** exponential growth is integrated with `RK4` and step 0.001 until time 1
- **THEN** the result matches `e` within 1e-10 relative error

#### Scenario: Same stochastic results as before
- **WHEN** a stochastic epidemic is simulated with a fixed random state by the compiled Gillespie stepper twice
- **THEN** the two trajectories are equal

### Requirement: Large groups run in real time with interactions
A simulation SHALL advance a group of 100000 members with a short-range pair interaction fast enough to keep a
real-time clock of speed 1 with frames every 0.01 time units.

#### Scenario: Interacting gas in real time
- **WHEN** a gas of 100000 soft spheres with cutoff pair forces runs with `RK4` for 2 seconds with a real-time clock
- **THEN** the run falls behind by no more than 0.2 seconds
