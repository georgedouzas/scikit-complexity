# Spec Delta

## MODIFIED Requirements

### Requirement: A solver is chosen for each form
A simulation SHALL choose a default solver for the form of its problem, and SHALL accept a solver chosen by name or
given as an instance. The default ODE solver SHALL detect stiffness and switch methods during the run, so stiff
systems run with the default solver, and SHALL choose a method whose storage fits the size of the state.

#### Scenario: Automatic solver
- **WHEN** a simulation of an ODE system runs with the solver left to its default
- **THEN** it uses the default ODE solver

#### Scenario: Solver for the wrong form
- **WHEN** a user gives a jump process solver to a simulation of an ODE system
- **THEN** running it raises a named error that names the form and the solver

#### Scenario: Stiff model built directly
- **WHEN** `Simulation(create('nanoparticle_corona')).run(until=10800)` runs with the solver left to its default
- **THEN** it finishes within 5 seconds, and its final state matches the run of its registered solver within a
  relative tolerance of 1e-4

#### Scenario: Non-stiff model keeps its accuracy
- **WHEN** a harmonic oscillator runs until 10 with the default solver
- **THEN** its final state matches the analytic solution within a relative tolerance of 1e-5

#### Scenario: Drift that vanishes at the start
- **WHEN** an oscillator at rest, forced by `sin(t)`, runs with the default solver
- **THEN** it leaves its zero drift at time 0 and follows the analytic solution within 1e-6

#### Scenario: Many coupled states
- **WHEN** 10 000 states coupled through their mean run with the default solver
- **THEN** the run uses a method other than `LSODA` and matches the analytic solution within 1e-5, and
  `ScipyODE(method='LSODA')` for the same system raises a named error that names the size of its dense matrix

#### Scenario: State resting on a discontinuity
- **WHEN** `bathtub` with 20 gallons runs until 6 with the default solver, so the empty tub rests on its drain switch
  once the faucet opens
- **THEN** the run warns that LSODA stalled, finishes with `RK45`, and the water drains to 0 by time 4, and
  `ScipyODE(method='LSODA')` for the same run raises a named error that says it stalled
