# simulation Specification

## Purpose

Runs a compiled problem with a solver that fits its form, offline or paced in real time, lets a user change parameters
while it runs, and records the result as a trajectory.

## Requirements

### Requirement: A solver is chosen for each form
A simulation SHALL choose a default solver for the form of its problem, and SHALL accept a solver chosen by name or
given as an instance.

#### Scenario: Automatic solver
- **WHEN** a simulation of an ODE system runs with the solver left to its default
- **THEN** it uses the default ODE solver

#### Scenario: Solver for the wrong form
- **WHEN** a user gives a jump process solver to a simulation of an ODE system
- **THEN** running it raises a named error that names the form and the solver

### Requirement: Simulations advance frame by frame
A simulation SHALL advance in frames of a fixed simulated duration, and SHALL hand each frame, with its time, state,
and parameters, to every observer.

#### Scenario: Frames of an ODE run
- **WHEN** an ODE simulation with frame duration 0.1 runs until time 1
- **THEN** its observers receive frames at times 0, 0.1, ..., 1

#### Scenario: Discrete time
- **WHEN** a discrete map simulation runs until step 10
- **THEN** its observers receive frames at the integer times 0 to 10

### Requirement: Runs can be paced in real time
A simulation SHALL accept a clock, and the real-time clock SHALL pace frames so that simulated time advances at a given
speed relative to wall time, and SHALL record how far the run fell behind.

#### Scenario: Real-time pacing
- **WHEN** a simulation with a real-time clock of speed 1 runs from time 0 to 0.5
- **THEN** the run takes at least 0.5 seconds of wall time

### Requirement: Parameters change while running
A running simulation SHALL accept new parameter values by dotted name, and SHALL apply them before the next frame.

#### Scenario: Live change
- **WHEN** a user sets the growth rate of a population to zero while its simulation runs in the background
- **THEN** the population stays constant in the frames after the change

#### Scenario: Unknown parameter
- **WHEN** a user sets a parameter name the problem does not have
- **THEN** a named error is raised that names the parameter

### Requirement: Simulations run in the background
A simulation SHALL start in a background thread, and SHALL support pausing, resuming, stopping, and waiting for the
run to finish.

#### Scenario: Stop a background run
- **WHEN** a user starts an unbounded background simulation and then stops it
- **THEN** the thread ends and the recorded trajectory holds the frames produced so far

### Requirement: Runs are recorded as trajectories
A simulation SHALL record its frames as a trajectory that returns the time points and the values of any state by its
dotted name.

#### Scenario: Read a state over time
- **WHEN** a user asks a trajectory for `rabbits.n`
- **THEN** it returns an array with one value per recorded frame

### Requirement: Stochastic runs are reproducible
A simulation SHALL accept a random state, and two runs with the same random state SHALL produce the same trajectory.

#### Scenario: Same seed
- **WHEN** a stochastic epidemic is simulated twice with random state 0
- **THEN** the two trajectories are equal

### Requirement: Trajectories return group variables as arrays
A trajectory SHALL return a group variable by name as an array with one leading entry per frame followed by the group
shape.

#### Scenario: Positions of a gas over time
- **WHEN** a user asks the trajectory of a 100-frame run for `gas.x` of a group of 500 molecules in two dimensions
- **THEN** it returns an array of shape `(100, 500, 2)`

### Requirement: Group parameters change while running
A running simulation SHALL accept a new value for a group parameter, either one value for every member or one value
per member.

#### Scenario: Heat one species
- **WHEN** a user sets the mass of every member of a group to 2 while the simulation runs
- **THEN** the frames after the change use mass 2 for every member

### Requirement: Groups run large in real time
A simulation SHALL advance a group of 10000 non-interacting members fast enough to keep a real-time clock of
speed 1 with frames every 0.01 time units.

#### Scenario: Ideal gas in real time
- **WHEN** an ideal gas of 10000 molecules runs for 5 seconds with a real-time clock of speed 1 and dt 0.01
- **THEN** the run falls behind by no more than 0.1 seconds

### Requirement: Fixed-step steppers run compiled
The fixed-step steppers SHALL run all internal steps of a frame in compiled code, and SHALL include a fourth-order
Runge-Kutta stepper for ODE systems and a velocity Verlet stepper for systems of positions and velocities.

#### Scenario: RK4 accuracy
- **WHEN** exponential growth is integrated with `RK4` and step 0.001 until time 1
- **THEN** the result matches `e` within 1e-10 relative error

#### Scenario: Velocity Verlet conserves energy
- **WHEN** a harmonic oscillator is integrated with `VelocityVerlet` and step 0.01 for 1000 time units
- **THEN** its energy stays within 1e-3 relative error of the initial energy

#### Scenario: Velocity Verlet needs position and velocity pairs
- **WHEN** a population model is given to `VelocityVerlet`
- **THEN** a named solver error names the states that are not positions or velocities

#### Scenario: Same stochastic results as before
- **WHEN** a stochastic epidemic is simulated with a fixed random state by the compiled Gillespie stepper twice
- **THEN** the two trajectories are equal

### Requirement: Large groups run in real time with interactions
A simulation SHALL advance a group of 100000 members with a short-range pair interaction fast enough to keep a
real-time clock of speed 1 with frames every 0.01 time units.

#### Scenario: Interacting gas in real time
- **WHEN** a gas of 100000 soft spheres with cutoff pair forces runs with `VelocityVerlet` and step 0.01 for 2 seconds
  with a real-time clock
- **THEN** the run falls behind by no more than 0.2 seconds

### Requirement: Trajectories record observables
Trajectories SHALL hold the value of every observable of every system at every frame under the observable's path,
alongside the states.

#### Scenario: Observable by path
- **WHEN** a user asks a trajectory for `market/volume`, an observable of the subsystem `market`
- **THEN** it returns one value per frame
