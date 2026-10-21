# simulation Specification

## Purpose

Runs a compiled problem with a solver that fits its form, offline or paced in real time, lets a user change parameters
while it runs, and records the result as a trajectory.

## Requirements

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

### Requirement: Simulations advance frame by frame
A simulation SHALL advance in frames of a fixed simulated duration, and SHALL hand each frame, with its time, state,
and parameters, to every observer. For continuous forms the default duration SHALL be 0.01, raised to the smallest
value of the form 1, 2, or 5 times a power of ten that keeps a bounded run at most 1000 frames.

#### Scenario: Frames of an ODE run
- **WHEN** an ODE simulation with frame duration 0.1 runs until time 1
- **THEN** its observers receive frames at times 0, 0.1, ..., 1

#### Scenario: Discrete time
- **WHEN** a discrete map simulation runs until step 10
- **THEN** its observers receive frames at the integer times 0 to 10

#### Scenario: Default frames of a short run
- **WHEN** an ODE simulation with the default frame duration runs until time 1
- **THEN** its observers receive 101 frames, 0.01 apart

#### Scenario: Default frames of a long run
- **WHEN** an ODE simulation with the default frame duration runs until time 1000
- **THEN** its observers receive 1001 frames, 1 apart, instead of 100001

### Requirement: Runs can be paced in real time
A simulation SHALL accept a clock, and the real-time clock SHALL pace frames so that simulated time advances at a given
speed relative to wall time, and SHALL record how far the run fell behind.

#### Scenario: Real-time pacing
- **WHEN** a simulation with a real-time clock of speed 1 runs from time 0 to 0.5
- **THEN** the run takes at least 0.5 seconds of wall time

### Requirement: Parameters change while running
A running simulation SHALL accept new parameter values by dotted name, and SHALL apply them before the next frame.
Values set between runs SHALL stay in force for the later runs of the same simulation.

#### Scenario: Live change
- **WHEN** a user sets the growth rate of a population to zero while its simulation runs in the background
- **THEN** the population stays constant in the frames after the change

#### Scenario: Unknown parameter
- **WHEN** a user sets a parameter name the problem does not have
- **THEN** a named error is raised that names the parameter

#### Scenario: Change between runs
- **WHEN** a user runs a logistic population until time 5, sets its growth rate to zero, and runs it again until 10
- **THEN** the second run uses the growth rate zero, and its parameters in every frame are the ones the user set

#### Scenario: Agent group parameter between runs
- **WHEN** a user runs an agent group whose rule adds the group parameter `rate` to a state until time 0, sets `rate`
  to 5, and runs it until time 2
- **THEN** the state of every agent is 10 at time 2

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

### Requirement: Runs extend from their last frame
A simulation that has run SHALL extend its run to a later time from its last frame, with its current state, parameters,
stepper, clock, and observers, and SHALL append the new frames to its trajectory. Extending a simulation that has not
run, or to a time not after its current time, SHALL raise a named simulation error.

#### Scenario: Run in two parts
- **WHEN** a user runs a logistic population until time 5 with frame duration 1, sets its growth rate to zero, and
  extends the run until time 10
- **THEN** the trajectory has 11 frames at times 0 to 10, follows the logistic curve until 5, and stays constant after

#### Scenario: Extend backwards
- **WHEN** a user extends a simulation that ran until time 5 to time 3
- **THEN** a named simulation error is raised that names the current time

### Requirement: SciPy integrators persist across frames
The SciPy ODE solver SHALL keep one integrator across the frames of a run, and SHALL restart it only when the
parameters change, when the state handed to it is not the state it returned, or at a breakpoint, event, or boundary.

#### Scenario: Many frames
- **WHEN** exponential decay runs with `ScipyODE` until time 10 with 1000 frames
- **THEN** the drift is evaluated at most twice as often as by one SciPy integration over the same interval, and the
  final state is the same within the solver tolerance

#### Scenario: Parameter change restarts
- **WHEN** the decay rate changes to zero at time 5 of the same run
- **THEN** the state stays constant from time 5 within the solver tolerance

### Requirement: Frames and trajectories carry rewritten structures
A frame of a rewriting system SHALL carry the structure of every rewriting element at that generation, its word,
patch, or host graph, with its geometry where an interpretation is declared. A trajectory SHALL return the structure of
an element at a frame by the element's path, and the recorded structures SHALL be copies that later generations do not
change.

#### Scenario: Structure of a generation
- **WHEN** the Cantor substitution has run 4 generations and the user asks the trajectory for the structure of
  `cantor` at frame 2
- **THEN** it returns a patch of 4 intervals, and their geometry is the same after the run as during it

#### Scenario: Unknown element
- **WHEN** the user asks a trajectory for the structure of a path that is not a rewriting element
- **THEN** it raises a key error that lists the rewriting elements of the system
