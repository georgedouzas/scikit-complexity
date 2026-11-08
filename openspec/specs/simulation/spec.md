# simulation Specification

## Purpose

Runs a compiled problem with a solver that fits its form, offline or paced in real time, lets a user change parameters
while it runs, and records the result as a trajectory.

## Requirements

### Requirement: A solver is chosen for each form
A simulation SHALL compile its system with the compiler it is given, `PassCompiler()` by default, and SHALL run the
default solver that the compiler names for the problem it compiled, or the solver given as an instance. The default ODE
solver SHALL detect stiffness and switch methods during the run, so stiff systems run with the default solver, and
SHALL choose a method whose storage fits the size of the state.

#### Scenario: Automatic solver
- **WHEN** a simulation of an ODE system runs with the solver left to its default
- **THEN** it uses the default ODE solver

#### Scenario: Solver for the wrong form
- **WHEN** a user gives a jump process solver to a simulation of an ODE system
- **THEN** running it raises a named error that names the form and the solver

#### Scenario: Stiff system built directly
- **WHEN** `Simulation(create('nanoparticle_corona')).run(until=10800)` runs with the solver left to its default
- **THEN** it finishes within 5 seconds, and its final state matches the run of its registered solver within a
  relative tolerance of 1e-4

#### Scenario: Non-stiff system keeps its accuracy
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

### Requirement: Runs are recorded as trajectories
A simulation SHALL record its frames as a trajectory that returns the time points and the values of any state by its
dotted name.

#### Scenario: Read a state over time
- **WHEN** a user asks a trajectory for `rabbits.n`
- **THEN** it returns an array with one value per recorded frame

### Requirement: Stochastic runs are reproducible
A simulation SHALL accept a random state, and two runs with the same random state SHALL produce the same trajectory.
Runs of the same system with the same random state, solver, and step SHALL produce the same trajectory on the NumPy and
the Numba backends, within floating-point rounding, from any start time to any end time.

#### Scenario: Same seed
- **WHEN** a stochastic epidemic is simulated twice with random state 0
- **THEN** the two trajectories are equal

#### Scenario: Same seed on both backends
- **WHEN** geometric Brownian motion runs with Euler-Maruyama, a step of 0.1, and random state 0 from time 0.1 to time
  0.4 on the NumPy backend and on the Numba backend
- **THEN** both runs take the same number of steps and their final states agree to within 1e-12 relative error

### Requirement: Trajectories return group variables as arrays
A trajectory SHALL return a group variable by name as an array with one leading entry per frame followed by the group
shape.

#### Scenario: Positions of a gas over time
- **WHEN** a user asks the trajectory of a 100-frame run for `gas.x` of a group of 500 molecules in two dimensions
- **THEN** it returns an array of shape `(100, 500, 2)`

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
- **WHEN** a population system is given to `VelocityVerlet`
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

### Requirement: Simulations take initial states
A simulation SHALL take initial values that replace those of the components, by dotted state name. The whole state of
a quantum system SHALL be named `<system label>.state` and take a vector, a density matrix, or `'ground'`. A catalog
entry SHALL be able to declare the initial values its runs use, as a function of the factory's arguments. Limits of a
solver, such as the largest dimension of a Hilbert space, SHALL be arguments of the solver, not of the system.

#### Scenario: Ground state of a chain
- **WHEN** a spin chain is simulated with `initial_state={'chain.state': 'ground'}`
- **THEN** it starts from the lowest eigenstate of its Hamiltonian, and every expectation stays constant

#### Scenario: Unknown state name
- **WHEN** a simulation is given an initial value of a state the system does not have
- **THEN** it raises a validation error that names the state

#### Scenario: Dimension limit of the solver
- **WHEN** a quantum system above the `max_dimension` of its solver is compiled for that solver
- **THEN** it raises an error that names the subsystems, the dimension, and the limit

### Requirement: SDE steppers draw shared and correlated increments
The SDE steppers SHALL draw the increments of shared and correlated Wiener processes from the run's random stream, in
Python and in compiled code. Two runs with the same random state SHALL be equal, and a system that names no process
SHALL reproduce the trajectory it had before shared processes existed. A backend that cannot apply a shared process
SHALL not declare support for it, so validation rejects the combination before compilation.

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
- **WHEN** a system with a shared process is validated with the LAMMPS backend
- **THEN** the report holds an error that names the process's first law and lists the backends that support shared noise

### Requirement: A system package plugs in its own compiler
A compiler defined outside the engine, on `BaseCompiler`, SHALL compile a system passed to `Simulation` with
`compiler=`, and the simulation SHALL run the solver that compiler names, without the engine
importing the package.

#### Scenario: Rewriting from the systems library
- **WHEN** `Simulation(create_system('koch_curve'), compiler=RewritingCompiler())` runs, with `RewritingCompiler` imported
  from `skcomplex.core.engine.compile`
- **THEN** the run rewrites the word once per frame, and `skcomplex.core.engine` imports nothing from `skcomplex.systems`

#### Scenario: Quantum system without its compiler
- **WHEN** a quantum system runs with the default compiler
- **THEN** checking it raises a validation error that names `compiler=QuantumCompiler()`

### Requirement: A symplectic solver of fourth order
`Yoshida4Solver` SHALL integrate continuous-time systems by three velocity Verlet steps with the Yoshida coefficients,
after splitting the states into two blocks whose rates each read only the other block, a state of order 2 into its
position and velocity and states of order 1 by the sparsity of the Jacobian of the problem. A system with no such split
SHALL raise `SolverError` naming the state that prevents it and a solver that needs no split.

#### Scenario: Fourth order
- **WHEN** a harmonic oscillator written with `HamiltonianInteraction` runs to time 10 with steps of 0.1 and of 0.05
- **THEN** the error at the end shrinks by a factor between 14 and 18

#### Scenario: Non-separable Hamiltonian
- **WHEN** the restricted three-body problem, whose Hamiltonian couples coordinates and momenta in one term, is prepared
  with `Yoshida4Solver`
- **THEN** `SolverError` is raised that names a momentum whose rate reads a momentum and suggests `RK4Solver`

### Requirement: A simulation holds a default end time
A simulation SHALL take an optional end time `until` as an argument. `run()` without an end time SHALL run to it, an
end time passed to `run` SHALL take precedence, and a run with neither SHALL raise a validation error that asks for one.

#### Scenario: Stored end time
- **WHEN** a user calls `Simulation(system, until=20).prepare().run()`
- **THEN** the run ends at time 20

#### Scenario: Passed end time wins
- **WHEN** a user calls `Simulation(system, until=20).prepare().run(until=5)`
- **THEN** the run ends at time 5

#### Scenario: No end time
- **WHEN** a user calls `Simulation(system).prepare().run()`
- **THEN** it raises a validation error that shows how to pass an end time

### Requirement: Simulating is a function
`simulate(system, *, until, dt=None, initial_state=None, solver=None, backend=None, observables=(), observers=(),
random_state=None, ...)` SHALL prepare the system if needed, compile it, run it until `until` with frames every `dt`,
and return a `Trajectory`. A solver SHALL take the role of a splitter, a backend the role of `n_jobs`, observables the
role of `scoring`, and observers, such as a Rerun recording, the role of callbacks. The defaults SHALL come from the
tags of the system.

#### Scenario: Simulating a ready-made system
- **WHEN** a user calls `simulate(SIRSystem(), until=10)`
- **THEN** it returns a trajectory of the system up to time 10, with the default solver of its tags

#### Scenario: Tuned settings of a system class
- **WHEN** a user calls `simulate(SIRSystem())` without run settings
- **THEN** the run uses the end time, solver, observables, and initial state the class tunes

### Requirement: Simulating is paced by a clock
`simulate(..., clock=None)` SHALL pace the frames against wall time with `clock`, as fast as possible by default, and in
real time with `RealTimeClock()`, to watch a run in Rerun as it goes.

#### Scenario: Real-time pacing through a clock
- **WHEN** a user calls `simulate(system, until=0.5, dt=0.1, clock=RealTimeClock(speed=1.0))`
- **THEN** the run takes at least 0.5 seconds of wall time, and the clock records how far the run fell behind

### Requirement: Simulating checks states and scheduled values
With `checked=True`, `simulate` SHALL raise a `ValidationError` naming the state and the time when a state leaves its
domain, and `checked` SHALL default to `False`. A value a schedule sets SHALL be checked against the domain of its
parameter, as an initial value is, and against the transition and emission matrices it enters, and a failure SHALL
raise a `ValidationError`, or a `ParameterError`, which derives from it, naming the parameter and the time.

#### Scenario: Checked run
- **WHEN** a law moves a state of an alphabet of labels outside the alphabet in a run with `checked=True`
- **THEN** a `ValidationError` names the state, its values, and the time

#### Scenario: Scheduled probability above one
- **WHEN** a schedule sets the probability of staying in a two-state chain to 1.5 at time 3
- **THEN** a `ParameterError`, caught by `except ValidationError`, names the parameter, the time 3, and the negative
  entry of the transition matrix
