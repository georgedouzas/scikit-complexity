## MODIFIED Requirements

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
- **WHEN** `simulate(NanoparticleCoronaSystem(), until=10800)` runs with the solver left to its default
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
  `ScipyODESolver(method='LSODA')` for the same system raises a named error that names the size of its dense matrix

#### Scenario: State resting on a discontinuity
- **WHEN** `bathtub` with 20 gallons runs until 6 with the default solver, so the empty tub rests on its drain switch
  once the faucet opens
- **THEN** the run warns that LSODA stalled, finishes with `RK45`, and the water drains to 0 by time 4, and
  `ScipyODESolver(method='LSODA')` for the same run raises a named error that says it stalled

### Requirement: Fixed-step steppers run compiled
The fixed-step steppers SHALL run all internal steps of a frame in compiled code, and SHALL include a fourth-order
Runge-Kutta stepper for ODE systems and a velocity Verlet stepper for systems of positions and velocities.

#### Scenario: RK4 accuracy
- **WHEN** exponential growth is integrated with `RK4` and step 0.001 until time 1
- **THEN** the result matches `e` within 1e-10 relative error

#### Scenario: Velocity Verlet conserves energy
- **WHEN** a harmonic oscillator is integrated with `VelocityVerletSolver` and step 0.01 for 1000 time units
- **THEN** its energy stays within 1e-3 relative error of the initial energy

#### Scenario: Velocity Verlet needs position and velocity pairs
- **WHEN** a population system is given to `VelocityVerletSolver`
- **THEN** a named solver error names the states that are not positions or velocities

#### Scenario: Same stochastic results as before
- **WHEN** a stochastic epidemic is simulated with a fixed random state by the compiled Gillespie stepper twice
- **THEN** the two trajectories are equal

### Requirement: Large groups run in real time with interactions
A simulation SHALL advance a group of 100000 members with a short-range pair interaction fast enough to keep a
real-time clock of speed 1 with frames every 0.01 time units.

#### Scenario: Interacting gas in real time
- **WHEN** a gas of 100000 soft spheres with cutoff pair forces runs with `VelocityVerletSolver` and step 0.01 for 2 seconds
  with a real-time clock
- **THEN** the run falls behind by no more than 0.2 seconds

### Requirement: SciPy integrators persist across frames
The SciPy ODE solver SHALL keep one integrator across the frames of a run, and SHALL restart it only when the
parameters change, when the state handed to it is not the state it returned, or at a breakpoint, event, or boundary.

#### Scenario: Many frames
- **WHEN** exponential decay runs with `ScipyODESolver` until time 10 with 1000 frames
- **THEN** the drift is evaluated at most twice as often as by one SciPy integration over the same interval, and the
  final state is the same within the solver tolerance

#### Scenario: Parameter change restarts
- **WHEN** the decay rate changes to zero at time 5 of the same run
- **THEN** the state stays constant from time 5 within the solver tolerance

### Requirement: SDE steppers draw shared and correlated increments
The SDE steppers SHALL draw the increments of shared and correlated Wiener processes from the run's random stream, in
Python and in compiled code. Two runs with the same random state SHALL be equal, and a system that names no process
SHALL reproduce the trajectory it had before shared processes existed. A backend that cannot apply a shared process
SHALL not declare support for it, so validation rejects the combination before compilation.

#### Scenario: Same seed with correlated noise
- **WHEN** a system with a correlated Wiener element runs twice with random state 0 on the Numba backend
- **THEN** the two trajectories are equal

#### Scenario: Unchanged seeded runs
- **WHEN** geometric Brownian motion runs with random state 0 and `EulerMaruyamaSolver`
- **THEN** its trajectory equals the one stored before this change

#### Scenario: Shared process on both backends
- **WHEN** one process drives two states with unit noise on the NumPy and on the Numba backend
- **THEN** on each backend the two states keep their initial difference at every frame

#### Scenario: Backend without shared noise
- **WHEN** a system with a shared process is validated with the LAMMPS backend
- **THEN** the report holds an error that names the process's first law and lists the backends that support shared noise
