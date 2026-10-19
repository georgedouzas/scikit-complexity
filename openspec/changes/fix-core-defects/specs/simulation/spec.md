## MODIFIED Requirements

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

## ADDED Requirements

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
