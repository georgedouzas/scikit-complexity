## ADDED Requirements

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

## REMOVED Requirements

### Requirement: Simulations take observables
**Reason**: Observables become the `observables` argument of `simulate`, as scoring is an argument of
`cross_validate`.
**Migration**: Pass the observables to `simulate`. A catalog entry's standard observables come with its tuned settings.

### Requirement: Solvers and backends are chosen separately from the system
**Reason**: Solvers and backends become arguments of `simulate`, and the tags of the system choose their defaults.
**Migration**: Pass `solver=` and `backend=` to `simulate`.

### Requirement: Parameters change while running
**Reason**: `simulate` runs in the calling thread and returns a trajectory, so nothing holds a running simulation to
change. A parameter that changes during a run is a function of time.
**Migration**: Pass `schedule={'growth.r': lambda t: ...}` to `simulate`. Scheduled values are checked against the
domains of their parameters at every frame. To change a parameter between two runs, start the second run from the last
frame of the first with `initial_state`.

### Requirement: Simulations run in the background
**Reason**: `simulate` is a function that returns a trajectory, as `cross_validate` is, and real-time watching needs
only a clock, not a background thread.
**Migration**: Call `simulate` and read its trajectory. To watch a run in Rerun as it goes, pass
`clock=RealTimeClock()` and a `RerunObserver`.

### Requirement: Group parameters change while running
**Reason**: A running simulation no longer takes parameter changes, so a group parameter changes during a run by its
schedule, like any other parameter.
**Migration**: Schedule the group parameter with one value for every member or one value per member, such as
`schedule={'gas.m': lambda t: 1.0 if t < 0.2 else 2.0}`.

### Requirement: Runs extend from their last frame
**Reason**: Runs are calls of `simulate`, which keeps no simulation object to extend. Continuing a run is a new call.
**Migration**: Pass the states of the last frame as the `initial_state` of the next `simulate` call, with the changed
parameters set on the system, and join the two trajectories.
