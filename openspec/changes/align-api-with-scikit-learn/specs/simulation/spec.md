## ADDED Requirements

### Requirement: Simulating is a function
`simulate(system, *, until, dt=None, initial_state=None, solver=None, backend=None, observables=(), observers=(),
random_state=None)` SHALL prepare the system if needed, compile it, run it until `until` with frames every `dt`, and
return a `Trajectory`. A solver SHALL take the role of a splitter, a backend the role of `n_jobs`, observables the role
of `scoring`, and observers, such as a Rerun recording, the role of callbacks. The defaults SHALL come from the
tags of the system.

#### Scenario: Simulating a ready-made system
- **WHEN** a user calls `simulate(SIRSystem(), until=10)`
- **THEN** it returns a trajectory of the system up to time 10, with the default solver of its tags

#### Scenario: Tuned settings of a system class
- **WHEN** a user calls `simulate(SIRSystem())` without run settings
- **THEN** the run uses the end time, solver, observables, and initial state the class tunes

## REMOVED Requirements

### Requirement: Simulations take observables
**Reason**: Observables become the `observables` argument of `simulate`, as scoring is an argument of
`cross_validate`.
**Migration**: Pass the observables to `simulate`. A catalog entry's standard observables come with its tuned settings.

### Requirement: Solvers and backends are chosen separately from the system
**Reason**: Solvers and backends become arguments of `simulate`, and the tags of the system choose their defaults.
**Migration**: Pass `solver=` and `backend=` to `simulate`.
