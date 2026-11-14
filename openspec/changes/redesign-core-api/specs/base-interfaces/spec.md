## ADDED Requirements

### Requirement: Each family has prepare and its own verb
Systems SHALL have `prepare` and `simulate`. Solvers, backends, observers, compilers, and kernels SHALL have `prepare`
and `run`, which the system calls. Elements, interactions, spaces, and neighbourhoods SHALL have neither.

#### Scenario: Listing the verbs
- **WHEN** a test lists the public methods of the public classes of `skcomplex.core`
- **THEN** systems have `prepare` and `simulate`, engine classes have `prepare` and `run`, and parts have only the
  parameter methods and their hooks

### Requirement: Extension hooks a user implements are public
The hooks a user implements to extend the library, `get_laws` on interactions and `build` on ready-made systems, SHALL
be public. Hooks that only the engine implements SHALL stay private.

#### Scenario: A user interaction
- **WHEN** a user writes an interaction that defines `get_laws`
- **THEN** the system calls it, and the user writes no method whose name starts with an underscore

### Requirement: Engine bases share prepare and run
Every base of an engine class SHALL derive from `BaseConfig` in `skcomplex.core.base`, which defines `prepare(context)`
and `run(input, final=False)` and calls the abstract hooks `_prepare` and `_run`. `BaseSystem` SHALL define `prepare`
and `simulate`. `BaseComponent` SHALL give parameters, cloning, and tags without either. No public `check` SHALL exist.

#### Scenario: One vocabulary in the engine
- **WHEN** a user calls `prepare(...)` and `run(...)` on a solver, a backend, and an observer
- **THEN** every call works, and every class is a `BaseConfig`

#### Scenario: A missing hook
- **WHEN** a user writes an engine class that implements `_prepare` but not `_run`, and creates it
- **THEN** creating it raises a `TypeError` that names `_run`

### Requirement: Public specs are the declarations and values users handle
A spec SHALL be public only when users build it or read it from a result, and SHALL then live in the package of the
types its fields name. The public declarations SHALL be `StateSpec`, `ParameterSpec`, and `RoleSpec`, with the aliases
`State`, `Parameter`, and `Role`. Engine records SHALL be private dataclasses beside the code that uses them.

#### Scenario: The declarations
- **WHEN** a user imports `StateSpec`, `ParameterSpec`, `RoleSpec`, `State`, `Parameter`, and `Role` from
  `skcomplex.core.specs`
- **THEN** every import succeeds, and `State[float]` declares a real state

#### Scenario: No engine record in the public API
- **WHEN** a test lists the public names of `skcomplex.core.specs`
- **THEN** none of `LoweredFunctionsSpec`, `DrawLogSpec`, or `EventRecordSpec` is among them

### Requirement: Prepare keeps what it derives and simulate returns its trajectory
`prepare` SHALL return the object and keep what it derives in attributes ending in `_`, such as `problem_`,
`solver_`, and `backend_` on a system. `simulate` and the experiment and analysis functions SHALL return their results.
Engine classes SHALL keep the results of `run` in the attributes their family documents.

#### Scenario: A system's problem
- **WHEN** a user prepares a system of rate laws
- **THEN** `prepare()` returns the system, and `problem_` is an ODE problem

#### Scenario: A simulation returns its trajectory
- **WHEN** a user calls `system.simulate(until=20)` twice with different initial states
- **THEN** each call returns its own trajectory, and the system stores neither

## MODIFIED Requirements

### Requirement: Running needs preparing
`simulate` SHALL raise `NotPreparedError` on a system that was not prepared or whose parameters changed since it was
prepared, and `run` SHALL raise it on an engine object that was not prepared. No main public method SHALL prepare an
object on its own.

#### Scenario: Running before preparing
- **WHEN** a user runs an `RK4Solver()` without preparing it
- **THEN** the run raises `NotPreparedError`, which names the solver and asks for `prepare`

#### Scenario: A spec runs without preparing
- **WHEN** a user looks for `run` on a spec
- **THEN** it does not exist, since a spec is a value and not an estimator

#### Scenario: Simulating before preparing
- **WHEN** a user calls `simulate` on a system that was never prepared
- **THEN** it raises `NotPreparedError`, which names the system and asks for `prepare`

#### Scenario: Parameters changed after preparing
- **WHEN** a user prepares a system, calls `set_params(gravity__g=1.62)`, and then `simulate`
- **THEN** it raises `NotPreparedError`, which names the changed parameter

### Requirement: Estimators are the components and the engine
The estimators of the library SHALL be the systems, which are prepared and simulated, and the engine classes, which
are prepared and run. Elements, interactions, spaces, and neighbourhoods SHALL be configurable parts of a system, as the
kernels of a Gaussian process are parts of it in scikit-learn. Experiments and calibration SHALL be functions.

#### Scenario: Listing estimators
- **WHEN** a test lists the public classes of `skcomplex.core` that have `prepare`
- **THEN** every one is a system or an engine class, and none is an element, an interaction, or an experiment

### Requirement: Ready-made systems are classes with model parameters
Every ready-made system SHALL be a class deriving from `BaseSystem`, whose constructor only stores the model's own
parameters, whose public hook `build` returns the `System` of its parts, and whose `prepare` builds and prepares it.
`get_params`, `set_params`, and `clone` SHALL work on the model parameters.

#### Scenario: Changing a model parameter
- **WHEN** a user calls `SIRSystem(beta=0.3).set_params(beta=0.5).prepare().simulate(until=10)`
- **THEN** the run uses the infection rate 0.5, and `get_params()` returns `beta=0.5`

## REMOVED Requirements

### Requirement: Configurable bases share two main public methods
**Reason**: Replaced by "Engine bases share prepare and run" and "Each family has prepare and its own verb". Parts are
no longer prepared or run.
**Migration**: Call `prepare` and `simulate` on systems only.

### Requirement: Specs live in one package
**Reason**: Replaced by "Public specs are the declarations and values users handle". Most specs were engine records
that no user reads.
**Migration**: Import the declarations from `skcomplex.core.specs`. Engine records are no longer importable.

### Requirement: Runs keep their results as attributes
**Reason**: Replaced by "Prepare keeps what it derives and simulate returns its trajectory". Laws and parts no longer
run.
**Migration**: Read `problem_` after `prepare`, and use the trajectory `simulate` returns.
