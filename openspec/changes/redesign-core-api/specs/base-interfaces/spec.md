## ADDED Requirements

### Requirement: Each family has prepare and its own verb
Systems SHALL have `prepare` and `simulate`. Solvers, backends, observers, compilers, and kernels SHALL have `prepare`
and `run`, which the system calls. Elements, interactions, spaces, and neighbourhoods SHALL have `prepare` alone.

#### Scenario: Listing the verbs
- **WHEN** a test lists the public methods of the public classes of `skcomplex.core`
- **THEN** systems have `prepare` and `simulate`, engine classes have `prepare` and `run`, and the other components
  have `prepare` and the parameter methods

### Requirement: Public methods call the hooks subclasses implement
A base SHALL define each public method once, and a subclass SHALL implement the private hook the method calls, as
`BaseSearchCV.fit` calls `_run_search` in scikit-learn. An interaction SHALL implement `_prepare`, which sets `laws_`,
and a ready-made system SHALL implement `_prepare`, which sets `system_`.

#### Scenario: A user interaction
- **WHEN** a user writes an interaction whose `_prepare` sets `laws_`
- **THEN** preparing the interaction against a namespace sets `laws_`, and preparing a system with it compiles them

### Requirement: Engine bases share prepare and run
Every base of an engine class SHALL derive from `BaseConfig` in `skcomplex.core.base`, which defines `prepare(context)`
and `run(input, final=False)` and calls the abstract hooks `_prepare` and `_run`. `BaseSystem` SHALL define `prepare`
and `simulate`. `BaseComponent` SHALL give parameters, cloning, tags, and `prepare`, and no `run`. No public `check`
SHALL exist.

#### Scenario: One vocabulary in the engine
- **WHEN** a user calls `prepare(...)` and `run(...)` on a solver, a backend, and an observer
- **THEN** every call works, and every class is a `BaseConfig`

#### Scenario: A missing hook
- **WHEN** a user writes an engine class that implements `_prepare` but not `_run`, and creates it
- **THEN** creating it raises a `TypeError` that names `_run`

### Requirement: Values live beside the classes that use them
No specs package SHALL exist. A value that users build or read SHALL live in the package of the class that uses it, the
only declaration SHALL be `RoleSpec` with the alias `Role`, in the interactions package, and no `StateSpec` or
`ParameterSpec` SHALL exist. Engine records SHALL be private dataclasses beside the code that uses them.

#### Scenario: The declarations
- **WHEN** a user imports `RoleSpec` and `Role` from `skcomplex.core.components.interactions`
- **THEN** both imports succeed, and importing `skcomplex.core.specs` fails

#### Scenario: No engine record in the public API
- **WHEN** a test lists the public classes of `skcomplex.core`
- **THEN** none of `LoweredFunctionsSpec`, `DrawLogSpec`, or `EventRecordSpec` is among them, and every public class
  that ends with `Spec` is a value that a system, a test, or a document builds or reads

### Requirement: Prepare keeps what it derives and simulate returns its trajectory
`prepare` SHALL return the object and keep what it derives in attributes ending in `_`, such as `states_`,
`equations_`, `form_`, `solver_`, and `backend_` on a system, and `laws_` on an interaction. `simulate` and the
experiment and analysis functions SHALL return their results.
Engine classes SHALL keep the results of `run` in the attributes their family documents.

#### Scenario: A system's problem
- **WHEN** a user prepares a system of rate laws
- **THEN** `prepare()` returns the system, and `form_` is the ordinary differential equations

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
are prepared and run. Elements, interactions, spaces, and neighbourhoods SHALL be components that a system prepares,
with no `run` and no `simulate`. Experiments and calibration SHALL be functions.

#### Scenario: Listing estimators
- **WHEN** a test lists the public classes of `skcomplex.core` that have `simulate` or `run`
- **THEN** every one is a system or an engine class, and none is an element, an interaction, or an experiment

### Requirement: Ready-made systems are classes with model parameters
Every ready-made system SHALL be a class deriving from `BaseSystem`, whose constructor only stores the model's own
parameters, and whose `_prepare` sets `system_`, the `System` of its components, which `prepare` then compiles.
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
**Reason**: Replaced by "Values live beside the classes that use them". Most specs were engine records that no user
reads, and the others belong beside the classes that use them.
**Migration**: Import `RoleSpec` and `Role` from `skcomplex.core.components.interactions`, and other values from the
package of the class that uses them. Engine records are no longer importable.

### Requirement: Runs keep their results as attributes
**Reason**: Replaced by "Prepare keeps what it derives and simulate returns its trajectory". Laws and parts no longer
run.
**Migration**: Read `states_`, `equations_`, and `form_` after `prepare`, and use the trajectory `simulate` returns.
