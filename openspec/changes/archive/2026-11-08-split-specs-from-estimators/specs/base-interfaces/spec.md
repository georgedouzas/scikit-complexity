## RENAMED Requirements

- FROM: `### Requirement: Estimators are the parts and calibration`
- TO: `### Requirement: Estimators are the parts and the engine`

## MODIFIED Requirements

### Requirement: Estimators are the parts and the engine
The classes with `prepare` and `run` SHALL be the parts a modeler defines, which are elements, interactions, laws,
spaces, times, and systems, and the engine that runs them, which is solvers, backends, compilers and their passes,
observers, visuals, clocks, kernels, and printers. Simulating, analysing, running experiments, and calibrating SHALL be
functions of a system or of its results.

#### Scenario: Listing estimators
- **WHEN** a test lists the public classes of `skcomplex.core` that have `prepare` and `run`
- **THEN** every one is a part or a class of the engine, and none is an experiment or a calibration


### Requirement: Configurable bases share two main public methods
Every base class of an estimator of the library SHALL derive from `BaseConfig` in `skcomplex.core.base`, which defines
`prepare(context)` and `run(input, final=False)` once. `prepare` SHALL call the private hook `_prepare`, which checks the
settings of the object before it derives anything, and `run` SHALL call `_run`. No public `check` SHALL exist.
`BaseConfig` SHALL declare the hooks `_prepare` and `_run` as abstract methods, so a class that does not implement both
cannot be instantiated, and SHALL give only the optional hook `_release` a default that does nothing. A class SHALL
implement each hook itself or inherit it from the base of its family. Specs SHALL derive from `BaseSpec`, which is not a
`BaseConfig`. No base SHALL derive from `BaseEstimator` directly, and no configurable class SHALL add a public method
beyond its constructors and the extension hooks a user implements.

#### Scenario: One vocabulary
- **WHEN** a user calls `prepare(...)` and `run(...)` on a space, a solver, a backend, a visual, and an observer
- **THEN** every call works, every class is a `BaseConfig`, and none of `check`, `initialize`, `advance`, `lower`,
  `on_frame`, `compile`, or `forecast` exists

#### Scenario: A new child through hooks
- **WHEN** a user writes a space that implements `_prepare`, which checks its arguments and sets `size_`, `n_dim_`, and
  `positions_`, and `_run` setting the neighbours it finds
- **THEN** the space is prepared and run through the base's main public methods, and the runtime uses it

#### Scenario: A missing hook
- **WHEN** a user writes a configurable class that implements `_prepare` but not `_run`, and creates it
- **THEN** creating it raises a `TypeError` that names `_run`

#### Scenario: A hook from the family base
- **WHEN** a user writes a pass that implements only `_run`
- **THEN** the pass is created, and preparing it uses the `_prepare` of the pass base

#### Scenario: A spec without hooks
- **WHEN** a user writes a frozen spec with two fields and no hook
- **THEN** the spec has neither `prepare` nor `run`, and its owner checks it through `_check`

### Requirement: Constructors only store their arguments
Every `__init__` SHALL assign each argument to the attribute of the same name and do nothing else. Every other
attribute SHALL end in `_` and be set only by `prepare` or `run`, and the checks of arguments SHALL run at the start of
`_prepare`, before any attribute ending in `_` is set.

#### Scenario: Invalid argument
- **WHEN** a user builds a lattice space with a negative side and then prepares a system that holds it
- **THEN** construction succeeds, and preparing raises an error that names the space and the side

#### Scenario: Facts are prepared
- **WHEN** a user prepares a lattice space of shape `(4, 5)`
- **THEN** `size_` is 20, `n_dim_` is 2, and `positions_` has 20 rows

### Requirement: Values carry fields and validate when prepared
Every dataclass of the library outside the systems SHALL be a frozen spec that derives from `BaseSpec`, which is not a
`BaseConfig`, SHALL be named with the suffix `Spec`, and SHALL live in the package the base-interfaces rules give it. A
spec SHALL hold only its fields, the hook `_check`, and readers of its own fields. Its owner SHALL call `_check` when it
reads the spec, and preparing a system SHALL check every law and spec it holds.

#### Scenario: Invalid law
- **WHEN** a component returns a `RateLaw` whose value is a NumPy array, and the system is prepared
- **THEN** the law is created without error, and preparing raises an error that names the component and the law

#### Scenario: Terms by value
- **WHEN** a law kind is pickled and loaded in another process
- **THEN** it equals the registered kind with `==`

### Requirement: Running needs preparing
`run` SHALL raise `NotPreparedError` for a configurable object that was not prepared with `prepare`. No main public
method SHALL prepare an object on its own: a simulation runs only once its system is compiled.

#### Scenario: Running before preparing
- **WHEN** a user runs an `RK4Solver()` without preparing it
- **THEN** the run raises `NotPreparedError`, which names the solver and asks for `prepare`

#### Scenario: A spec runs without preparing
- **WHEN** a user looks for `run` on a spec
- **THEN** it does not exist, since a spec is a value and not an estimator

### Requirement: Preparing validates first
`prepare` SHALL check the settings of an object, and the specs it owns, at the start of `_prepare`, before the object
derives anything or sets any attribute ending in `_`, raising the error that names the part and the argument at fault.
A bad setting SHALL be reported before a missing or wrong context. An owner SHALL prepare the parts it owns, and SHALL
call `_check` on the specs it reads. A composite that fails on one part MAY leave earlier parts prepared, and preparing
again SHALL replace their attributes.

#### Scenario: A bad spec fails at prepare
- **WHEN** a user builds a system with an interaction that holds an invalid spec, and simulates it
- **THEN** building succeeds, and preparing raises an error that names the interaction and the argument, before the
  interaction sets any attribute ending in `_`

#### Scenario: An invalid setting without a context
- **WHEN** a user prepares a compiler that has an invalid setting and passes no system
- **THEN** the error names the setting, not the missing system

#### Scenario: Changing a setting
- **WHEN** a user prepares an object, sets a parameter to an invalid value with `set_params`, and prepares it again
- **THEN** the second `prepare` raises an error that names the parameter

#### Scenario: Removed names
- **WHEN** a user lists the attributes of `BaseConfig`, `BaseSpec`, and `BasePart`, and the names in
  `skcomplex.core.errors`
- **THEN** none has `check`, `BaseConfig` and `BasePart` have no `_check` and no `_validated`, `BaseSpec` has no
  `prepare` and no `run`, and `NotCheckedError` does not exist
