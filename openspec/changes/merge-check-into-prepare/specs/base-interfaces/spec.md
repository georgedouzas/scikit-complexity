## RENAMED Requirements

- FROM: `### Requirement: Configurable bases share three verbs`
- TO: `### Requirement: Configurable bases share two verbs`

- FROM: `### Requirement: Values carry fields and a check`
- TO: `### Requirement: Values carry fields and validate when prepared`

## MODIFIED Requirements

### Requirement: Configurable bases share two verbs
Every base class of the library SHALL derive from `BaseConfig` in `skcomplex.base`, which defines
`prepare(context)` and `run(input, final=False)` once. `prepare` SHALL call the private hook `_check` and then `_prepare`,
and `run` SHALL call `_run`. No public `check` SHALL exist. `BaseConfig` SHALL declare the
hooks `_check`, `_prepare`, and `_run` as abstract methods, so a class that does not implement all three cannot be
instantiated, and SHALL give only the optional hook `_release` a default that does nothing. A class SHALL implement
each hook itself or inherit it from the base of its family, which implements the hooks its whole family shares, such
as the `_check` of every pass or solver. `BaseSpec` SHALL be the only base that implements all three hooks for its
children: `_check` accepts every field, and `_prepare` and `_run` do nothing. No base SHALL derive from
`BaseEstimator` directly, and no configurable class SHALL add a public verb beyond its constructors and the extension
hooks a user implements.

#### Scenario: One vocabulary
- **WHEN** a user calls `prepare(...)` and `run(...)` on a space, a solver, a backend, a visual, an
  observer, a design, and an analysis
- **THEN** every call works, every class is a `BaseConfig`, and none of `check`, `initialize`, `advance`, `lower`,
  `on_frame`, `compile`, or `forecast` exists

#### Scenario: A new child through hooks
- **WHEN** a user writes a space that implements `_check`, `_prepare` setting `size_`, `n_dim_`, and `positions_`,
  and `_run` setting the neighbours it finds
- **THEN** the space is prepared and run through the base's verbs, and the runtime uses it

#### Scenario: A missing hook
- **WHEN** a user writes a configurable class that implements `_check` and `_prepare` but not `_run`, and creates it
- **THEN** creating it raises a `TypeError` that names `_run`

#### Scenario: A hook from the family base
- **WHEN** a user writes a pass that implements only `_prepare` and `_run`
- **THEN** the pass is created, and preparing it uses the `_check` of the pass base

#### Scenario: A spec without hooks
- **WHEN** a user writes a frozen spec with two fields and no hook, and calls `prepare()` and `run()` on it
- **THEN** every verb returns the spec unchanged

### Requirement: Constructors only store their arguments
Every `__init__` SHALL assign each argument to the attribute of the same name and do nothing else. Every other
attribute SHALL end in `_` and be set only by `prepare` or `run`, and the checks of arguments SHALL run in `_check`, which `prepare` calls first.

#### Scenario: Invalid argument
- **WHEN** a user builds a lattice space with a negative side and then validates a system that holds it
- **THEN** construction succeeds, and validation raises an error that names the space and the side

#### Scenario: Facts are prepared
- **WHEN** a user prepares a lattice space of shape `(4, 5)`
- **THEN** `size_` is 20, `n_dim_` is 2, and `positions_` has 20 rows

### Requirement: Values carry fields and validate when prepared
Every dataclass of the library outside the models SHALL be a frozen spec that derives from `BaseSpec`, itself a
`BaseConfig`, SHALL be named with the suffix `Spec`, and SHALL be importable from `skcomplex.specs`. A spec SHALL hold
only its fields, the hook `_check`, and readers of its own fields. Validation SHALL validate every law of a system, and
a spec SHALL validate when it is prepared on its own or when its owner is prepared.

#### Scenario: Invalid law
- **WHEN** a component returns a `RateLaw` whose value is a NumPy array, and the system is validated
- **THEN** the law is created without error, and validation raises an error that names the component and the law

#### Scenario: Terms by value
- **WHEN** a law kind is pickled and loaded in another process
- **THEN** it equals the registered kind with `==`

### Requirement: Runs keep their results as attributes
Every verb of a configurable object SHALL return the object itself, and every hook SHALL return `None`. A run SHALL
keep what it produces in an attribute ending in `_`, named once for each family:
- elements, interactions, and systems keep their laws in `laws_`, and no other part has `laws_`;
- laws keep their contribution in `value_`;
- compilers keep the compiled problem in `problem_`;
- lowerings keep what they lowered in `lowered_`;
- backends keep the lowered functions in `functions_`;
- solvers keep the advanced state in `state_`;
- kernels keep their values in `values_`, `None` for a kernel that changes the state in place;
- searches keep the pairs they find in `pairs_`;
- printers keep the code they print in `code_`;
- observers of a compilation keep what they observed in `observed_`;
- the quantum lowering keeps its kernels in `kernels_`;
- designs keep their rows in `rows_`;
- market protocols keep their trades in `trades_`.

Analyses, experiments, simulations, observers of a run, clocks, and visuals SHALL keep their results in the
attributes they already document. The overrides of `run` that name their input, such as `Simulation.run(until=...)`,
SHALL forward to the base verb and return the object.

#### Scenario: A compiler's problem
- **WHEN** a user prepares and runs a compiler on a system
- **THEN** `run()` returns the compiler, and `problem_` is the compiled problem

#### Scenario: A solver's state
- **WHEN** the runtime runs a prepared solver on one step
- **THEN** `run()` returns the solver, and `state_` is the state vector at the end of the step

#### Scenario: A law's contribution
- **WHEN** a user prepares a rate law and runs it at given values
- **THEN** `run()` returns the law, and `value_` is its rate at those values

#### Scenario: A simulation keeps its named input
- **WHEN** a user runs `Simulation(system).prepare().run(until=20)`
- **THEN** the call returns the simulation, and its trajectory covers the times up to 20

### Requirement: Running needs preparing
`run` SHALL raise `NotPreparedError` for a configurable object that was not prepared with `prepare`. A spec SHALL be exempt, since a frozen value cannot be
marked. No verb SHALL prepare an object on its own: a simulation runs only once `prepare` compiled its system.

#### Scenario: Running before preparing
- **WHEN** a user runs `Simulation(system).run(until=20)` without preparing it
- **THEN** the run raises `NotPreparedError`, which names the simulation and asks for `prepare`

#### Scenario: A spec runs without preparing
- **WHEN** a user runs a spec without preparing it
- **THEN** the run returns the spec unchanged

### Requirement: The root base holds only the verbs and hooks
`BaseConfig` SHALL define only the verbs `prepare` and `run` and their hooks. Loading generated source
SHALL be the one action of a `SourceLoader` in the engine, which takes the cache directory in `prepare`, defaulting
to `~/.cache/skcomplex/kernels`, and runs on the generated source, keeping the imported module in `module_`. Every
class that generates source SHALL load it through a `SourceLoader`.

#### Scenario: Loading generated source
- **WHEN** a user prepares a `SourceLoader` with no cache directory and runs it on source that defines a function
- **THEN** `module_` holds the function, and the source is a file under `~/.cache/skcomplex/kernels`

#### Scenario: No helper on the root
- **WHEN** a user lists the attributes of `BaseConfig`
- **THEN** it has no method that loads source and no cache directory

## ADDED Requirements

### Requirement: Preparing validates first
`prepare` SHALL validate an object and the parts it owns before it derives anything, raising the error that names the
part and the argument at fault, so no object is left partly prepared by a validation failure. A bad setting SHALL be
reported before a missing or wrong context. `BaseConfig` SHALL define the private `_validated`, which runs the hook `_check`
and, on a part, validates its label and declared specs first, and returns the object. `prepare` SHALL call it, and an
owner inside the library SHALL call it on a part it does not prepare.

#### Scenario: A bad spec fails at prepare
- **WHEN** a user builds a system with an interaction that holds an invalid spec, and prepares a simulation of it
- **THEN** building succeeds, `prepare` raises an error that names the interaction and the argument, and no part of the
  system has an attribute ending in `_`

#### Scenario: An invalid setting without a context
- **WHEN** a user prepares a compiler that has an invalid setting and passes no system
- **THEN** the error names the setting, not the missing system

#### Scenario: Changing a setting
- **WHEN** a user prepares an object, sets a parameter to an invalid value with `set_params`, and prepares it again
- **THEN** the second `prepare` raises an error that names the parameter

#### Scenario: Removed names
- **WHEN** a user lists the attributes of `BaseConfig`, `BaseSpec`, and `BasePart`, and the names in `skcomplex.errors`
- **THEN** none has `check`, and `NotCheckedError` does not exist
