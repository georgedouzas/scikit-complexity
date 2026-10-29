# base-interfaces Specification

## Purpose
Gives every base one vocabulary: configurable objects check, prepare, and run, keep their arguments as given and their
derived state in attributes ending in `_`, and values carry only fields and `check()`, with no hidden magic methods.

## Requirements

### Requirement: Configurable bases share three verbs
Every base class of the library SHALL derive from `BaseConfig` in `skcomplex.base`, which defines `check()`,
`prepare(context)`, and `run(input, final=False)` once, each calling a private hook. `BaseConfig` SHALL declare the
hooks `_check`, `_prepare`, and `_run` as abstract methods, so a class that does not implement all three cannot be
instantiated, and SHALL give only the optional hook `_release` a default that does nothing. A class SHALL implement
each hook itself or inherit it from the base of its family, which implements the hooks its whole family shares, such
as the `_check` of every pass or solver. `BaseSpec` SHALL be the only base that implements all three hooks for its
children: `_check` accepts every field, and `_prepare` and `_run` do nothing. No base SHALL derive from
`BaseEstimator` directly, and no configurable class SHALL add a public verb beyond its constructors and the extension
hooks a user implements.

#### Scenario: One vocabulary
- **WHEN** a user calls `check()`, `prepare(...)`, and `run(...)` on a space, a solver, a backend, a visual, an
  observer, a design, and an analysis
- **THEN** every call works, every class is a `BaseConfig`, and none of `initialize`, `advance`, `lower`,
  `on_frame`, `compile`, or `forecast` exists

#### Scenario: A new child through hooks
- **WHEN** a user writes a space that implements `_check`, `_prepare` setting `size_`, `n_dim_`, and `positions_`,
  and `_run` setting the neighbours it finds
- **THEN** the space is checked, prepared, and run through the base's verbs, and the runtime uses it

#### Scenario: A missing hook
- **WHEN** a user writes a configurable class that implements `_check` and `_prepare` but not `_run`, and creates it
- **THEN** creating it raises a `TypeError` that names `_run`

#### Scenario: A hook from the family base
- **WHEN** a user writes a pass that implements only `_prepare` and `_run`
- **THEN** the pass is created, and checking it uses the `_check` of the pass base

#### Scenario: A spec without hooks
- **WHEN** a user writes a frozen spec with two fields and no hook, and calls `check()`, `prepare()`, and `run()` on it
- **THEN** every verb returns the spec unchanged

### Requirement: Constructors only store their arguments
Every `__init__` SHALL assign each argument to the attribute of the same name and do nothing else. Every other
attribute SHALL end in `_` and be set only by `prepare` or `run`, and the checks of arguments SHALL run in `check()`.

#### Scenario: Invalid argument
- **WHEN** a user builds a lattice space with a negative side and then validates a system that holds it
- **THEN** construction succeeds, and validation raises an error that names the space and the side

#### Scenario: Facts are prepared
- **WHEN** a user prepares a lattice space of shape `(4, 5)`
- **THEN** `size_` is 20, `n_dim_` is 2, and `positions_` has 20 rows

### Requirement: Values carry fields and a check
Every dataclass of the library outside the models SHALL be a frozen spec that derives from `BaseSpec`, itself a
`BaseConfig`, SHALL be named with the suffix `Spec`, and SHALL be importable from `skcomplex.specs`. A spec SHALL hold
only its fields, `check()`, and readers of its own fields. Validation SHALL call `check()` on every law of a system.

#### Scenario: Invalid law
- **WHEN** a component returns a `RateLaw` whose value is a NumPy array, and the system is validated
- **THEN** the law is created without error, and validation raises an error that names the component and the law

#### Scenario: Terms by value
- **WHEN** a law kind is pickled and loaded in another process
- **THEN** it equals the registered kind with `==`

### Requirement: Behaviour is named, not magic
No class of `core`, `engine`, or `formalisms` SHALL define a magic method except `__init__` and the arithmetic of
the quantum operators. Collections SHALL be read with named methods and properties, the symbol namespace of laws
SHALL store its symbols as plain attributes, and an agent rule SHALL read and set states through the properties of a
view class generated per group.

#### Scenario: Magic methods
- **WHEN** every class of `core`, `engine`, and `formalisms` is listed with the magic methods it defines itself
- **THEN** none defines one besides `__init__` and the quantum operator arithmetic

#### Scenario: Reading a trajectory
- **WHEN** a user runs a Lotka-Volterra model and reads `trajectory.values('prey.n')`
- **THEN** the values of every frame return, and `trajectory['prey.n']` raises `TypeError`

#### Scenario: Unknown symbol
- **WHEN** a law reads `s.rabit` in a component that declares `rabbit`
- **THEN** Python raises `AttributeError`, which suggests `rabbit`

#### Scenario: Agent views
- **WHEN** a rule sets `agent.food += 1` and reads `agent.wealth` of an agent whose group declares no `wealth`
- **THEN** the state changes, and reading `wealth` raises Python's own `AttributeError`

#### Scenario: Designs combine by name
- **WHEN** a user combines a grid design and a rows design with `product`
- **THEN** the result runs every combination of their rows

### Requirement: Surrogates are estimators
A surrogate SHALL take its settings in `__init__`, learn with `prepare(series)`, store `system_` and its errors, and
forecast with `run(steps)`, which the base defines by simulating the prepared system.

#### Scenario: Prepare and run
- **WHEN** a user prepares a `SINDySurrogate` to the series of a Lotka-Volterra run and runs it for 50 steps
- **THEN** `system_` is a system, `training_error_` is a number, and the forecast has 50 rows

### Requirement: The runtime reads problems through their base
The runtime, the visuals, and the analyses SHALL read a problem only through the fields and readers of
`BaseProblem`, without `getattr` fallbacks, and the engine SHALL call no method of a formalism's problem.

#### Scenario: Quantum run through the base
- **WHEN** a quantum system is simulated, recorded, and shown in Rerun
- **THEN** the run succeeds, and no module of `skcomplex.engine` reads a quantum problem's own methods

### Requirement: Specs live in one package
Every spec SHALL be a frozen `BaseSpec` in a module of its own, in the lowest package that can import every type its
fields name: `skcomplex.specs` for specs built from root types, `skcomplex.core.specs` for specs that name core
classes, the engine package whose classes they name, such as `skcomplex.engine.problems` for the problem and kernel
specs, and `skcomplex.experiments.specs` for experiment specs. Each of these packages SHALL export its public specs.

#### Scenario: Specs from one place
- **WHEN** a user imports `StateSpec` and `SupportSpec` from `skcomplex.specs`, `NodeSpec` from
  `skcomplex.core.specs`, and `ProblemSpec` and `FrameSpec` from `skcomplex.engine.problems`
- **THEN** every import succeeds and each class is a frozen `BaseSpec`

#### Scenario: No scattered dataclass
- **WHEN** a test lists every dataclass defined in `skcomplex.core`, `skcomplex.engine`, `skcomplex.analysis`, and
  `skcomplex.experiments`
- **THEN** every one of them is a frozen `BaseSpec`

### Requirement: Acting objects are classes, not dataclasses
An object that keeps state between calls or acts SHALL be a class under the base of its package, deriving from
`BaseConfig`, whose constructor only stores its settings and whose state lives in attributes ending in `_`.

#### Scenario: Observer
- **WHEN** a user inspects the class that records observables during a run
- **THEN** it derives from `BaseConfig` and is not a dataclass

### Requirement: Kinds and protocols are types
Every enum of the library outside the models SHALL live in `skcomplex.types` and end in `Kind`, and every protocol
SHALL live in `skcomplex.types` and end in `Protocol`.

#### Scenario: Kinds from types
- **WHEN** a user imports `NoiseKind`, `OrderKind`, `FormKind`, and `RecordProtocol` from `skcomplex.types`
- **THEN** every import succeeds and no other module of the library defines an enum or a protocol

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
- **WHEN** a user checks, prepares, and runs a compiler on a system
- **THEN** `run()` returns the compiler, and `problem_` is the compiled problem

#### Scenario: A solver's state
- **WHEN** the runtime runs a prepared solver on one step
- **THEN** `run()` returns the solver, and `state_` is the state vector at the end of the step

#### Scenario: A law's contribution
- **WHEN** a user prepares a rate law and runs it at given values
- **THEN** `run()` returns the law, and `value_` is its rate at those values

#### Scenario: A simulation keeps its named input
- **WHEN** a user runs `Simulation(system).check().prepare().run(until=20)`
- **THEN** the call returns the simulation, and its trajectory covers the times up to 20

### Requirement: Running needs preparing
`run` SHALL raise `NotPreparedError` for a configurable object that was not prepared with `prepare`, as `prepare` and
`run` raise `NotCheckedError` for one that was not checked. A spec SHALL be exempt, since a frozen value cannot be
marked. No verb SHALL prepare an object on its own: a simulation runs only once `prepare` compiled its system.

#### Scenario: Running before preparing
- **WHEN** a user runs `Simulation(system).check().run(until=20)` without preparing it
- **THEN** the run raises `NotPreparedError`, which names the simulation and asks for `prepare`

#### Scenario: A spec runs without preparing
- **WHEN** a user checks a spec and runs it without preparing it
- **THEN** the run returns the spec unchanged

### Requirement: The root base holds only the verbs and hooks
`BaseConfig` SHALL define only the verbs `check`, `prepare`, and `run` and their hooks. Loading generated source
SHALL be the one action of a `SourceLoader` in the engine, which takes the cache directory in `prepare`, defaulting
to `~/.cache/skcomplex/kernels`, and runs on the generated source, keeping the imported module in `module_`. Every
class that generates source SHALL load it through a `SourceLoader`.

#### Scenario: Loading generated source
- **WHEN** a user prepares a `SourceLoader` with no cache directory and runs it on source that defines a function
- **THEN** `module_` holds the function, and the source is a file under `~/.cache/skcomplex/kernels`

#### Scenario: No helper on the root
- **WHEN** a user lists the attributes of `BaseConfig`
- **THEN** it has no method that loads source and no cache directory
