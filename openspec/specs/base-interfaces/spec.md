# base-interfaces Specification

## Purpose
Gives every base one vocabulary: configurable objects check, prepare, and run, keep their arguments as given and their
derived state in attributes ending in `_`, and values carry only fields and `check()`, with no hidden magic methods.

## Requirements

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

### Requirement: Behaviour is named, not magic
No class of `core`, `engine`, or `formalisms` SHALL define a magic method except `__init__` and the arithmetic of
the quantum operators. Collections SHALL be read with named methods and properties, the symbol namespace of laws
SHALL store its symbols as plain attributes, and an agent rule SHALL read and set states through the properties of a
view class generated per group.

#### Scenario: Magic methods
- **WHEN** every class of `core`, `engine`, and `formalisms` is listed with the magic methods it defines itself
- **THEN** none defines one besides `__init__` and the quantum operator arithmetic

#### Scenario: Reading a trajectory
- **WHEN** a user runs a Lotka-Volterra system and reads `trajectory.values('prey.n')`
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

### Requirement: Surrogates are analysis functions
A surrogate SHALL be an analysis function of `skcomplex.core.analysis`, `mode_decomposition`, `sparse_identification`,
or `reservoir`, which takes its settings as keyword arguments and returns a frozen `ModeDecompositionResult`,
`SparseIdentificationResult`, or `ReservoirResult`. The result SHALL hold the fitted `system`, its `training_error` and
`validation_error`, and a `forecast` of `horizon` samples.

#### Scenario: Prepare and run
- **WHEN** a user calls `sparse_identification` on the series of a Lotka-Volterra run with `horizon=50`
- **THEN** its result's `system` is a system, `training_error` is a number, and `forecast` has 50 rows

### Requirement: The runtime reads problems through their base
The runtime, the visuals, and the analyses SHALL read a problem only through the fields and readers of
`BaseProblem`, without `getattr` fallbacks, and the engine SHALL call no method of a formalism's problem.

#### Scenario: Quantum run through the base
- **WHEN** a quantum system is simulated, recorded, and shown in Rerun
- **THEN** the run succeeds, and no module of `skcomplex.core.engine` reads a quantum problem's own methods

### Requirement: Specs live in one package
Every spec SHALL be a frozen `BaseSpec` in a module of its own, in the lowest package that can import every type its
fields name: `skcomplex.core.specs` for specs built from root types, `skcomplex.core.specs` for specs that name core
classes, the engine package whose classes they name, such as `skcomplex.core.engine.problems` for the problem and kernel
specs, and `skcomplex.core.experiments.specs` for experiment specs. Each of these packages SHALL export its public specs.

#### Scenario: Specs from one place
- **WHEN** a user imports `StateSpec` and `SupportSpec` from `skcomplex.core.specs`, `NodeSpec` from
  `skcomplex.core.specs`, and `ProblemSpec` and `FrameSpec` from `skcomplex.core.engine.problems`
- **THEN** every import succeeds and each class is a frozen `BaseSpec`

#### Scenario: No scattered dataclass
- **WHEN** a test lists every dataclass defined in `skcomplex.core.components`, `skcomplex.core.engine`, `skcomplex.core.analysis`, and
  `skcomplex.core.experiments`
- **THEN** every one of them is a frozen `BaseSpec`

### Requirement: Acting objects are classes, not dataclasses
An object that keeps state between calls or acts SHALL be a class under the base of its package, deriving from
`BaseConfig`, whose constructor only stores its settings and whose state lives in attributes ending in `_`.

#### Scenario: Observer
- **WHEN** a user inspects the class that records observables during a run
- **THEN** it derives from `BaseConfig` and is not a dataclass

### Requirement: Kinds and protocols are types
Every enum of the library outside the systems SHALL live in `skcomplex.core.types` and end in `Kind`, and every protocol
SHALL live in `skcomplex.core.types` and end in `Protocol`.

#### Scenario: Kinds from types
- **WHEN** a user imports `NoiseKind`, `OrderKind`, `FormKind`, and `RecordProtocol` from `skcomplex.core.types`
- **THEN** every import succeeds and no other module of the library defines an enum or a protocol

### Requirement: Runs keep their results as attributes
Every main public method of a configurable object SHALL return the object itself, and every hook SHALL return `None`. A run SHALL
keep what it produces in an attribute ending in `_`, named once for each family:
- elements, interactions, and systems keep their laws in `laws_`, and no other part has `laws_`,
- laws keep their contribution in `value_`,
- compilers keep the compiled problem in `problem_`,
- lowerings keep what they lowered in `lowered_`,
- backends keep the lowered functions in `functions_`,
- solvers keep the advanced state in `state_`,
- kernels keep their values in `values_`, `None` for a kernel that changes the state in place,
- searches keep the pairs they find in `pairs_`,
- printers keep the code they print in `code_`,
- observers of a compilation keep what they observed in `observed_`,
- the quantum lowering keeps its kernels in `kernels_`, and
- market protocols keep their trades in `trades_`.

Observers of a run, clocks, and visuals SHALL keep their results in the attributes they already document, and
analyses, experiments, and `simulate` SHALL return their results. The overrides of `run` that name their input, such as `BaseClock.run(elapsed=...)`,
SHALL forward to the base method and return the object.

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
- **WHEN** a user runs `simulate(system, until=20)`
- **THEN** the call returns the trajectory, which covers the times up to 20

### Requirement: Running needs preparing
`run` SHALL raise `NotPreparedError` for a configurable object that was not prepared with `prepare`. No main public
method SHALL prepare an object on its own: a simulation runs only once its system is compiled.

#### Scenario: Running before preparing
- **WHEN** a user runs an `RK4Solver()` without preparing it
- **THEN** the run raises `NotPreparedError`, which names the solver and asks for `prepare`

#### Scenario: A spec runs without preparing
- **WHEN** a user looks for `run` on a spec
- **THEN** it does not exist, since a spec is a value and not an estimator

### Requirement: The root base holds only the main public methods and hooks
`BaseConfig` SHALL define only the main public methods `prepare` and `run` and their hooks. Loading generated source
SHALL be the one action of a `SourceLoader` in the engine, which takes the cache directory in `prepare`, defaulting
to `~/.cache/skcomplex/kernels`, and runs on the generated source, keeping the imported module in `module_`. Every
class that generates source SHALL load it through a `SourceLoader`.

#### Scenario: Loading generated source
- **WHEN** a user prepares a `SourceLoader` with no cache directory and runs it on source that defines a function
- **THEN** `module_` holds the function, and the source is a file under `~/.cache/skcomplex/kernels`

#### Scenario: No helper on the root
- **WHEN** a user lists the attributes of `BaseConfig`
- **THEN** it has no method that loads source and no cache directory

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

### Requirement: Estimators are the parts and the engine
The classes with `prepare` and `run` SHALL be the parts a modeler defines, which are elements, interactions, laws,
spaces, times, and systems, and the engine that runs them, which is solvers, backends, compilers and their passes,
observers, visuals, clocks, kernels, and printers. Simulating, analysing, running experiments, and calibrating SHALL be
functions of a system or of its results.

#### Scenario: Listing estimators
- **WHEN** a test lists the public classes of `skcomplex.core` that have `prepare` and `run`
- **THEN** every one is a part or a class of the engine, and none is an experiment or a calibration

### Requirement: Systems form a class hierarchy
`BaseSystem` SHALL be the base of every system: it SHALL hold `prepare`, `run`, `get_tags()` derived from the parts,
nesting, and the passing of model parameters down to the parts. `CompositeSystem(BaseSystem)` SHALL be the generic composite of
elements, interactions, and nested systems. A class SHALL NOT be chosen by mathematical form, which the tags give.

#### Scenario: Generic composite
- **WHEN** a user builds `CompositeSystem(elements=[...], interactions=[...])`
- **THEN** it is a `BaseSystem`, and its tags come from its parts

### Requirement: Ready-made systems are classes with model parameters
Every ready-made system SHALL be a class deriving from `BaseSystem` or from a general system of the core, whose
constructor only stores the model's own parameters and whose `prepare` builds its parts from them. `get_params`,
`set_params`, and `clone` SHALL work on those parameters.

#### Scenario: Changing a model parameter
- **WHEN** a user calls `SIRSystem(beta=0.3).set_params(beta=0.5).prepare()`
- **THEN** the prepared system's infection rate is 0.5, and `get_params()` returns `beta=0.5`

### Requirement: General systems enter the core only when fields share them
A general system of the core SHALL be named after its mathematics, such as `MassActionSystem`, and SHALL enter the
core only when systems of two fields share its structure. It SHALL live in its own module of
`skcomplex.core.components.systems` and SHALL be re-exported from that package. A ready-made system SHALL subclass only
general systems of the core or systems of its own field.

#### Scenario: Shared structure
- **WHEN** an epidemic model and a chemical kinetics model both build compartments with mass-action rates
- **THEN** both subclass `MassActionSystem` of the core, and neither imports the other's field

#### Scenario: Import a general system
- **WHEN** a user imports `MassActionSystem` from `skcomplex.core.components.systems`
- **THEN** the import succeeds, and the class is a subclass of `BaseSystem` from the same package
