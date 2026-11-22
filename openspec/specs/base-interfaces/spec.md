# base-interfaces Specification

## Purpose
Gives every base one vocabulary: configurable objects check, prepare, and run, keep their arguments as given and their
derived state in attributes ending in `_`, and values carry only fields and `check()`, with no hidden magic methods.

## Requirements

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
No class of `skcomplex.core` SHALL define a magic method except `__init__` and the arithmetic of
the quantum operators. Collections SHALL be read with named methods and properties, the symbol namespace of laws
SHALL store its symbols as plain attributes, and an agent rule SHALL read and set states through the properties of a
view class generated per group.

#### Scenario: Magic methods
- **WHEN** every class of `skcomplex.core` is listed with the magic methods it defines itself
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
`BaseProblemSpec`, without `getattr` fallbacks, and the engine SHALL call no method of a problem subclass.

#### Scenario: Quantum run through the base
- **WHEN** a quantum system is simulated, recorded, and shown in Rerun
- **THEN** the run succeeds, and no module of `skcomplex.core.engine` reads a quantum problem's own methods

### Requirement: Acting objects are classes, not dataclasses
An object that keeps state between calls or acts SHALL be a class under the base of its package, deriving from
`BaseConfig`, whose constructor only stores its settings and whose state lives in attributes ending in `_`.

#### Scenario: Observer
- **WHEN** a user inspects the class that records observables during a run
- **THEN** it derives from `BaseConfig` and is not a dataclass

### Requirement: Enums and protocols live in their modules
Every enum of the core SHALL live in the `enums` module of the lowest layer that reads it, `skcomplex.core.enums` or
`skcomplex.core.engine.enums`. Every type alias and protocol SHALL live in the `types` module of its layer, and every
protocol SHALL end in `Protocol`.

#### Scenario: Kinds from enums
- **WHEN** a user imports `NoiseKind`, `OrderKind`, and `FormKind` from `skcomplex.core.enums`, and
  `AgentViewProtocol` from `skcomplex.core.components.types`
- **THEN** every import succeeds, and no module of the core outside an `enums` module defines an enum, and none
  outside a `types` module defines a protocol

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
derives anything or sets any attribute ending in `_`, raising the error that names the component and the argument at
fault. A bad setting SHALL be reported before a missing or wrong context. An owner SHALL prepare the components it
owns, and SHALL call `_check` on the specs it reads. A composite that fails on one component MAY leave earlier
components prepared, and preparing again SHALL replace their attributes.

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
- **WHEN** a user lists the attributes of `BaseConfig`, `BaseSpec`, and `BaseComponent`, and the names in
  `skcomplex.core.errors`
- **THEN** none has `check`, `BaseConfig` and `BaseComponent` have no `_check` and no `_validated`, `BaseSpec` has no
  `prepare` and no `run`, and `NotCheckedError` does not exist

### Requirement: Estimators are the components and the engine
The estimators of the library SHALL be the systems, which are prepared and simulated, and the engine classes, which
are prepared and run. Elements, interactions, spaces, and neighbourhoods SHALL be components that a system prepares,
with no `run` and no `simulate`. Experiments and calibration SHALL be functions.

#### Scenario: Listing estimators
- **WHEN** a test lists the public classes of `skcomplex.core` that have `simulate` or `run`
- **THEN** every one is a system or an engine class, and none is an element, an interaction, or an experiment

### Requirement: Systems form a class hierarchy
`BaseSystem` SHALL be the base of every system: it SHALL hold `prepare`, `run`, `get_tags()` derived from the
components, nesting, and the passing of model parameters down to the components. `CompositeSystem(BaseSystem)` SHALL
be the generic composite of elements, interactions, and nested systems. A class SHALL NOT be chosen by mathematical
form, which the tags give.

#### Scenario: Generic composite
- **WHEN** a user builds `CompositeSystem(elements=[...], interactions=[...])`
- **THEN** it is a `BaseSystem`, and its tags come from its components

### Requirement: Ready-made systems are classes with model parameters
Every ready-made system SHALL be a class deriving from `BaseSystem`, whose constructor only stores the model's own
parameters, and whose `_prepare` sets `system_`, the `System` of its components, which `prepare` then compiles.
`get_params`, `set_params`, and `clone` SHALL work on the model parameters.

#### Scenario: Changing a model parameter
- **WHEN** a user calls `SIRSystem(beta=0.3).set_params(beta=0.5).prepare().simulate(until=10)`
- **THEN** the run uses the infection rate 0.5, and `get_params()` returns `beta=0.5`

### Requirement: General systems enter the core only when fields share them
The core SHALL hold only the systems that are building blocks: `BaseSystem`, the base of every system, and
`CompositeSystem`, the composite of elements, interactions, and nested systems that the engine flattens. A general
system that builds a `CompositeSystem` from core components, such as `ReactionNetworkSystem`, is a recipe. It SHALL be
named after its mathematics, never after a field quantity, and SHALL live in a subpackage of
`skcomplex.systems.general` when systems of two fields share its structure, or in the `common` package of one field
otherwise. A ready-made system SHALL subclass only `BaseSystem`, general systems of `skcomplex.systems.general`, or
systems of its own field.

#### Scenario: Shared structure
- **WHEN** an epidemic model and a chemical kinetics model both build species and reactions with product propensities
- **THEN** both subclass `ReactionNetworkSystem` of `skcomplex.systems.general.reactions`, and neither imports the
  other's field

#### Scenario: Import a general system
- **WHEN** a user imports `ReactionNetworkSystem` from `skcomplex.systems.general.reactions`
- **THEN** the import succeeds, and the class is a subclass of `BaseSystem` from `skcomplex.core.components.systems`

#### Scenario: The core holds no general system
- **WHEN** a user lists `skcomplex.core.components.systems`
- **THEN** it exports `BaseSystem`, `RunSettings`, and `CompositeSystem`, and importing `ReactionNetworkSystem` from it
  fails with `ImportError`

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
