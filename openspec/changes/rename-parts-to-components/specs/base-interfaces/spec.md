## RENAMED Requirements

- FROM: `### Requirement: Estimators are the parts and the engine`
- TO: `### Requirement: Estimators are the components and the engine`

## MODIFIED Requirements

### Requirement: Runs keep their results as attributes
Every main public method of a configurable object SHALL return the object itself, and every hook SHALL return `None`. A run SHALL
keep what it produces in an attribute ending in `_`, named once for each family:
- elements, interactions, and systems keep their laws in `laws_`, and no other component has `laws_`,
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
The classes with `prepare` and `run` SHALL be the components a modeler defines, which are elements, interactions,
laws, spaces, times, and systems, and the engine that runs them, which is solvers, backends, compilers and their
passes, observers, visuals, clocks, kernels, and printers. Simulating, analysing, running experiments, and calibrating
SHALL be functions of a system or of its results.

#### Scenario: Listing estimators
- **WHEN** a test lists the public classes of `skcomplex.core` that have `prepare` and `run`
- **THEN** every one is a component or a class of the engine, and none is an experiment or a calibration

### Requirement: Systems form a class hierarchy
`BaseSystem` SHALL be the base of every system: it SHALL hold `prepare`, `run`, `get_tags()` derived from the
components, nesting, and the passing of model parameters down to the components. `CompositeSystem(BaseSystem)` SHALL
be the generic composite of elements, interactions, and nested systems. A class SHALL NOT be chosen by mathematical
form, which the tags give.

#### Scenario: Generic composite
- **WHEN** a user builds `CompositeSystem(elements=[...], interactions=[...])`
- **THEN** it is a `BaseSystem`, and its tags come from its components

### Requirement: Ready-made systems are classes with model parameters
Every ready-made system SHALL be a class deriving from `BaseSystem` or from a general system of the core, whose
constructor only stores the model's own parameters and whose `prepare` builds its components from them. `get_params`,
`set_params`, and `clone` SHALL work on those parameters.

#### Scenario: Changing a model parameter
- **WHEN** a user calls `SIRSystem(beta=0.3).set_params(beta=0.5).prepare()`
- **THEN** the prepared system's infection rate is 0.5, and `get_params()` returns `beta=0.5`

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
