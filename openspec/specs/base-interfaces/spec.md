# base-interfaces Specification

## Purpose
Gives every base one vocabulary: configurable objects check, prepare, and run, keep their arguments as given and their
derived state in attributes ending in `_`, and values carry only fields and `check()`, with no hidden magic methods.

## Requirements

### Requirement: Configurable bases share three verbs
Every base on `BaseEstimator` SHALL define `check()`, `prepare(context)`, and `run(input, final=False)` once, each calling
a private hook a child overrides, with trivial defaults, so every configurable object answers all three. No
configurable class SHALL add a public verb beyond its constructors and the extension hooks a user implements.

#### Scenario: One vocabulary
- **WHEN** a user calls `check()`, `prepare(...)`, and `run(...)` on a space, a solver, a visual, an observer, a design,
  and a surrogate
- **THEN** every call works, and none of `initialize`, `advance`, `lower`, `on_frame`, `compile`, or `forecast`
  exists

#### Scenario: A new child through hooks
- **WHEN** a user writes a space that implements `_prepare` setting `size_`, `n_dim_`, and `positions_`, and `_run`
  returning neighbours
- **THEN** the space is checked, prepared, and run through the base's verbs, and the runtime uses it

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
Laws, domains, problems, and compiled records SHALL be frozen dataclasses, and terms strings of their value, all
without hand-written methods and compared by value, whose base defines the fields every child has and the method `check()`. Validation SHALL call
`check()` on every law of a system.

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
