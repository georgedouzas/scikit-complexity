## ADDED Requirements

### Requirement: A base defines the public API of its family
Every public method and property of a family SHALL be defined once on its base, which calls a private hook that a
child overrides. A child SHALL add no public method beyond its constructors and the extension hooks a user implements,
and a family concept that does not apply to a child SHALL have a trivial answer from the base.

#### Scenario: Same questions for every space
- **WHEN** a user asks `size`, `n_dim`, `shape`, `torus`, `positions()`, and `neighbors(site)` of a lattice, a
  network, a Cayley, a mesh, and a continuous space
- **THEN** every call answers, with `None` for the size and an empty array for the positions of the continuous space

#### Scenario: Children add no public methods
- **WHEN** the public names of every class of a family are listed
- **THEN** each name is a name of the base, a constructor of the class, or an extension hook of the class that calls
  it

#### Scenario: A new child through hooks
- **WHEN** a user writes a space that implements `_n_dim`, `_size`, and `_positions`
- **THEN** `size`, `n_dim`, `positions()`, and `check()` of the base answer for it

### Requirement: Behaviour is named, not magic
No class of `core`, `engine`, or `formalisms` SHALL define a magic method except `__init__`, the arithmetic of the
quantum operators, the collection protocol of collections, and `__getattr__` of the symbol namespace of laws.

#### Scenario: Magic methods
- **WHEN** every class of `core`, `engine`, and `formalisms` is listed with the magic methods it defines itself
- **THEN** none defines one outside the four allowed

#### Scenario: Designs combine by name
- **WHEN** a user combines a grid design and a rows design with `product`
- **THEN** the result runs every combination of their rows

### Requirement: Laws and records are checked by validation
Laws and value records SHALL NOT validate at construction. Validation SHALL call `check()` on every law of a system,
and the error SHALL name the component that holds the law.

#### Scenario: Invalid law
- **WHEN** a component returns a `RateLaw` whose value is a NumPy array, and the system is validated
- **THEN** the law is created without error, and validation raises an error that names the component and the law

### Requirement: Values are records and behaviour derives from estimators
Terms, domains, laws, compiled records, and problems SHALL be frozen dataclasses without hand-written methods,
compared by value. Components, interactions, spaces, times, solvers, backends, visuals, observers, designs, and
surrogates SHALL derive from their base on `BaseEstimator` with an explicit `__init__`.

#### Scenario: Terms by value
- **WHEN** a law kind is pickled and loaded in another process
- **THEN** it equals the registered kind with `==`

### Requirement: Surrogates are estimators
A surrogate SHALL take its settings in `__init__`, train with `fit(series)`, store `system_` and its errors, and
forecast with `forecast`, which the base defines by simulating the fitted system.

#### Scenario: Fit and forecast
- **WHEN** a user fits a `SINDySurrogate` to the series of a Lotka-Volterra run and forecasts 50 steps
- **THEN** `system_` is a system, `training_error_` is a number, and the forecast has 50 rows

### Requirement: The runtime reads problems through their base
The runtime, the visuals, and the analyses SHALL read a problem only through the API of `BaseProblem`, without
`getattr` fallbacks, and the engine SHALL call no method of a formalism's problem.

#### Scenario: Quantum run through the base
- **WHEN** a quantum system is simulated, recorded, and shown in Rerun
- **THEN** the run succeeds, and no module of `skcomplex.engine` reads a quantum problem's own methods
