## ADDED Requirements

### Requirement: The core base holds the base classes of the framework
`skcomplex.core.base` SHALL define `BaseComponent`, `BaseInteraction`, `BaseSystem`, `BaseSpace`, `BaseTime`,
`BaseLaw`, `BaseDomain`, `BaseProblem`, `BaseSolver`, and `BaseBackend`, and the helpers they are declared with. It
SHALL hold no name that only one formalism uses, and SHALL import no other package of `skcomplex`.

#### Scenario: Base classes
- **WHEN** a user imports the ten base classes from `skcomplex.core.base`
- **THEN** every import succeeds

#### Scenario: Old base names are gone
- **WHEN** a user imports `Component` from `skcomplex.core.components`, `Law` from `skcomplex.core.laws`, `Space` or
  `Time` from `skcomplex.core.space`, `Stepper` from `skcomplex.engine.solvers`, `Backend` from
  `skcomplex.engine.compile`, or `ExternalBackend` from `skcomplex.engine.backends`
- **THEN** the import fails

#### Scenario: Formalism types
- **WHEN** a user imports `HilbertSpace` and `Structure` from `skcomplex.core.base`
- **THEN** the import fails, and they import from `skcomplex.formalisms.quantum` and `skcomplex.formalisms.rewriting`

### Requirement: No base class is another base class
No base class SHALL derive from another base class. `BaseComponent`, `BaseInteraction`, `BaseSystem`, `BaseSpace`,
`BaseTime`, `BaseSolver`, and `BaseBackend` SHALL derive from `sklearn.base.BaseEstimator` alone. `BaseLaw`,
`BaseDomain`, and `BaseProblem` SHALL derive from no class.

#### Scenario: Unrelated bases
- **WHEN** a test checks every pair of base classes with `issubclass`
- **THEN** no base is a subclass of another

#### Scenario: System and interaction
- **WHEN** a test checks `System` and `Interaction` against `BaseComponent`
- **THEN** neither is a subclass, and a system built with a nested system and an interaction compiles and runs

### Requirement: Every class derives from the base it is
Every element and group SHALL derive from `BaseComponent`, every interaction from `BaseInteraction`, the system from
`BaseSystem`, every space from `BaseSpace`, every time from `BaseTime`, every law from `BaseLaw`, every domain from
`BaseDomain`, every problem from `BaseProblem`, every solver from `BaseSolver`, and every backend from `BaseBackend`.
Validation SHALL read a domain's kind from the domain.

#### Scenario: Derived classes
- **WHEN** a test checks `Element` and `Group` against `BaseComponent`, `PairPotential` against `BaseInteraction`,
  `QuantumProblem` against `BaseProblem`, `RK4` and `LammpsSolver` against `BaseSolver`, and `NumbaBackend` against
  `BaseBackend`
- **THEN** every check holds, and `QuantumProblem` is not a subclass of `engine.compile.Problem`

#### Scenario: Formalism domain
- **WHEN** a quantum state declares a `HilbertSpace` domain and the system is validated
- **THEN** validation reads its kind from the domain, and `skcomplex.engine.compile` imports nothing from
  `skcomplex.formalisms.quantum`

#### Scenario: Solver over a base problem
- **WHEN** a solver is written against `BaseSolver` alone and initialized with a `BaseProblem` of its form
- **THEN** it advances the problem without importing `skcomplex.engine`
