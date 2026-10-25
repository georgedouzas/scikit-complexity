## ADDED Requirements

### Requirement: The core base holds the base classes of the framework
`skcomplex.core.base` SHALL define `BaseComponent`, `BaseInteraction`, `BaseSpace`, `BaseTime`, `BaseLaw`,
`BaseDomain`, `BaseProblem`, `BaseSolver`, and `BaseBackend`, and the helpers they are declared with, in modules
grouped by concept. It SHALL hold no name that only one formalism uses, and SHALL import no other package of
`skcomplex`.

#### Scenario: Base classes
- **WHEN** a user imports the nine base classes from `skcomplex.core.base`
- **THEN** every import succeeds

#### Scenario: Old base names are gone
- **WHEN** a user imports `Component` from `skcomplex.core.components`, `Law` from `skcomplex.core.laws`, `Space`
  from `skcomplex.core.space`, `Stepper` from `skcomplex.engine.solvers`, or anything from
  `skcomplex.core.interactions`
- **THEN** the import fails

#### Scenario: Formalism types
- **WHEN** a user imports `HilbertDomain` and `StructureDomain` from `skcomplex.core.base`
- **THEN** the import fails, and they import from `skcomplex.formalisms.quantum` and `skcomplex.formalisms.rewriting`

### Requirement: No base class is another base class
No base class SHALL derive from another. `BaseComponent`, `BaseInteraction`, `BaseSpace`, `BaseTime`,
`BaseSolver`, and `BaseBackend` SHALL derive from `sklearn.base.BaseEstimator` alone, and `BaseLaw`, `BaseDomain`,
and `BaseProblem` from no class. `System` SHALL derive from `BaseEstimator` alone.

#### Scenario: Unrelated bases
- **WHEN** a test checks every pair of base classes with `issubclass`
- **THEN** no base is a subclass of another

#### Scenario: System and interaction
- **WHEN** a test checks `System` and `BaseInteraction` against `BaseComponent`
- **THEN** neither is a subclass, and a system built with a nested system and an interaction compiles and runs

### Requirement: A class is named after what it is
Every class whose ancestry reaches a base, or the concrete roots `Problem`, `Solver`, `System`, and `Schedule`,
SHALL end with that concept's name: `Component`, `Interaction`, `Space`, `Time`, `Law`, `Domain`, `Problem`,
`Solver`, `Backend`, `Observer`, `Visual`, `Clock`, `Design`, `Surrogate`, `Pass`, or `Schedule`. This SHALL hold
in the library, its models, its tests, and its documents, and for names given to `define`.

#### Scenario: Framework names
- **WHEN** a user imports `RateLaw` from `skcomplex.core.laws`, `RK4Solver` from `skcomplex.engine.solvers`,
  `MeshSpace` from `skcomplex.formalisms.fields`, `ElementComponent` from `skcomplex.core.components`, and
  `ExternalSolver` from `skcomplex.engine.solvers`
- **THEN** every import succeeds, and `Rate`, `RK4`, `Mesh`, `Element`, and `ExternalBackend` are gone

#### Scenario: Model names
- **WHEN** every class that the model library defines or creates with `define` is listed after importing it
- **THEN** each class's name ends with the concept of the base it derives from

### Requirement: Every class derives from the base it is
Every element and group SHALL derive from `BaseComponent`, every interaction from `BaseInteraction`, every space from
`BaseSpace`, every time from `BaseTime`, every law from `BaseLaw`, every domain from `BaseDomain`, every problem from
`BaseProblem`, every solver from `BaseSolver`, and every backend from `BaseBackend`. Validation SHALL read a
domain's kind from the domain.

#### Scenario: Derived classes
- **WHEN** a test checks `ElementComponent` and `GroupComponent` against `BaseComponent`,
  `PairPotentialInteraction` against `BaseInteraction`, `QuantumProblem` against `BaseProblem`, `RK4Solver` and
  `LammpsSolver` against `BaseSolver`, and `NumbaBackend` against `BaseBackend`
- **THEN** every check holds, and `QuantumProblem` is not a subclass of `engine.compile.Problem`

#### Scenario: Formalism domain
- **WHEN** a quantum state declares a `HilbertDomain` and the system is validated
- **THEN** validation reads its kind from the domain, and `skcomplex.engine.compile` imports nothing from
  `skcomplex.formalisms.quantum`

#### Scenario: Solver over a base problem
- **WHEN** a solver is written against `BaseSolver` alone and initialized with a `BaseProblem` of its form
- **THEN** it advances the problem without importing `skcomplex.engine`
