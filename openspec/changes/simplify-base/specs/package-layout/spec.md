## ADDED Requirements

### Requirement: The core base holds the bases of a system's parts
`skcomplex.core.base` SHALL hold seven modules: `_parts` with `BasePart`, `_elements` with `BaseElement`,
`_interactions` with `BaseInteraction`, `_laws` with `BaseLaw`, and the shared `_types`, `_errors`, and `_utils`. It SHALL define no other
base, and SHALL import no other package of `skcomplex`. The base of any other concept SHALL live in that concept's
package.

#### Scenario: Three bases
- **WHEN** a user imports `BasePart`, `BaseElement`, `BaseInteraction`, and `BaseLaw` from `skcomplex.core.base`
- **THEN** every import succeeds, and the modules of `skcomplex.core.base` are exactly `_parts`, `_elements`,
  `_interactions`, `_laws`, `_types`, `_errors`, and `_utils`

#### Scenario: Other bases
- **WHEN** a user imports `BaseSpace` and `BaseTime` from `skcomplex.core.space`, and `Layout` and `BaseProblem` from
  `skcomplex.core.problems`
- **THEN** every import succeeds, and none of them imports from `skcomplex.core.base`

### Requirement: Bases are unrelated
No base class SHALL derive from another, except that `BaseElement`, `BaseInteraction`, `BaseLaw`, and `System`
SHALL derive from `BasePart`. `BasePart`, `BaseSpace`, `BaseTime`, `BaseSolver`, and `BaseBackend` SHALL derive from
the framework's `BaseEstimator`, and `BaseDomain`, `BaseProblem`, and `BaseSystemCompiler` from no base.

#### Scenario: Unrelated bases
- **WHEN** a test checks every pair of base classes with `issubclass`
- **THEN** no base is a subclass of another, except the bases of parts, which are subclasses of `BasePart`

#### Scenario: System and interaction
- **WHEN** a test checks `System` and `BaseInteraction` against `BaseElement`
- **THEN** neither is a subclass, and a system built with a nested system and an interaction compiles and runs

### Requirement: Classes are named after their base
Every class whose ancestry reaches a base, or the concrete roots `Problem`, `Solver`, `System`, and `Schedule`, SHALL
end with that concept's name, such as `Element`, `Interaction`, `Space`, `Time`, `Law`, `Domain`, `Problem`,
`Solver`, or `Backend`, in the library, its models, its tests, its documents, and the names given to `define`.

#### Scenario: Framework names
- **WHEN** a user imports `RateLaw` from `skcomplex.core.laws`, `Element` and `GroupElement` from
  `skcomplex.core.components`, and `RK4Solver` from `skcomplex.engine.solvers`
- **THEN** every import succeeds

#### Scenario: Model names
- **WHEN** every class that the model library defines or creates with `define` is listed after importing it
- **THEN** each class's name ends with the concept of the base it derives from, and none ends with `Component`

## MODIFIED Requirements

### Requirement: Every class derives from the base it is
Every element and group SHALL derive from `BaseElement`, every interaction from `BaseInteraction`, every space from
`BaseSpace`, every time from `BaseTime`, every law from `BaseLaw`, every domain from `BaseDomain`, every problem from
`BaseProblem`, every solver from `BaseSolver`, and every backend from `BaseBackend`. Validation SHALL read a domain's kind from the domain.

#### Scenario: Derived classes
- **WHEN** a test checks `Element` and `GroupElement` against `BaseElement`, `PairPotentialInteraction` against
  `BaseInteraction`, and `RK4Solver` against `BaseSolver`
- **THEN** each is a subclass of its base

#### Scenario: Formalism domain
- **WHEN** a quantum state declares a `HilbertDomain` and the system is validated
- **THEN** validation reads its kind from the domain, and `skcomplex.engine.compile` imports nothing from
  `skcomplex.formalisms.quantum`

#### Scenario: Solver over a base problem
- **WHEN** a solver is written against `BaseSolver` alone and initialized with a `BaseProblem` of its form
- **THEN** it advances the problem without importing `skcomplex.engine`

## REMOVED Requirements

### Requirement: The core base holds the base classes of the framework
**Reason**: The base now holds only the bases of a system's three parts, with shared types, errors, and helpers.
**Migration**: Import `BaseElement`, `BaseInteraction`, and `BaseLaw` from `skcomplex.core.base`, `BaseSpace` and
`BaseTime` from `skcomplex.core.space`, and `BaseProblem`, `BaseSolver`, and `Layout` from `skcomplex.core.problems`.

### Requirement: No base class is another base class
**Reason**: Replaced by "Bases are unrelated", which names `BaseElement`.
**Migration**: Read `BaseComponent` as `BaseElement`.

### Requirement: A class is named after what it is
**Reason**: Replaced by "Classes are named after their base", where elements end in `Element`.
**Migration**: Rename every class ending in `Component` to end in `Element`.
