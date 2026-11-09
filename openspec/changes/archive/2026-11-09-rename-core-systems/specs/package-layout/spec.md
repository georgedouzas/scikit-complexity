## MODIFIED Requirements

### Requirement: Bases are unrelated
No base class SHALL derive from another, except that `BaseElement`, `BaseInteraction`, `BaseLaw`, and `CompositeSystem`
SHALL derive from `BasePart`, which derives from scikit-learn's `BaseEstimator`. `BaseSpace`, `BaseTime`, `BaseSolver`,
and `BaseBackend` SHALL derive from the framework's `BaseEstimator`, and `BaseDomain`, `BaseProblem`, and `BaseSystemCompiler` from no base.

#### Scenario: Unrelated bases
- **WHEN** a test checks every pair of base classes with `issubclass`
- **THEN** no base is a subclass of another, except the bases of parts, which are subclasses of `BasePart`

#### Scenario: System and interaction
- **WHEN** a test checks `CompositeSystem` and `BaseInteraction` against `BaseElement`
- **THEN** neither is a subclass, and a system built with a nested system and an interaction compiles and runs

### Requirement: Classes are named after their base
Every class whose ancestry reaches a base, or the concrete roots `Problem`, `Solver`, `CompositeSystem`, and `Schedule`, SHALL
end with that concept's name, such as `Element`, `Interaction`, `Space`, `Time`, `Law`, `Domain`, `Problem`,
`Solver`, or `Backend`, in the library, its systems, its tests, its documents, and the names given to `define`.

#### Scenario: Framework names
- **WHEN** a user imports `RateLaw` from `skcomplex.core.components.laws`, `Element` and `GroupElement` from
  `skcomplex.core.components`, and `RK4Solver` from `skcomplex.core.engine.solvers`
- **THEN** every import succeeds

#### Scenario: System names
- **WHEN** every class that the system library defines or creates with `define` is listed after importing it
- **THEN** each class's name ends with the concept of the base it derives from, and none ends with `Component`

### Requirement: Systems of the core form a components package
The systems of the core SHALL live in the package `skcomplex.core.components.systems`, laid out like the other parts
of a system: a `_base` module with the base, one private module per class, and an `__init__` that re-exports the
public names. The package SHALL re-export `BaseSystem`, `RunSettings`, `CompositeSystem`, and `ReactionNetworkSystem`.
`skcomplex.core.components` SHALL hold no module of systems outside this package.

#### Scenario: Import the systems of the core
- **WHEN** a user imports `BaseSystem`, `RunSettings`, `CompositeSystem`, and `ReactionNetworkSystem` from
  `skcomplex.core.components.systems`
- **THEN** every import succeeds, and the modules of the package are `_base` and one private module per concrete
  system

#### Scenario: Former modules are gone
- **WHEN** a user imports `skcomplex.core.components.system` or `skcomplex.core.components.mass_action`
- **THEN** the import fails with `ModuleNotFoundError`, since the project ships no alias for a moved module

#### Scenario: Former class names are gone
- **WHEN** a user imports `System` or `MassActionSystem` from `skcomplex.core.components.systems`
- **THEN** the import fails with `ImportError`, since the project ships no alias for a renamed class

#### Scenario: Systems import the public package
- **WHEN** a module of `skcomplex.systems` imports `BaseSystem`, `RunSettings`, or `CompositeSystem`
- **THEN** it imports them from `skcomplex.core.components.systems`, and the import contracts of the checks session
  pass

### Requirement: The base of systems imports no other module of its package
The `_base` module of `skcomplex.core.components.systems` SHALL define `BaseSystem` and `RunSettings` and SHALL import
no other module of its package. A concrete system module SHALL import the base downward.

#### Scenario: Base module stands alone
- **WHEN** the imports of `skcomplex.core.components.systems._base` are listed
- **THEN** none of them names `_composite`, `_reaction_network`, or any other module of the package

#### Scenario: Building a system that is not a composite
- **WHEN** a subclass of `BaseSystem` builds its parts as a `BaseSystem` that is not its own composite and is prepared
- **THEN** `prepare` raises `ValidationError` naming the class, as before the move

#### Scenario: Composite of a ready-made system
- **WHEN** `LotkaVolterraSystem().prepare().system_` is read
- **THEN** it is an instance of `CompositeSystem` imported from `skcomplex.core.components.systems`
