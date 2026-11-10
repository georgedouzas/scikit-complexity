## ADDED Requirements

### Requirement: Systems of the core form a components package
The systems of the core SHALL live in the package `skcomplex.core.components.systems`, laid out like the other parts
of a system: a `_base` module with the base, one private module per class, and an `__init__` that re-exports the
public names. The package SHALL re-export `BaseSystem`, `RunSettings`, `System`, and `MassActionSystem`.
`skcomplex.core.components` SHALL hold no module of systems outside this package.

#### Scenario: Import the systems of the core
- **WHEN** a user imports `BaseSystem`, `RunSettings`, `System`, and `MassActionSystem` from
  `skcomplex.core.components.systems`
- **THEN** every import succeeds, and the modules of the package are `_base` and one private module per concrete
  system

#### Scenario: Former modules are gone
- **WHEN** a user imports `skcomplex.core.components.system` or `skcomplex.core.components.mass_action`
- **THEN** the import fails with `ModuleNotFoundError`, since the project ships no alias for a moved module

#### Scenario: Systems import the public package
- **WHEN** a module of `skcomplex.systems` imports `BaseSystem`, `RunSettings`, or `System`
- **THEN** it imports them from `skcomplex.core.components.systems`, and the import contracts of the checks session
  pass

### Requirement: The base of systems imports no other module of its package
The `_base` module of `skcomplex.core.components.systems` SHALL define `BaseSystem` and `RunSettings` and SHALL import
no other module of its package. A concrete system module SHALL import the base downward.

#### Scenario: Base module stands alone
- **WHEN** the imports of `skcomplex.core.components.systems._base` are listed
- **THEN** none of them names `_system`, `_mass_action`, or any other module of the package

#### Scenario: Building a system that is not a composite
- **WHEN** a subclass of `BaseSystem` builds its parts as a `BaseSystem` that is not its own composite and is prepared
- **THEN** `prepare` raises `ValidationError` naming the class, as before the move

#### Scenario: Composite of a ready-made system
- **WHEN** `LotkaVolterraSystem().prepare().system_` is read
- **THEN** it is an instance of `System` imported from `skcomplex.core.components.systems`
