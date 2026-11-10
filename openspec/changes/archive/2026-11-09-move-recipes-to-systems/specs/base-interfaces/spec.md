## MODIFIED Requirements

### Requirement: General systems enter the core only when fields share them
The core SHALL hold only the systems that are building blocks: `BaseSystem`, the base of every system, and
`CompositeSystem`, the composite of elements, interactions, and nested systems that the engine flattens. A general
system that builds a `CompositeSystem` from core parts, such as `ReactionNetworkSystem`, is a recipe. It SHALL be named
after its mathematics, never after a field quantity, and SHALL live in a subpackage of `skcomplex.systems.general` when
systems of two fields share its structure, or in the `common` package of one field otherwise. A ready-made system SHALL
subclass only `BaseSystem`, general systems of `skcomplex.systems.general`, or systems of its own field.

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
