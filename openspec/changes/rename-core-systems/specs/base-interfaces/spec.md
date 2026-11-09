## MODIFIED Requirements

### Requirement: Systems form a class hierarchy
`BaseSystem` SHALL be the base of every system: it SHALL hold `prepare`, `run`, `get_tags()` derived from the parts,
nesting, and the passing of model parameters down to the parts. `CompositeSystem(BaseSystem)` SHALL be the generic composite of
elements, interactions, and nested systems. A class SHALL NOT be chosen by mathematical form, which the tags give.

#### Scenario: Generic composite
- **WHEN** a user builds `CompositeSystem(elements=[...], interactions=[...])`
- **THEN** it is a `BaseSystem`, and its tags come from its parts

### Requirement: General systems enter the core only when fields share them
A general system of the core SHALL be named after its mathematics, such as `ReactionNetworkSystem`, and SHALL enter the
core only when systems of two fields share its structure. It SHALL live in its own module of
`skcomplex.core.components.systems` and SHALL be re-exported from that package. A ready-made system SHALL subclass only
general systems of the core or systems of its own field.

#### Scenario: Shared structure
- **WHEN** an epidemic model and a chemical kinetics model both build species and reactions with product propensities
- **THEN** both subclass `ReactionNetworkSystem` of the core, and neither imports the other's field

#### Scenario: Import a general system
- **WHEN** a user imports `ReactionNetworkSystem` from `skcomplex.core.components.systems`
- **THEN** the import succeeds, and the class is a subclass of `BaseSystem` from the same package
