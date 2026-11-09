## MODIFIED Requirements

### Requirement: General systems enter the core only when fields share them
A general system of the core SHALL be named after its mathematics, never after a field quantity, such as
`ReactionNetworkSystem` or `DynamicalSystem`, and SHALL enter the core only when systems of two fields share its
structure, or the structure can serve two fields. It SHALL add a feature no other core part offers. It SHALL live in its
own module of `skcomplex.core.components.systems` and SHALL be re-exported from that package. A ready-made system SHALL
subclass only general systems of the core or systems of its own field.

#### Scenario: Shared structure
- **WHEN** an epidemic model and a chemical kinetics model both build species and reactions with product propensities
- **THEN** both subclass `ReactionNetworkSystem` of the core, and neither imports the other's field

#### Scenario: Import a general system
- **WHEN** a user imports `ReactionNetworkSystem`, `DynamicalSystem`, `NetworkDynamicalSystem`,
  `ReactionDiffusionSystem`, or `StockFlowSystem` from `skcomplex.core.components.systems`
- **THEN** the import succeeds, and the class is a subclass of `BaseSystem` from the same package

#### Scenario: Shape of one field
- **WHEN** a system shape appears in one field only, such as the Boolean networks of systems biology
- **THEN** it stays in the `common` package of that field, and no general system of the core holds it

#### Scenario: Shape covered by one core part
- **WHEN** a system shape is one core interaction on a group, such as a cellular automaton, a `CompositeSystem` with a
  `LocalRuleInteraction`
- **THEN** no general system duplicates it
