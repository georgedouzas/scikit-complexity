## MODIFIED Requirements

### Requirement: General systems enter the core only when fields share them
A general system of the core SHALL be named after its mathematics, such as `MassActionSystem`, and SHALL enter the
core only when systems of two fields share its structure. It SHALL live in its own module of
`skcomplex.core.components.systems` and SHALL be re-exported from that package. A ready-made system SHALL subclass only
general systems of the core or systems of its own field.

#### Scenario: Shared structure
- **WHEN** an epidemic model and a chemical kinetics model both build compartments with mass-action rates
- **THEN** both subclass `MassActionSystem` of the core, and neither imports the other's field

#### Scenario: Import a general system
- **WHEN** a user imports `MassActionSystem` from `skcomplex.core.components.systems`
- **THEN** the import succeeds, and the class is a subclass of `BaseSystem` from the same package
