## MODIFIED Requirements

### Requirement: Ready-made systems use the general system of their shape
A ready-made system whose structure is that of a general system SHALL subclass it, imported from
`skcomplex.systems.general`, unless its element class is a component that other systems or users compose, such as an
element of a field `common` package. Its own element class SHALL then leave the catalog, and its equations SHALL move
into the hooks of the system.

#### Scenario: Map of one field
- **WHEN** a contributor reads `HenonMapSystem` of `mathematics/dynamical_systems`
- **THEN** it subclasses `DynamicalSystem` in discrete time, and the module defines no element class for it

#### Scenario: Shared element
- **WHEN** an element class such as the Lorenz element of `mathematics/common/flows` serves several systems
- **THEN** the systems that compose it keep it, and none of them rebases onto `DynamicalSystem`
