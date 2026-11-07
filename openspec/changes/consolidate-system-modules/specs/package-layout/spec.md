## ADDED Requirements

### Requirement: A system module defines its system completely
Every module of a subfield package of `skcomplex.systems` that registers a system SHALL hold the elements,
interactions, views, and helpers that only that system uses, so the module reads as a complete example. A subfield
package SHALL NOT hold a component module that serves a single system.

#### Scenario: Reading one system
- **WHEN** a reader opens the module that registers `active_walkers`
- **THEN** the element, interaction, and view classes that only `active_walkers` uses are defined in that module

#### Scenario: Component module of one system
- **WHEN** a subfield package holds a module that registers no system and is imported by the module of only one
  system
- **THEN** the layout test fails and names the module and the system that should hold it

## MODIFIED Requirements

### Requirement: System packages are self-contained
Every subfield package of `skcomplex.systems` SHALL be self-contained: it SHALL NOT import any other subfield package,
of its own field or another, except the `common` package of its own field, `skcomplex.systems.<field>.common`, which
holds the components that several systems of that field share. A component that more than one field needs SHALL be
copied into each field, or, when it is domain-agnostic, live in the core.

#### Scenario: Import across system packages
- **WHEN** a module of `skcomplex.systems.biology.collective_behaviour` imports from
  `skcomplex.systems.physics.mechanics`, or from `skcomplex.systems.biology.ecology`
- **THEN** the import contracts of the checks session fail and name the module and the import

#### Scenario: Realizations of the own field
- **WHEN** a module of `skcomplex.systems.physics.mechanics` imports from `skcomplex.systems.physics.common`
- **THEN** the import contracts of the checks session pass

#### Scenario: Removing a system package
- **WHEN** any one subfield package is deleted from the library
- **THEN** every other subfield package still imports, and its systems still pass the conformance suite
