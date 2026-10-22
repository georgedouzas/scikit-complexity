# Spec Delta

## ADDED Requirements

### Requirement: Acausal component library and engineering models
The library SHALL provide, in `skcomplex.models.engineering.realizations`, the electrical, translational, rotational,
thermal, and hydraulic port domains and components with ports and equations. It SHALL provide the RLC circuit, Chua's
circuit, a quarter-car suspension, a DC motor with an inertia, and a heated room with a thermostat as registered
factories, each with the standard observables of its catalog entry.

#### Scenario: Chua's double scroll
- **WHEN** Chua's circuit runs with its standard component values
- **THEN** the trajectory visits both scrolls and its largest Lyapunov exponent is positive

#### Scenario: Domains from the library
- **WHEN** a user imports the electrical domain from `skcomplex.models.engineering.realizations`
- **THEN** it is a `PortDomain` of the core, and no core module names it

### Requirement: Acausal component models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, tagged with the time kind, space kind, component kinds, interaction bindings, and law classes of the system
vocabulary, and a model the inventory records as pending this change SHALL register under the name of its entry. A
model that the library already implements SHALL be rebuilt on the new formalism under the same name, not registered
twice.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `heated_room` in `engineering/control`
  - `quarter_car` in `engineering/mechatronics`
  - `dc_motor` in `engineering/mechatronics`

#### Scenario: New model recorded
- **WHEN** the change registers a model with no inventory entry
- **THEN** the inventory check fails until the entry is added with its sources and field path
