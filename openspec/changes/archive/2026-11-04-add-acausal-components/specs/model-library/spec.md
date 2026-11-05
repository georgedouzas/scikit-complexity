# Spec Delta

## ADDED Requirements

### Requirement: Acausal component library
The library SHALL provide the electrical connector and components with ports and equations in
`skcomplex.models.engineering.common.electrical`, and the thermal connector and components in
`skcomplex.models.engineering.control`.

#### Scenario: Connector from the library
- **WHEN** a user imports the electrical connector from `skcomplex.models.engineering.common.electrical`
- **THEN** it is a `ConnectorSpec`, and no module of the core names it

#### Scenario: Component Chua circuit
- **WHEN** Chua's circuit is assembled from the electrical components with the values of `chua`
- **THEN** its capacitor voltages follow the trajectory of `chua` within the tolerance of its test

### Requirement: Engineering models on components and Lagrangians
The library SHALL provide `heated_room` in `engineering/control`, and `dc_motor` and `quarter_car` in
`engineering/mechatronics`, as registered factories, each with the standard observables of its catalog entry.
`quarter_car` SHALL be built on `SpringDamperInteraction`, and the inertia of `dc_motor` on `EulerLagrangeInteraction`.

#### Scenario: Heated room
- **WHEN** `heated_room` runs from a cold room with the thermostat set to 20 degrees and a hysteresis band of 1 degree
- **THEN** the room temperature rises to the band and then stays within it, with the heater switching on and off

#### Scenario: Quarter car over a step
- **WHEN** `quarter_car` drives over a road step with its standard values
- **THEN** the body and wheel displacements settle at the step height, and the energy of the springs decays

### Requirement: Acausal component models are recorded in the model inventory
Every model this change adds SHALL register under the field path of its subfield and SHALL have an entry in the model
inventory, tagged with the time kind, space kind, component kinds, interaction bindings, and law classes of the system
it builds.

#### Scenario: Inventory models registered
- **WHEN** the change is applied and the inventory check runs
- **THEN** these entries are registered under their field paths and marked implemented:
  - `heated_room` in `engineering/control`
  - `quarter_car` in `engineering/mechatronics`
  - `dc_motor` in `engineering/mechatronics`
