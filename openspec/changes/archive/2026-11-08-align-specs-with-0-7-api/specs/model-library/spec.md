## RENAMED Requirements

- FROM: `### Requirement: Ready-made systems are registered factories`
- TO: `### Requirement: Ready-made systems are classes`

## MODIFIED Requirements

### Requirement: Ready-made systems are classes
Every ready-made system SHALL be a class on `BaseSystem`, or on a general core system such as `MassActionSystem`, in
the module of its subfield. It SHALL take its model parameters as arguments of its constructor and SHALL give its field
path, topics, references, and tuned run settings as class attributes. `all_systems()` SHALL find every such class.

#### Scenario: Browse the catalog
- **WHEN** a user calls `all_systems(field='biology')` or `all_systems(field='biology/ecology')`
- **THEN** the result holds the name and class of every system of that field or subfield, and each class gives its
  field, topics, and references as class attributes

#### Scenario: Create by name
- **WHEN** a user builds `LotkaVolterraSystem(a=0.2)`, imported from `skcomplex.systems.biology.ecology`
- **THEN** the result is a system that `simulate` runs like the same system built from its components, to the tuned
  end time of its class
