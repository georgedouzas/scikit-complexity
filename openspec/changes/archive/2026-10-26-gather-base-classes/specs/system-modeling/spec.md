## MODIFIED Requirements

### Requirement: A system is components, interactions, space, and time
A system SHALL be defined by its label, its components, its interactions, its space, and its time, and by nothing
else. A system SHALL accept no argument about how it runs: no solver, backend, form, noise convention, delay history,
activation schedule outside its time, or observables.

#### Scenario: Four-part system
- **WHEN** a user builds `System('ecology', components=[rabbits, foxes], interactions=[predation],
  space=None, time=ContinuousTime())`
- **THEN** the system compiles and runs, and its parameters list only the parameters of its components and interactions

#### Scenario: Execution arguments are rejected
- **WHEN** a user passes `noise=`, `history=`, `phases=`, or `observables=` to `System`
- **THEN** a `TypeError` is raised as for any unknown argument

## ADDED Requirements

### Requirement: Systems hold nested systems
A system SHALL hold nested systems among its parts without being a component, so systems nest to any depth, and an
interaction SHALL refer to a component of a nested system by a slash path relative to the system that holds the
interaction. Two direct children of a system with the same label SHALL be reported by validation.

#### Scenario: Interaction across nested systems
- **WHEN** a parent system holds a predation interaction between `forest/foxes` and `meadow/rabbits`
- **THEN** the compiled system couples the fox count of the forest system to the rabbit count of the meadow system

#### Scenario: Duplicate labels
- **WHEN** two direct children of a system share a label
- **THEN** validation reports one issue whose path is the system and whose message names the duplicated label

#### Scenario: System is not a component
- **WHEN** a user checks whether a `System` instance is an instance of `BaseComponent`
- **THEN** the check is false, and the instance is an instance of `BaseSystem`

## REMOVED Requirements

### Requirement: Systems are components
**Reason**: A system is the whole that holds components, not one of them. It used only the label of the component
base.
**Migration**: Check systems against `BaseSystem`. Nested systems keep working as parts, through the requirement
"Systems hold nested systems".
