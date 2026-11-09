## MODIFIED Requirements

### Requirement: A system is components, interactions, space, and time
A system SHALL be defined by its label, its components, its interactions, its space, and its time, and by nothing
else. A system SHALL accept no argument about how it runs: no solver, backend, form, noise convention, delay history,
activation schedule outside its time, or observables.

#### Scenario: Four-part system
- **WHEN** a user builds `CompositeSystem('ecology', components=[rabbits, foxes], interactions=[predation],
  space=None, time=ContinuousTime())`
- **THEN** the system compiles and runs, and its parameters list only the parameters of its components and interactions

#### Scenario: Execution arguments are rejected
- **WHEN** a user passes `noise=`, `history=`, `phases=`, or `observables=` to `CompositeSystem`
- **THEN** a `TypeError` is raised as for any unknown argument

### Requirement: Core packages mirror the system definition
The modelling packages of `skcomplex.core.components` SHALL be `system`, `components`, `interactions`, `space`, which holds space
and time, and `laws`, each exporting the general primitives of its part of a system. A primitive that belongs to one
formalism SHALL live in that formalism's package under `skcomplex.formalisms`, never in a modelling package.

#### Scenario: Primitives by part
- **WHEN** a user imports `CompositeSystem`, `Element`, `Interaction`, `LatticeSpace`, `DiscreteTime`, and `Rate`
- **THEN** they come from `core.system`, `core.components`, `core.interactions`, `core.space`, `core.space`, and
  `core.laws`

#### Scenario: Formalism primitives by formalism
- **WHEN** a user imports `Particle`, `PairPotential`, `Rule`, and `Field`
- **THEN** they come from `formalisms.particles`, `formalisms.particles`, `formalisms.agents`, and
  `formalisms.fields`

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
- **WHEN** a user checks whether a `CompositeSystem` instance is an instance of `BaseComponent`
- **THEN** the check is false, and the instance is an instance of `BaseSystem`

### Requirement: Every element derives from one base
Elements and groups SHALL derive from `BaseElement`, which holds a label, declares states, parameters, and roles, and
returns laws. A modeller SHALL define a type of element by subclassing `Element`.

#### Scenario: Hierarchy
- **WHEN** a user checks `Element`, `GroupElement`, and `FieldElement` against `BaseElement`
- **THEN** each is a subclass of `BaseElement`, and `CompositeSystem` and `BaseInteraction` are not
