## ADDED Requirements

### Requirement: Systems nest as components
A system SHALL hold nested systems among its components, and a nested system SHALL itself be a component, so systems
nest to any depth. An interaction SHALL refer to a component of a nested system by a slash path relative to the system
that holds the interaction. Two direct children of a system with the same label SHALL be reported by validation.

#### Scenario: Interaction across nested systems
- **WHEN** a parent system holds a predation interaction between `forest/foxes` and `meadow/rabbits`
- **THEN** the compiled system couples the fox count of the forest system to the rabbit count of the meadow system

#### Scenario: Duplicate labels
- **WHEN** two direct children of a system share a label
- **THEN** validation reports one issue whose path is the system and whose message names the duplicated label

#### Scenario: A system is a component
- **WHEN** a user checks whether a `CompositeSystem` instance is an instance of `BaseComponent`
- **THEN** the check is true, the instance is an instance of `BaseSystem`, and it is not an instance of `BaseElement`

### Requirement: A system is elements, interactions, space, and time
A system SHALL be defined by its label, its elements, its interactions, its nested systems, its space, and its time,
and by nothing else. A system SHALL accept no argument about how it runs: no solver, backend, form, noise convention,
delay history, activation schedule outside its time, or observables.

#### Scenario: Four-component system
- **WHEN** a user builds `CompositeSystem('ecology', elements=[rabbits, foxes], interactions=[predation],
  space=None, time=ContinuousTime())`
- **THEN** the system compiles and runs, and its parameters list only the parameters of its elements and interactions

#### Scenario: Execution arguments are rejected
- **WHEN** a user passes `noise=`, `history=`, `phases=`, or `observables=` to `CompositeSystem`
- **THEN** a `TypeError` is raised as for any unknown argument

## RENAMED Requirements

- FROM: `### Requirement: Parts share two main public methods`
- TO: `### Requirement: Components share two main public methods`

## REMOVED Requirements

### Requirement: Systems hold nested systems
**Reason**: Its scenario said that a system is not a component, which no longer holds since `BaseSystem` derives from `BaseComponent`.
**Migration**: See the requirement "Systems nest as components", which replaces it.

### Requirement: A system is components, interactions, space, and time
**Reason**: It used component in the sense of an element from before `BaseComponent` was renamed `BaseElement`, and a scenario name said part.
**Migration**: See the requirement "A system is elements, interactions, space, and time", which replaces it.

## MODIFIED Requirements

### Requirement: Noise processes are not elements
`WienerProcessSpec` SHALL derive from no component base, and an interaction SHALL bind it through a role that takes
noise processes.

#### Scenario: Correlated noise
- **WHEN** an interaction binds a `WienerProcessSpec` with covariance `C` and two noise laws draw its processes
- **THEN** the increments over a step `h` have the covariance `h C`, and `WienerProcessSpec` is not a subclass of
  `BaseComponent`

### Requirement: A system is defined by elements, interactions, and laws
A system SHALL be defined by its elements, its interactions, and the laws they return, and SHALL declare no other
kind of component. A value that only names an expression SHALL be written in the laws, and what a run records or
checks SHALL be an argument of the run.

#### Scenario: Auxiliary as an expression
- **WHEN** a law computes `leverage = s.debt / s.equity` and uses it in a rate, and the run records an observable of
  the same expression
- **THEN** the trajectory holds `leverage`, and the system has no quantity among its components

### Requirement: Components share two main public methods
Elements, interactions, laws, spaces, times, and systems SHALL derive from `BaseComponent`, which defines `label`,
`prepare(context)`, and `run(input)`. Elements, interactions, and systems SHALL return `self` from `prepare`, and
their laws from `run(s)`, and a system SHALL return the laws of its children. A law SHALL return a prepared copy, and
SHALL evaluate its contribution from `run(values)`.

#### Scenario: System yields laws
- **WHEN** a system of a population element and a growth interaction is prepared and run with a namespace
- **THEN** it returns the laws of the element and of the interaction, in order

#### Scenario: Law run alone
- **WHEN** a user prepares `RateLaw(s.n, r * s.n)` against a namespace with `n` and `r`, and runs it at `n=10`,
  `r=0.5`
- **THEN** the result is `5.0`, its `targets_` holds `n`, and its `reads_` holds `n` and `r`
