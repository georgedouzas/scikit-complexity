## ADDED Requirements

### Requirement: General systems are recipes of several fields
A general system SHALL serve systems of two fields, or be able to, and SHALL be named after its mathematics, never
after a field quantity. As a recipe of several fields, by step 3 of the placement rule, it SHALL live in a concept
subpackage of `skcomplex.systems.general`, here `skcomplex.systems.general.dynamics`, and no core module SHALL import
it.

#### Scenario: Import a general system
- **WHEN** a user imports `DynamicalSystem`, `NetworkDynamicalSystem`, `ReactionDiffusionSystem`, or
  `StockFlowSystem` from `skcomplex.systems.general.dynamics`
- **THEN** the import succeeds, and the class is a subclass of `BaseSystem` from `skcomplex.core.components.systems`

#### Scenario: The core holds none of them
- **WHEN** a user imports `DynamicalSystem` from `skcomplex.core.components.systems`
- **THEN** the import fails with `ImportError`, and no module under `src/skcomplex/core` imports
  `skcomplex.systems.general.dynamics`

#### Scenario: Shape of one field
- **WHEN** a system shape appears in one field only, such as the Boolean networks of systems biology
- **THEN** it stays in the `common` package of that field, and no general system holds it

#### Scenario: Shape covered by one core component
- **WHEN** a system shape is one core interaction on a group, such as a cellular automaton, a `CompositeSystem` with a
  `LocalRuleInteraction`
- **THEN** no general system duplicates it

## REMOVED Requirements

### Requirement: General systems are recipes that several fields share
**Reason**: Its text and a scenario name called the building blocks of a system parts, and a scenario cannot be renamed in place.
**Migration**: See the requirement "General systems are recipes of several fields", which replaces it.

## MODIFIED Requirements

### Requirement: Dynamical systems run in continuous or discrete time
`DynamicalSystem` SHALL read its equations as rates in the time `'continuous'` and as the next values in the time
`'discrete'`. In continuous time it SHALL take optional noise amplitudes by state, read in the Itô or Stratonovich
convention. Its tags SHALL follow from the components it builds, with no class chosen by the form of the equations.

#### Scenario: Map of the logistic family
- **WHEN** a user simulates `DynamicalSystem(states={'x': 0.2}, equations={'x': lambda s: s.target.r * s.target.x *
  (1 - s.target.x)}, parameters={'r': 2.0}, time='discrete')` for 30 steps
- **THEN** the last value of `x` is `0.5` within 1e-9, and the tags name the law `'update'` and a discrete time

#### Scenario: Ornstein-Uhlenbeck process
- **WHEN** a user gives a rate `-x` and the noise amplitude `0.3` for `x` and reads the tags
- **THEN** the tags name the laws `'rate'` and `'noise'` and the randomness `'stochastic'`

#### Scenario: Noise in discrete time
- **WHEN** a user prepares a `DynamicalSystem` in discrete time with a noise amplitude
- **THEN** preparing raises a `ValidationError` that says noise amplitudes need continuous time
