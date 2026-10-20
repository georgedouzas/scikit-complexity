# Spec Delta

## MODIFIED Requirements

### Requirement: Roles bind to groups
An interaction role SHALL bind to a group, to a single element, or to a set of single elements of one type, and the
interaction's laws SHALL then apply to every member of the group or every element of the set.

#### Scenario: Wall law on a whole gas
- **WHEN** a reflecting wall interaction binds its role to a group of 1000 molecules
- **THEN** every molecule is reflected at the walls

#### Scenario: Role bound to an element set
- **WHEN** an interaction binds its role to the three single elements `unit_0`, `unit_1`, and `unit_2` of one type and
  reads `total(x)` over the role
- **THEN** its law applies to each of the three elements, the total equals the sum of their three states, and the
  elements keep their own state names

#### Scenario: Element set of mixed types
- **WHEN** a role binds to a set of elements of two different types
- **THEN** compiling raises a named error that names the role and both types

### Requirement: Systems declare observables
A system SHALL declare observables, named symbolic expressions of its states and parameters or reductions over its
groups, that are evaluated at every frame, readable at the current state by laws of their own partition when they are
symbolic, and readable by laws of other partitions at synchronisation points.

#### Scenario: Order parameter
- **WHEN** a system of phase oscillators declares the observable `order` as the modulus of the mean of `exp(i θ)`
- **THEN** the trajectory returns `order` at every frame with values between 0 and 1

#### Scenario: Reduction of a group
- **WHEN** a system declares the mean of the state `wealth` over a group of agents as an observable
- **THEN** the trajectory returns the mean at every frame, and an ODE in another partition can read it

#### Scenario: Observable read in its own partition
- **WHEN** a law of an ODE system reads the observable `mean_wealth`, the mean of `W` over the system's group of savers
- **THEN** the system compiles to one ODE partition, and the value the law reads equals the recorded observable at
  every frame within 1e-12
