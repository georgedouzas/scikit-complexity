# Spec Delta

## ADDED Requirements

### Requirement: Components are defined without subclassing
The modelling layer SHALL build element and interaction types from declared states, parameters, roles, and a function
that returns their laws, and such types SHALL behave like subclassed components in groups, cloning, and parameter
introspection.

#### Scenario: One-off growth model
- **WHEN** a user defines an element type with a state `n` and an interaction type with a role and a rate law in a few
  lines, without writing a class
- **THEN** a system of them compiles and simulates exactly like the same model written with classes

#### Scenario: Defined types form groups
- **WHEN** a user creates a group of 100 members from a defined element type
- **THEN** the group compiles once and its parameters appear in `get_params` like those of a subclassed type

### Requirement: States take discrete and periodic domains
States SHALL accept boolean, finite-alphabet, and angle domains besides the real and integer domains, and angle states
SHALL stay within one turn after every step.

#### Scenario: Categorical state
- **WHEN** an element declares a state over the alphabet `susceptible`, `infected`, `recovered`
- **THEN** trajectories return the state's values as the alphabet's labels, and a law that produces a value outside the
  alphabet raises a named error in checked mode

#### Scenario: Angle wraps
- **WHEN** an angle state grows past 2π during a step
- **THEN** its value after the step lies in `[0, 2π)`

### Requirement: Systems declare observables
A system SHALL declare observables, named symbolic expressions of its states and parameters or reductions over its
groups, that are evaluated at every frame and readable by laws of other partitions.

#### Scenario: Order parameter
- **WHEN** a system of phase oscillators declares the observable `order` as the modulus of the mean of `exp(i θ)`
- **THEN** the trajectory returns `order` at every frame with values between 0 and 1

#### Scenario: Reduction of a group
- **WHEN** a system declares the mean of the state `wealth` over a group of agents as an observable
- **THEN** the trajectory returns the mean at every frame, and an ODE in another partition can read it
