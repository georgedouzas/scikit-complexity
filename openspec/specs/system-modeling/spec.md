# system-modeling Specification

## Purpose

Lets a user describe a model from any science as elements that carry state, interactions that change that state
through laws, a space the state lives in, and systems that nest and compose.

## Requirements

### Requirement: Elements declare typed state and parameters
An element type SHALL declare its state variables and parameters with a domain, and an element instance SHALL take
its label, initial state values, and parameter values as constructor arguments stored unmodified.

#### Scenario: Element stores its arguments
- **WHEN** a user creates a population element with label `rabbits` and initial count 40
- **THEN** the element exposes `label == 'rabbits'` and `n == 40`

#### Scenario: Invalid value is rejected at compile time
- **WHEN** a system contains an element whose positive parameter is set to a negative number and it is compiled
- **THEN** a named validation error is raised that names the element, the variable, and the domain

### Requirement: Interactions bind elements through roles
An interaction type SHALL declare named roles and parameters, and an interaction instance SHALL bind each role to an
element label of its system.

#### Scenario: Role refers to an unknown element
- **WHEN** a system contains an interaction whose role names a label that no element in reach has
- **THEN** compiling the system raises a named error that names the interaction, the role, and the label

### Requirement: Laws describe how state changes
An interaction or element SHALL describe its effect as a list of laws of the kinds rate, noise, jump, and update, each
law targeting state variables, and each law SHALL accept a symbolic expression or a Python callable.

#### Scenario: Symbolic rate law
- **WHEN** an interaction returns a rate law on `prey.n` with the expression `-a * predator.n * prey.n`
- **THEN** the compiled system includes that term in the time derivative of the prey count

#### Scenario: Callable rate law
- **WHEN** an interaction returns a rate law whose value is a Python callable of time and current values
- **THEN** the simulation evaluates the callable at each solver evaluation and adds its result to the derivative

### Requirement: Rate and noise contributions add up
Rate and noise laws on the same state variable SHALL combine by summation, so adding an interaction adds a term
without changing other interactions.

#### Scenario: Two forces on one particle
- **WHEN** a particle is subject to a constant force and a harmonic potential
- **THEN** its velocity derivative equals the sum of both contributions divided by its mass

### Requirement: Systems nest and compose
A system SHALL contain elements, interactions, and child systems, and an interaction SHALL refer to an element of a
child system by a slash-separated path relative to the system that holds the interaction.

#### Scenario: Interaction across subsystems
- **WHEN** a parent system holds a predation interaction between `forest/foxes` and `meadow/rabbits`
- **THEN** the compiled system couples the fox count of the forest subsystem to the rabbit count of the meadow

#### Scenario: Duplicate labels are rejected
- **WHEN** two direct children of a system share a label
- **THEN** compiling the system raises a named error that names the duplicated label

### Requirement: Spaces give vector state its dimension
A system SHALL hold an optional space, and a vector state variable SHALL take its dimension from the space of the
system that holds the element.

#### Scenario: Particle in a plane
- **WHEN** a particle is placed in a system whose Euclidean space has two dimensions
- **THEN** its position and velocity each have two components

#### Scenario: Vector state without a space
- **WHEN** an element with a vector state variable is compiled in a system with no space
- **THEN** a named error is raised that tells the user to give the system a space
