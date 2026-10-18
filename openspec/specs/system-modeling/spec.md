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

### Requirement: Element types create groups
An element type SHALL create a group of N identical members with a label, a size, and per-member values given as an
array whose first dimension is N or as one value shared by every member.

#### Scenario: Group with per-member positions
- **WHEN** a user creates a group of 1000 molecules with an array of 1000 positions in a two-dimensional space
- **THEN** the group stores its label, size, and values unmodified, and its position state has shape `(1000, 2)`

#### Scenario: Wrong number of values
- **WHEN** a user gives 999 positions to a group of size 1000
- **THEN** compiling the system raises a named validation error that names the group, the variable, and both sizes

### Requirement: Roles bind to groups
An interaction role SHALL bind to a group as well as to a single element, and the interaction's laws SHALL then apply
to every member of the group.

#### Scenario: Wall law on a whole gas
- **WHEN** a reflecting wall interaction binds its role to a group of 1000 molecules
- **THEN** every molecule is reflected at the walls

### Requirement: Pair interactions sum over member pairs
A pair interaction SHALL be written once for one pair of members and SHALL contribute the sum of its law over every
pair of members within a cutoff distance, inside one group or between two groups.

#### Scenario: Pair force inside a group
- **WHEN** a Lennard-Jones pair interaction with cutoff 2.5 applies to a group
- **THEN** the force on each member equals the sum of the pair forces from the members closer than 2.5

#### Scenario: Newton's third law
- **WHEN** a pair interaction acts inside an isolated group
- **THEN** the total momentum of the group stays constant within solver tolerance

### Requirement: Boundary laws constrain states after each step
An interaction SHALL be able to return boundary laws, which map states to new values after every solver step, combine
with any law kind without changing the inferred form, and describe walls and periodic boxes.

#### Scenario: Reflecting wall in an ODE system
- **WHEN** a particle group with rate laws also has reflecting wall boundaries
- **THEN** the system compiles to the ODE form and no member crosses a wall

#### Scenario: Periodic box
- **WHEN** a member leaves a periodic box through one side
- **THEN** it reappears at the opposite side with the same velocity

### Requirement: Laws express conditions and remainders
The modelling layer SHALL provide symbolic functions for a value chosen by a condition, for the conjunction and
disjunction of conditions, and for the remainder of a division, which laws use like any other expression and which
every backend lowers.

#### Scenario: Reflection by condition
- **WHEN** a boundary sets a velocity to its negative where a position lies outside a box and keeps it elsewhere
- **THEN** a particle that crosses a wall moves back into the box with its velocity reversed

#### Scenario: Periodic remainder
- **WHEN** a boundary sets a position to its remainder by the box size
- **THEN** a particle that leaves through one side enters through the opposite side

#### Scenario: Conditions in LAMMPS
- **WHEN** a pair energy uses a condition on the distance and the LAMMPS backend runs it
- **THEN** the energies match those of the NumPy backend within a relative tolerance of 1e-10
