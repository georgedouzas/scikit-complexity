# Spec Delta

## ADDED Requirements

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
