# Spec Delta

## MODIFIED Requirements

### Requirement: Interactions bind elements through roles
An interaction type SHALL declare named roles and parameters, and an interaction instance SHALL bind each role to an
element label of its system, or a variadic role to a sequence of element paths and group member references.

#### Scenario: Role refers to an unknown element
- **WHEN** a system contains an interaction whose role names a label that no element in reach has
- **THEN** compiling the system raises a named error that names the interaction, the role, and the label

#### Scenario: Variadic entry refers to a missing member
- **WHEN** a variadic role holds the member reference `nodes[12]` and the group `nodes` has 10 members
- **THEN** compiling the system raises a named role error that names the interaction, the role, the entry, and the
  group size

### Requirement: Element types create groups
An element type SHALL create a group of N identical members with a label and a size given by position, and per-member
values given by keyword as an array whose first dimension is N or as one value shared by every member, so a state of
any name that is not reserved takes values.

#### Scenario: Group with per-member positions
- **WHEN** a user creates a group of 1000 molecules with an array of 1000 positions in a two-dimensional space
- **THEN** the group stores its label, size, and values unmodified, and its position state has shape `(1000, 2)`

#### Scenario: Wrong number of values
- **WHEN** a user gives 999 positions to a group of size 1000
- **THEN** compiling the system raises a named validation error that names the group, the variable, and both sizes

#### Scenario: State named size
- **WHEN** an element type declares a state `size` and a user creates `Cell.group('cells', 3, size=[1.0, 2.0, 3.0])`
- **THEN** the group has 3 members and its `size` state holds 1, 2, and 3

### Requirement: Roles bind to groups
An interaction role SHALL bind to a group as well as to a single element, and the interaction's laws SHALL then apply
to every member of the group. Roles of one interaction SHALL bind several groups of equal size, read member by member.

#### Scenario: Wall law on a whole gas
- **WHEN** a reflecting wall interaction binds its role to a group of 1000 molecules
- **THEN** every molecule is reflected at the walls

#### Scenario: Two groups member by member
- **WHEN** an interaction adds `s.source.n` to the rate of `s.sink.n`, with both roles bound to groups of 5 members
- **THEN** the rate of member `i` of the sink equals the count of member `i` of the source for every `i`

#### Scenario: Groups of different sizes
- **WHEN** an interaction with symbolic laws binds its roles to groups of 5 and 7 members
- **THEN** compiling the system raises a named role error that names both groups and both sizes

## ADDED Requirements

### Requirement: Variadic roles bind sequences of elements
A role declared variadic SHALL bind any number of element paths or group member references in order, laws SHALL read
it as a sequence of namespaces of that length, and a parameter declared per entry of the role SHALL hold one value per
entry, each addressed by path and index.

#### Scenario: Sum over a variable number of inputs
- **WHEN** an interaction whose variadic role binds the elements `a`, `b`, and `c` updates `target.x` to the sum of
  `x` over its inputs
- **THEN** one step sets `target.x` to `a.x + b.x + c.x`

#### Scenario: Member reference
- **WHEN** a variadic role binds the entries `hub` and `nodes[3]`
- **THEN** the second namespace reads the states of member 3 of the group `nodes`

#### Scenario: Per-entry parameter with the wrong length
- **WHEN** a variadic role binds 3 entries and its per-entry weights hold 2 values
- **THEN** compiling the system raises a named validation error that names the interaction, the parameter, and both
  lengths

#### Scenario: Live per-entry parameter
- **WHEN** a user updates the parameter `regulation.weights[1]` of a running simulation
- **THEN** the next step uses the new weight for the second entry only

### Requirement: Laws read group members at an offset
The modelling layer SHALL provide a symbolic function that reads an expression of a group's member variables at the
member a given number of places along the group order, with periodic, reflecting, or fixed ends, and SHALL compile it
into the group's kernel on every backend.

#### Scenario: Ring of members
- **WHEN** a law of a group of 4 members reads `shift_member(s.cars.x, 1)` with periodic ends
- **THEN** member 3 reads the value of member 0, and every other member `i` reads member `i + 1`

#### Scenario: Fixed end
- **WHEN** a law of a group of 4 members reads `shift_member(s.cars.x, 1, boundary='fixed', fill=-1.0)`
- **THEN** member 3 reads -1

#### Scenario: Same result on both backends
- **WHEN** a ring of 1000 members with offset reads runs on the NumPy and the Numba backends
- **THEN** the trajectories agree within a relative tolerance of 1e-10

#### Scenario: Offset of a single element
- **WHEN** a law applies `shift_member` to a state of a single element
- **THEN** compiling the system raises a named validation error that names the component and the state

### Requirement: Reserved variable names are rejected
Defining an element or agent type SHALL reject a state or parameter whose name is reserved by the component or by the
views rules read it through, with a named label error raised when the type is created.

#### Scenario: State named visual
- **WHEN** a user defines an element type with a state named `visual`
- **THEN** a named label error is raised that names the state and lists the reserved names

#### Scenario: Agent state named location
- **WHEN** a user defines an agent type with a state named `location` or `alive`
- **THEN** a named label error is raised that names the state and lists the reserved names
