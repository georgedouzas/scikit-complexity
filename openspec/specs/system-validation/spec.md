# system-validation Specification

## Purpose
Validate a system, and its chosen solver and backend, before compilation. Every primitive, solver, and backend declares
what it requires and supports, and the validator reports every invalid combination at once, each with its path, reason,
and fix.

## Requirements

### Requirement: Primitives, solvers, and backends declare requirements and support
Every law kind, component kind, space kind, and time kind SHALL declare what it requires. Every solver and backend SHALL
declare what it supports. The declarations SHALL cover:
- time kinds;
- space kinds;
- state domains;
- role arities and bindings;
- law kinds;
- fixed or variable groups.

A test SHALL fail when a registered primitive, solver, or backend has no declaration.

#### Scenario: Missing declaration
- **WHEN** a contributor registers a new solver without a support declaration
- **THEN** the declaration test fails and names the solver

#### Scenario: Plug-in backend extends validity
- **WHEN** a new backend declaring support for discrete time with variable groups is registered
- **THEN** systems it supports validate with that backend, without any change to the validator

### Requirement: One pass reports every problem
`validate(system, solver=None, backend=None)` SHALL return a report of every problem in the system, each with:
- a stable code;
- the slash path of the component or law;
- a message;
- a fix;
- a severity.

It SHALL check:
- structure: labels, paths, bound roles, and arity;
- components with the space;
- laws with the time;
- laws with state domains;
- laws with each other within a partition;
- the system with the solver;
- the solver with the backend;
- the availability of optional backends.

#### Scenario: Several errors at once
- **WHEN** a system has an unbound role, a rate law under discrete time, and a vector state without a space
- **THEN** the report holds three errors, each with its own path and code

#### Scenario: Law with the wrong time
- **WHEN** a rate law is placed in a system declared with discrete time
- **THEN** the report names the law's path, says that rates need continuous time, and suggests either continuous time or
  an update law

#### Scenario: Law with the wrong domain
- **WHEN** a rate law targets a categorical state
- **THEN** the report names the state and the law and suggests a jump or transition law

#### Scenario: Unsupported solver
- **WHEN** a system with a variable group is validated with the solver `'rk4'`
- **THEN** the report says that the solver does not support variable groups and lists the solvers that do

#### Scenario: Unavailable backend
- **WHEN** a system is validated with the LAMMPS backend and LAMMPS is not installed
- **THEN** the report holds one error whose fix names the installation extra

### Requirement: Compilation and simulation validate first
A compiler and `simulate` SHALL validate before doing any other work, raise one `ValidationError` listing
every error of the report, and emit warnings as named warnings. A valid system SHALL compile without any further
compatibility error.

#### Scenario: Error before compilation
- **WHEN** an invalid system is passed to `simulate`
- **THEN** a `ValidationError` is raised before any kernel is generated, and its message lists every error with its
  path and fix

### Requirement: Mixed law classes suggest partitions
When a system without children mixes law classes that no single solver supports, validation SHALL report it, name the
components by class, and suggest the nested systems that would run as coupled partitions. A system whose nested systems
each have one supported class SHALL validate as a coupled system.

#### Scenario: Rates and updates in one system
- **WHEN** one system without children holds rate laws on one element and update laws on another
- **THEN** the report names both elements and suggests moving each into its own nested system

### Requirement: The compatibility table is generated
The documentation SHALL include a table of valid combinations of time, space, laws, domains, solvers, and backends, and of
elements and interactions, generated from the declarations at build time and never written by hand.

#### Scenario: Table follows declarations
- **WHEN** a solver's declared support gains a law kind
- **THEN** the next documentation build lists the solver as valid for that law kind

#### Scenario: Interactions in the table
- **WHEN** an interaction of the core declares what a role needs
- **THEN** the next documentation build lists its roles, the element kinds and states each needs, and the laws it returns

### Requirement: Solvers and backends declare the noise structures they support
The declarations SHALL include the noise structures diagonal, shared, and correlated, and every solver and backend that
runs noise laws SHALL declare which of them it supports. Validation SHALL compute a system's noise structure from its
noise laws and Wiener elements without compiling it, and SHALL report a chosen solver or backend that lacks a structure
the system uses, naming the first law or element that needs it and listing the solvers or backends that support it.

#### Scenario: Milstein with a shared process
- **WHEN** a system in which one process drives two states is validated with the solver `Milstein`
- **THEN** the report holds one error that names the process's first law and suggests `EulerMaruyamaSolver` or `Heun`

#### Scenario: Automatic solver for correlated noise
- **WHEN** a system with a correlated Wiener element is simulated with the solver `'auto'`
- **THEN** a solver that declares correlated noise is chosen

#### Scenario: Noise support in the compatibility table
- **WHEN** the documentation is built
- **THEN** the generated compatibility table lists the noise structures each solver and backend supports

### Requirement: Wiener elements are validated
A Wiener element SHALL require continuous time. Validation SHALL report a Wiener element under another time, a
covariance that is not square, symmetric, or positive semidefinite, and a process handle used anywhere but as the
process of a noise law, each with the element's or law's path and a fix.

#### Scenario: Invalid covariance
- **WHEN** a Wiener element of dimension 2 has covariance `[[1, 2], [2, 1]]`
- **THEN** the report holds an error that names the element and its negative eigenvalue

#### Scenario: Process used as a value
- **WHEN** a rate law uses a Wiener element's process handle inside its expression
- **THEN** the report holds an error that names the law and says the handle is only a noise law's process

#### Scenario: Wiener element under discrete time
- **WHEN** a Wiener element is part of a system declared with discrete time
- **THEN** the report holds an error that names the element and suggests continuous time

### Requirement: Invalid spaces are reported as issues
Validation SHALL report every space whose own check fails, such as a lattice with a wrong shape, boundary, or
geometry, as an issue with a stable code, the path of the system that holds the space, the reason, and a fix, and SHALL
never skip a space whose check fails.

#### Scenario: Lattice with a bad boundary
- **WHEN** a system holds a lattice space whose boundary is not one of the supported boundaries
- **THEN** the report holds an error with the code `space-invalid` that names the boundary and lists the supported ones

#### Scenario: Invalid space among other errors
- **WHEN** a system has an invalid lattice and an unbound role
- **THEN** the report holds both errors, and compiling the system raises the validation error instead of an error from
  inside the compiler

### Requirement: Interactions declare what their roles need
Every role of an interaction of the core SHALL declare the element kinds it accepts and the states it reads, and a state may
be named by a constructor argument. Interactions SHALL expose this through `get_requirements()`, and elements SHALL expose the
states they carry through `get_states()`.

#### Scenario: Reading requirements
- **WHEN** a user calls `get_requirements()` on an interaction that reads a state named by its `state` argument
- **THEN** the result names the role, the accepted element kinds, and that state

#### Scenario: Every core interaction declares
- **WHEN** the test suite reads the interactions of `skcomplex.core.components.interactions`
- **THEN** every role of each declares its requirements

### Requirement: A role filled by an unsuitable element is reported
Validation SHALL report a role whose element is of a kind the role does not accept, or lacks a state the role reads, as the
issue `role-element`, with its path, the state or the kind, and a fix, among the other problems of the same report.

#### Scenario: Missing state
- **WHEN** a system binds an interaction whose laws read `s.target.n` to an element without `n`, declared or not
- **THEN** the report names the interaction, the role, and `n`, and no `AttributeError` reaches the user

#### Scenario: Other errors are kept
- **WHEN** a law of an interaction raises an error that is not a missing attribute of a role
- **THEN** that error reaches the user unchanged

#### Scenario: Prebuilt systems stay valid
- **WHEN** every system of the catalog is validated
- **THEN** none reports `role-element`

### Requirement: A value outside its domain is reported as outside it

A value of a state that lies outside the domain of the state SHALL be reported with a `ValidationError` that names the
state, the value, and the domain. The error SHALL say that the value has a nonzero imaginary part only when the value
is complex and not real.

#### Scenario: A negative value of a nonnegative state

- **WHEN** a state with the domain `[0, +oo)` is given the value `-1.0` and the system is simulated
- **THEN** a `ValidationError` says that `-1.0` is not in the domain `[0, +oo)`, and it does not mention an imaginary part

#### Scenario: A fractional value of an integer state

- **WHEN** a state whose domain is the nonnegative integers is given `1.5`
- **THEN** the `ValidationError` says that `1.5` is not in that domain

#### Scenario: A complex value of a real state

- **WHEN** a real state is given `1 + 2j`
- **THEN** the `ValidationError` still says that the value has a nonzero imaginary part

### Requirement: An unknown variant lists only the accepted values

When an argument of a component takes one of several named values, or a spec a subclass handles, an unknown value
SHALL raise a `ValidationError` that names the component, the argument, the value given, and the accepted names. The
message SHALL describe an accepted spec in words and SHALL NOT print the name of a base class.

#### Scenario: An unknown mode of a map interaction

- **WHEN** `MapInteraction(..., mode='add')` is prepared
- **THEN** the `ValidationError` names `'add'` and the accepted names `'replace'` and `'increment'`, and its text holds no `BaseSpec`

### Requirement: The order of a law is checked against its state in every system

A rate law or a noise law that gives a derivative of an order other than the order of its state SHALL raise a
`ValidationError` when the system is compiled, whatever the orders of the other states of the system.

#### Scenario: A second-order law on a first-order state, alone

- **WHEN** a system whose only state has order 1 holds `RateLaw(x, -x, order=2)` and is simulated
- **THEN** a `ValidationError` says that the law gives the derivative of order 2 but the state has order 1

### Requirement: The ends of a transfer print by their fields

The source and the sink of a transfer law SHALL print as their class name with their fields, and two ends with equal
fields SHALL be equal, so errors that name them never show a memory address.

#### Scenario: An error that names a sink

- **WHEN** a transfer law is given a sink where a source belongs and is prepared
- **THEN** the error shows `Sink(label='environment')`, not an address in memory

### Requirement: Bad spaces raise library errors

A space SHALL check its arguments when it is prepared and raise a library error that names the space and the
argument at fault. No raw `IndexError`, `KeyError`, `ValueError`, or NumPy error SHALL escape from preparing a space or
from reading its neighbours.

#### Scenario: Mesh points of the wrong shape

- **WHEN** a `MeshSpace` is prepared with points that are not two or three rows of coordinates
- **THEN** a `ValidationError` names `points` and the shape it expects

#### Scenario: Mesh cells that name a missing point

- **WHEN** a `MeshSpace` is prepared with a cell index at or beyond the number of points, or with an empty or one-dimensional array of cells
- **THEN** a `ValidationError` names `cells` and says what is wrong

#### Scenario: A neighbourhood names a missing layer

- **WHEN** an adjacency neighbourhood names the layer `'work'` of a network space that has no such layer and the system is simulated
- **THEN** a `SpaceError` says that the network has no layer `'work'` and lists its layers

### Requirement: A law with no form names its class

A law whose class has no form the engine knows SHALL raise an `UnregisteredLawError` that names the class, before
any check of mixed law classes.

#### Scenario: A law class with no form

- **WHEN** a system holds a law of a class the engine has no form for and is compiled
- **THEN** an `UnregisteredLawError` names that class

### Requirement: Every error class is raised

Every public error class SHALL be raised somewhere in the library. A class that no part can raise SHALL be removed.

#### Scenario: The argument error is gone

- **WHEN** a user imports `ArgumentError` from `skcomplex.errors`
- **THEN** the import fails, and the ridesharing system reports a value of the wrong kind with a `ValidationError`
