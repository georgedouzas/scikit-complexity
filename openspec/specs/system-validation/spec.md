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
`compile_system` and `Simulation` SHALL validate before doing any other work, raise one `ValidationError` listing every
error of the report, and emit warnings as named warnings. A valid system SHALL compile without any further compatibility
error.

#### Scenario: Error before compilation
- **WHEN** an invalid system is passed to `Simulation`
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
The documentation SHALL include a table of valid combinations of time, space, laws, domains, solvers, and backends,
generated from the declarations at build time and never written by hand.

#### Scenario: Table follows declarations
- **WHEN** a solver's declared support gains a law kind
- **THEN** the next documentation build lists the solver as valid for that law kind
