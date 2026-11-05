# Spec Delta

## MODIFIED Requirements

### Requirement: The compatibility table is generated
The documentation SHALL include a table of valid combinations of time, space, laws, domains, solvers, and backends, and of
elements and interactions, generated from the declarations at build time and never written by hand.

#### Scenario: Table follows declarations
- **WHEN** a solver's declared support gains a law kind
- **THEN** the next documentation build lists the solver as valid for that law kind

#### Scenario: Interactions in the table
- **WHEN** an interaction of the core declares what a role needs
- **THEN** the next documentation build lists its roles, the element kinds and states each needs, and the laws it returns

## ADDED Requirements

### Requirement: Interactions declare what their roles need
Every role of an interaction of the core SHALL declare the element kinds it accepts and the states it reads, and a state may
be named by a constructor argument. Interactions SHALL expose this through `get_requirements()`, and elements SHALL expose the
states they carry through `get_states()`.

#### Scenario: Reading requirements
- **WHEN** a user calls `get_requirements()` on an interaction that reads a state named by its `state` argument
- **THEN** the result names the role, the accepted element kinds, and that state

#### Scenario: Every core interaction declares
- **WHEN** the test suite reads the interactions of `skcomplex.core.interactions`
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

#### Scenario: Prebuilt models stay valid
- **WHEN** every model of the catalog is validated
- **THEN** none reports `role-element`
