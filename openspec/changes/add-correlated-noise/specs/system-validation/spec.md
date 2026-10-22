# Spec Delta

## ADDED Requirements

### Requirement: Solvers and backends declare the noise structures they support
The declarations SHALL include the noise structures diagonal, shared, and correlated, and every solver and backend that
runs noise laws SHALL declare which of them it supports. Validation SHALL compute a system's noise structure from its
noise laws and Wiener elements without compiling it, and SHALL report a chosen solver or backend that lacks a structure
the system uses, naming the first law or element that needs it and listing the solvers or backends that support it.

#### Scenario: Milstein with a shared process
- **WHEN** a system in which one process drives two states is validated with the solver `Milstein`
- **THEN** the report holds one error that names the process's first law and suggests `EulerMaruyama` or `Heun`

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
