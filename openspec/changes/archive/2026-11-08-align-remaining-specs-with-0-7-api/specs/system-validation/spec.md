## MODIFIED Requirements

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
