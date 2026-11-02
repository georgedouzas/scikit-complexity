# Spec Delta

## ADDED Requirements

### Requirement: Groups compile independently of their size
Compiling a system SHALL build one set of numeric functions per group and law, so the compile time of a group does not
grow with its size.

#### Scenario: Large group compiles fast
- **WHEN** a group of 100000 molecules with reflecting walls is compiled
- **THEN** compiling takes no longer than compiling the same system with a group of 10 molecules, within a factor of 2

### Requirement: Group variables are addressed by group path
The compiled problem SHALL name each group variable by the group path and variable name, with a leading dimension of
the group size.

#### Scenario: Name of a group state
- **WHEN** the group `gas` of 500 molecules in a two-dimensional space has the state `x`
- **THEN** the problem has a state named `gas.x` with shape `(500, 2)`
