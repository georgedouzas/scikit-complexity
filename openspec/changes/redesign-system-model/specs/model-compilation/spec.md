# Spec Delta

## MODIFIED Requirements

### Requirement: The form is inferred from the law kinds
Compiling a validated system SHALL derive its form from the laws it lowers, and the form SHALL be an internal property
of the compiled problem rather than part of the system's definition. A system whose nested systems have different law
classes SHALL be split into partitions. Unsupported mixes SHALL be reported by validation before compilation, not by the compiler.

#### Scenario: Deterministic ecology
- **WHEN** a system contains only rate laws
- **THEN** the compiled problem has the ODE form

#### Scenario: Stochastic epidemic
- **WHEN** a system contains only jump laws
- **THEN** the compiled problem has the jump process form

#### Scenario: Unsupported mix
- **WHEN** one system without children contains both update laws and rate laws
- **THEN** validation reports the mix, names the components by law class, and suggests moving them into nested systems,
  before the compiler runs

#### Scenario: Mix across subsystems
- **WHEN** one nested system contains only update laws and a sibling contains only rate laws
- **THEN** compiling produces a partitioned problem with a discrete map partition and an ODE partition
