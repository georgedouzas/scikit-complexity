# Spec Delta

## MODIFIED Requirements

### Requirement: The form is inferred from the law kinds
Compiling a system SHALL infer its form from the law kinds it contains: rate laws alone give an ODE, rate and noise
laws give an SDE, jump laws alone give a jump process, and update laws alone give a discrete map. A system whose
subsystems have different forms SHALL be split into partitions, and only a single subsystem that mixes law kinds of no
one form SHALL be rejected.

#### Scenario: Deterministic ecology
- **WHEN** a system contains only rate laws
- **THEN** the compiled problem has the ODE form

#### Scenario: Stochastic epidemic
- **WHEN** a system contains only jump laws
- **THEN** the compiled problem has the jump process form

#### Scenario: Unsupported mix
- **WHEN** one subsystem without children contains both update laws and rate laws
- **THEN** compiling raises a named error that names the subsystem, lists the law kinds found and the supported
  combinations, and suggests moving the laws into separate subsystems

#### Scenario: Mix across subsystems
- **WHEN** one subsystem contains only update laws and a sibling subsystem contains only rate laws
- **THEN** compiling produces a partitioned problem with a discrete map partition and an ODE partition
