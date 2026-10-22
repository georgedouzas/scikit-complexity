# Spec Delta

## MODIFIED Requirements

### Requirement: The form is inferred from the law kinds
Compiling a validated system SHALL derive its form from the laws it lowers, and the form SHALL be an internal property
of the compiled problem rather than part of the system's definition. A system whose nested systems have different law
classes SHALL be split into partitions. Unsupported mixes SHALL be reported by validation before compilation, not by the compiler.
A constraint network's equation, bound, and objective laws SHALL NOT change the form of the system that holds them: a
network alone under static time SHALL give the constraint form, a network beside rate laws under continuous time SHALL
give the ODE form whose drift solves first, and a network beside update laws under discrete time SHALL give the map
form whose step solves first. An enumerate law SHALL give an enumeration form and an estimate law an estimation form,
whose frames are indexed by mode or by iteration instead of time.

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

#### Scenario: Static network
- **WHEN** a system under static time contains only a constraint network
- **THEN** the compiled problem has the constraint form, and each frame solves the network once

#### Scenario: Network with rate laws
- **WHEN** one system under continuous time holds a constraint network and rate laws that read its fluxes
- **THEN** the compiled problem has the ODE form, and its drift solves the network before evaluating the rates

#### Scenario: Network beside an ODE system
- **WHEN** a constraint network sits in a nested system beside a sibling with rate laws
- **THEN** compiling produces a constraint partition and an ODE partition

#### Scenario: Enumerated network
- **WHEN** a system under static time contains only a network declared for enumeration
- **THEN** the compiled problem has the enumeration form, and its frames are indexed by mode

#### Scenario: Labelling experiment
- **WHEN** a system under static time contains only a labelling experiment
- **THEN** the compiled problem has the estimation form, and its frames are indexed by iteration

#### Scenario: Estimation beside dynamics
- **WHEN** a system holds a labelling experiment and a nested system with rate laws
- **THEN** validation reports both paths and says that an estimation runs in a system of its own, before the compiler
  runs
