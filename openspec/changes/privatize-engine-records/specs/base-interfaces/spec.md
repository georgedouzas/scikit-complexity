## MODIFIED Requirements

### Requirement: Values carry fields and validate when prepared
Every public value of the library outside the systems SHALL be a frozen spec that derives from `BaseSpec`, SHALL be
named with the suffix `Spec`, and SHALL live in the package the base-interfaces rules give it. A spec SHALL hold only
its fields, the hook `_check`, and readers of its own fields. Its owner SHALL call `_check` when it reads the spec, and
preparing a system SHALL check every law and spec it holds.

#### Scenario: Invalid law
- **WHEN** a component returns a `RateLaw` whose value is a NumPy array, and the system is prepared
- **THEN** the law is created without error, and preparing raises an error that names the component and the law

#### Scenario: Terms by value
- **WHEN** a law kind is pickled and loaded in another process
- **THEN** it equals the registered kind with `==`

## ADDED Requirements

### Requirement: Internal records are private and few
Every record that only the library builds and reads SHALL be a private frozen dataclass with no base and no `_check`,
defined in the module that builds or reads it. A record with three fields or fewer, no methods and one reader SHALL
be plain arguments or a tuple instead.

#### Scenario: No trivial records
- **WHEN** a test lists the private dataclasses of `skcomplex.core`
- **THEN** none has three fields or fewer, no methods and a single reader

#### Scenario: No checks on records
- **WHEN** a test lists the private dataclasses of `skcomplex.core`
- **THEN** none derives from `BaseSpec` or defines `_check`

### Requirement: Engine records are private except the solver contract
The public names of `skcomplex.core.engine` SHALL include no record except `BaseProblemSpec`, the problem forms that
solvers receive and the records those forms expose as fields. The `problems` package SHALL hold one module per
problem kind.

#### Scenario: Public names of the engine
- **WHEN** a test lists the public names of every package of `skcomplex.core.engine` that end with `Spec`
- **THEN** each is `BaseProblemSpec`, a problem form a solver receives, or a record such a form holds as a field

#### Scenario: A solver written outside the library
- **WHEN** a solver written against `BaseSolver` and the public problem forms alone is prepared and run
- **THEN** it reads every field it needs without importing a private name
