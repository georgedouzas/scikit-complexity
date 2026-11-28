## MODIFIED Requirements

### Requirement: Values carry fields and validate when prepared
A public value of the library outside the systems SHALL be a class only when it has behaviour of its own, is a law,
or is a result returned to users. A variant SHALL be a string option of its owner, a group of values SHALL be plain
parameters of its owner or a tuple or a dict, and the owner SHALL check them in `_check` when it is prepared. A
remaining value SHALL be a frozen spec that derives from `BaseSpec` and is named with the suffix `Spec`.

#### Scenario: Invalid law
- **WHEN** a component returns a `RateLaw` whose value is a NumPy array, and the system is prepared
- **THEN** the law is created without error, and preparing raises an error that names the component and the law

#### Scenario: Terms by value
- **WHEN** a law kind is pickled and loaded in another process
- **THEN** it equals the registered kind with `==`

#### Scenario: Unknown option
- **WHEN** an owner that took a variant spec is given an unknown string for that option and is prepared
- **THEN** a `ValidationError` names the owner, the option and its allowed values

## ADDED Requirements

### Requirement: Public specs have behaviour
Every public class of `skcomplex` whose name ends with `Spec` SHALL define a public method or property of its own
besides reading its fields, or be a law, and every value a trajectory returns to users SHALL be a frozen `*Result`.

#### Scenario: No record specs
- **WHEN** a test lists the public classes of `skcomplex` that end with `Spec`
- **THEN** each defines a public method or property of its own or derives from `BaseLaw`

#### Scenario: Trajectory results
- **WHEN** `get_network` of a run of `IsingSystem(5)` is called
- **THEN** it returns a `NetworkSnapshotResult` with the nodes and the edges
