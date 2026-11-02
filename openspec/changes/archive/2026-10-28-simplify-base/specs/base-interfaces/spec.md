## MODIFIED Requirements

### Requirement: Values carry fields and a check
Laws, domains, problems, and compiled records SHALL be frozen dataclasses, and terms strings of their value, compared
by value. Laws SHALL be parts whose `check`, `prepare`, and `run` return checked and prepared copies or values, and
the other values SHALL carry only `check()`. Validation SHALL call `check()` on every law of a system.

#### Scenario: Invalid law
- **WHEN** a component returns a `RateLaw` whose value is a NumPy array, and the system is validated
- **THEN** the law is created without error, and validation raises an error that names the component and the law

#### Scenario: Terms by value
- **WHEN** a law kind is pickled and loaded in another process
- **THEN** it equals the registered kind with `==`
