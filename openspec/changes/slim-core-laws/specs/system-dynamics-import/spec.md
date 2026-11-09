## MODIFIED Requirements

### Requirement: Built-in functions map to the law utilities
Lookups SHALL become `series` of their input through their points, conditionals `where`, the input signals `step`,
`ramp`, `pulse`, and `pulse_train`, delays and smoothing `distributed_delay` of their order, with `conserved=False` for
smoothing, or `delayed`, `MIN` and `MAX` `minimum` and `maximum`, and the random functions draws. A start time other
than zero SHALL shift `Time` in every expression.

#### Scenario: Third-order delay
- **WHEN** a system with `DELAY3` of a step input is translated and run
- **THEN** its output is within 1e-6 of PySD's run of the same file

#### Scenario: Smoothing
- **WHEN** a system with `SMOOTH` of a step input is translated
- **THEN** the written module calls `distributed_delay` with `conserved=False`, and its run is within 1e-6 of PySD's
