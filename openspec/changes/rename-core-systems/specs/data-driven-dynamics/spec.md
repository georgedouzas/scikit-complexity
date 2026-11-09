## MODIFIED Requirements

### Requirement: Surrogates are systems
The result of every surrogate function SHALL hold `system`, a `CompositeSystem` of one element with ordinary laws and a declared
time, continuous for mode decomposition and sparse identification and discrete for reservoirs, and that system SHALL
validate and run through `simulate` like a hand-written system.

#### Scenario: Surrogate validated
- **WHEN** the `system` of the result of `sparse_identification` on a Lorenz trajectory is validated
- **THEN** the report has no error, its time is continuous, and the automatic solver is an ODE solver

#### Scenario: Reservoir under discrete time
- **WHEN** the `system` of the result of `reservoir` is simulated
- **THEN** it declares discrete time, and each step applies the readout once
