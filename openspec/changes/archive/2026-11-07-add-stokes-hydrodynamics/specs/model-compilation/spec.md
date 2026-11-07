## ADDED Requirements

### Requirement: Calls over a whole group couple every member in the Jacobian pattern

The Jacobian pattern of a law whose value is a call over the whole group (`CallSpec(..., vectorized=True)`) SHALL let
the value of every member depend on the arguments of every member, since the call receives the arguments of all
members at once.

#### Scenario: Mean-field call

- **WHEN** a rate law of a group of four members is a call over the whole group that returns the mean of a state
- **THEN** every entry of the Jacobian pattern of that state is set, and finite differences of the drift stay inside it
