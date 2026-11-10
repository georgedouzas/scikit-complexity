## MODIFIED Requirements

### Requirement: Constraints are declared on the members of a group
`SatisfiabilityInteraction`, in `skcomplex.systems.computer_science.neural_networks`, SHALL bind a group whose state is
Boolean and SHALL take a sequence of constraints, each a `ClauseSpec` of literals or a `CardinalitySpec` that at least
or at most a bound of its literals hold, where a literal is a member index from 1, negative for its negation. Its role
SHALL accept only the Boolean domain, so the role domain check of the composite system reports a state that is not
Boolean.

#### Scenario: Perceptron pattern as a constraint
- **WHEN** a binary perceptron of 21 weights `w_i = ±1` must give the output `sigma` for a pattern `xi`
- **THEN** the pattern becomes the `CardinalitySpec` that at least 11 of the literals `w_i = sigma xi_i` hold

#### Scenario: Real state
- **WHEN** a `SatisfiabilityInteraction` binds a group whose state is real
- **THEN** preparing the system, as `simulate` does, raises `RoleElementError` that names the role, the interaction, the
  group, the state, and its domain, and the compiler holds no check of its own for the class
