# Spec Delta

## ADDED Requirements

### Requirement: Structural identifiability before data
`identifiability(method='structural')` SHALL decide for every parameter whether it is locally structurally
identifiable from the observed paths, by the rank of the Jacobian of the Taylor coefficients of the outputs in the
parameters, computed on the symbolic laws, with no data values.

#### Scenario: Sum of two rates
- **WHEN** `x' = -(k1 + k2) x` is tested with `x` observed
- **THEN** both `k1` and `k2` are `'not identifiable'` and the rank is 1

#### Scenario: Predator and prey
- **WHEN** a Lotka-Volterra system is tested with both populations observed
- **THEN** all four rates are `'identifiable'`

#### Scenario: Laws that are not symbolic
- **WHEN** the system holds a law with a callable or a learned function
- **THEN** it raises `ValidationError` that names the law and says the structural test needs symbolic laws
