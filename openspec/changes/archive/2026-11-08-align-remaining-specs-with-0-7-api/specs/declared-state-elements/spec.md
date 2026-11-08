## MODIFIED Requirements

### Requirement: Moved systems keep their results

`sparse_identification`, `NextGenerationReservoirSystem`, and the metabolism family SHALL keep their results after moving
to the primitive.

#### Scenario: Reference values

- **WHEN** the moved systems run on inputs recorded before the move
- **THEN** their results agree with the recorded values within 1e-9 relative tolerance
