# Spec Delta

## ADDED Requirements

### Requirement: Dynamics analysis probe models
The library SHALL provide `kinouchi_copelli`, `nonreciprocal_cahn_hilliard`, `nonreciprocal_vicsek`, and `tipping_cascade_network`, each with a curated gallery example.

#### Scenario: Maximal dynamic range
- **WHEN** `kinouchi_copelli` sweeps the branching ratio
- **THEN** the dynamic range peaks at a branching ratio of 1, as Kinouchi and Copelli (2006) found

#### Scenario: Traveling patterns
- **WHEN** `nonreciprocal_cahn_hilliard` runs with nonreciprocal coupling above threshold
- **THEN** its demixed domains travel, as Brauns and Marchetti (2023) predict
