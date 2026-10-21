# Spec Delta

## ADDED Requirements

### Requirement: Multiscale network probe models
The library SHALL provide `popularity_similarity_network` and `urban_scaling_interactions`, each with a curated gallery example.

#### Scenario: Hyperbolic degrees
- **WHEN** `popularity_similarity_network` grows 2000 nodes with exponent 2.5
- **THEN** the fitted degree exponent is within 0.2 of 2.5 and clustering stays above 0.3

#### Scenario: Superlinear interactions
- **WHEN** `urban_scaling_interactions` sweeps city size
- **THEN** the number of interactions scales with an exponent above 1, as Ribeiro and Netto (2024) review
