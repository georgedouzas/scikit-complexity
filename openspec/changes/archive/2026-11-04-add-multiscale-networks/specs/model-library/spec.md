# Spec Delta

## ADDED Requirements

### Requirement: Multiscale network models
The library SHALL provide `popularity_similarity_network` in `mathematics/networks` and `urban_scaling_interactions` in
`social/organizations`, both grown by `TopologyInteraction` with attachment weights of the hyperbolic distance, each
with a curated gallery example and an inventory entry tagged with the system it builds.

#### Scenario: Hyperbolic degrees
- **WHEN** `popularity_similarity_network` grows 2000 nodes with degree exponent 2.5, two edges, and temperature 0.2
- **THEN** the fitted degree exponent is within 0.06 of the measured 2.70, as design.md records against the 2.5 of the
  source, and the clustering stays above 0.3

#### Scenario: Superlinear interactions
- **WHEN** `urban_scaling_interactions` is swept over city size and its interactions are fitted by
  `ScalingAnalysis(kind='allometric')`
- **THEN** the exponent of the number of interactions is above 1, as Ribeiro and Netto (2024) review
