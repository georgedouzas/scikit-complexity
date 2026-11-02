# Spec Delta

## ADDED Requirements

### Requirement: Networks renormalize across scales
`RenormalizationAnalysis` SHALL coarse-grain a graph, or the network of a trajectory, by its Laplacian (`'laplacian'`)
or by the angles of its nodes (`'geometric'`), and SHALL set the network at every scale with its degree distribution,
degree exponent, and clustering.

#### Scenario: Self-similar network
- **WHEN** a popularity-similarity network is renormalized geometrically three times
- **THEN** its degree exponent stays within 0.2 of the original at every scale

#### Scenario: Laplacian blocks
- **WHEN** two complete graphs of 20 nodes joined by one edge are renormalized by the Laplacian once
- **THEN** the network at the next scale has two nodes joined by one edge
