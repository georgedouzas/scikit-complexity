# Spec Delta

## ADDED Requirements

### Requirement: Network structure analysis model
The library SHALL provide `network_structure_analysis` under `mathematics/networks` (Rm 3.1, 3.3, 3.4, Bt 6.1.1,
6.1.2), a registered factory built on the public core that takes a NetworkX graph, records it as one agent per node,
and reports its centralities, mixing, communities, and triangle z-score as observables. The inventory SHALL mark it as
implemented.

#### Scenario: Measures of a generated network
- **WHEN** `network_structure_analysis` runs on a stochastic block model of 4 blocks of 100 nodes
- **THEN** its trajectory holds the network and observables of clustering, mean path length, assortativity, modularity,
  and the agreement of the detected communities with the blocks above 0.9

#### Scenario: Given graph
- **WHEN** `network_structure_analysis` runs on Zachary's karate club graph from NetworkX
- **THEN** its reported clustering and mean path length equal those of NetworkX within 1e-12, and the community
  detection splits the club into at least two communities with modularity above 0.35

#### Scenario: Gallery example
- **WHEN** a user opens the gallery example of `network_structure_analysis`
- **THEN** it shows the Rerun screenshot of the graph colored by community, and below it the degree distribution and the
  motif z-scores as analysis figures
