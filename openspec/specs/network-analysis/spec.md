# network-analysis Specification

## Purpose
Measures the structure of static and evolving networks from any source, a network space, agent contacts, a lattice, or
a given graph: centralities, mixing, communities, motifs against null ensembles, and their change over frames.

## Requirements

### Requirement: Centralities, clustering, and paths
The analysis layer SHALL compute degree, betweenness, closeness, eigenvector, and PageRank centralities, the moments of
the degree distribution, the clustering coefficient, and the mean shortest path length of the largest component.

#### Scenario: Small world
- **WHEN** a user measures Watts-Strogatz graphs of 1000 nodes and degree 10 at rewiring 0 and 0.01
- **THEN** the clustering at 0.01 stays above 0.8 of its value at 0, while the mean path length falls below 0.3 of its
  value at 0

#### Scenario: Star centralities
- **WHEN** a user computes the betweenness centrality of a star of 11 nodes
- **THEN** the hub has normalized betweenness 1 and every leaf 0

### Requirement: Degree mixing is measured
The analysis layer SHALL compute degree assortativity and assortativity by any node attribute, and the mean neighbour
degree against degree.

#### Scenario: Attribute mixing on two domains
- **WHEN** a user computes the assortativity by state of the contact network of a Schelling system after relaxation and
  of a stochastic block system with two blocks and in-block probability ten times the out-block probability
- **THEN** both exceed 0.5, while a random relabelling of the same states gives within 0.05 of 0

### Requirement: Communities are detected
The analysis layer SHALL detect communities by modularity maximization and by label propagation, and return the
partition, its modularity, and its agreement with a reference partition by normalized mutual information.

#### Scenario: Planted partition
- **WHEN** a user detects communities of a stochastic block system of 4 blocks of 100 nodes with mean in-block degree 16
  and out-block degree 2
- **THEN** modularity maximization agrees with the planted blocks above 0.9 with a modularity above 0.5, and label
  propagation agrees above 0.5

### Requirement: Motifs are scored against null ensembles
The analysis layer SHALL count motifs of three and four nodes in directed and undirected graphs, and give their
z-scores against ensembles of configuration-system graphs or degree-preserving edge switches of a chosen size and seed.

#### Scenario: Feed-forward loop
- **WHEN** a user scores the three-node motifs of a directed graph of 200 nodes with 50 planted feed-forward loops on a
  random background, against 100 edge-switched graphs
- **THEN** the feed-forward loop has a z-score above 3, and the same seed gives the same z-scores

### Requirement: Evolving networks are measured over frames
The analysis layer SHALL compute any network measure over the frames of an evolving network and return one row per
frame, with the fitted degree exponent where the degree distribution has a heavy tail.

#### Scenario: Growing network
- **WHEN** a user computes network measures over the frames of a preferential attachment run up to 10^4 nodes
- **THEN** the result holds one row per frame, and the degree exponent fitted at the last frame lies within 0.2 of 3

#### Scenario: Adaptive rewiring
- **WHEN** a user computes the assortativity by state over the frames of an adaptive voter system with rewiring
- **THEN** the assortativity rises over frames and ends above 0.9 when the network fragments

### Requirement: Networks renormalize across scales
`renormalization` SHALL coarse-grain a graph, or the network of a trajectory, by its Laplacian (`'laplacian'`)
or by the angles of its nodes (`'geometric'`), and SHALL set the network at every scale with its degree distribution,
degree exponent, and clustering.

#### Scenario: Self-similar network
- **WHEN** a popularity-similarity network is renormalized geometrically three times
- **THEN** its degree exponent, as a mean over four seeds, stays within 0.2 of the original at every scale

#### Scenario: Laplacian blocks
- **WHEN** two complete graphs of 20 nodes joined by one edge are renormalized by the Laplacian once
- **THEN** the network at the next scale has two nodes joined by one edge

### Requirement: Autocatalytic sets are found
`autocatalytic_sets` SHALL find the largest reflexively autocatalytic and food-generated set of a catalytic
reaction network, read from a network snapshot or chosen frames of a trajectory, with every hyperedge a reaction that
produces its last member and every edge from a catalyst to a product, by the algorithm of Hordijk and Steel. The food
set SHALL default to the nodes that no reaction produces.

#### Scenario: Catalysed reactions
- **WHEN** three reactions reachable from the food set each have a catalyst among their products
- **THEN** the largest RAF holds the three reactions, and leaves out a fourth whose reactant no reaction produces

#### Scenario: Lost catalyst
- **WHEN** one reaction of a chain loses its only catalyst
- **THEN** the reactions that need its product are removed with it, and no RAF exists
