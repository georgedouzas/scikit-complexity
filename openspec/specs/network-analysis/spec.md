# network-analysis Specification

## Purpose
Measures the structure of static and evolving networks from any source, a network space, agent contacts, a lattice, or
a given graph: centralities, mixing, communities, motifs against null ensembles, and their change over frames.

## Requirements

### Requirement: Centralities, clustering, and paths
The analysis layer SHALL compute degree, betweenness, closeness, eigenvector, PageRank, and Katz centralities, the
betweenness of every edge, the moments of the degree distribution, the average clustering coefficient, the
transitivity, and the mean shortest path length of the largest component.

#### Scenario: Small world
- **WHEN** a user measures Watts-Strogatz graphs of 1000 nodes and degree 10 at rewiring 0 and 0.01
- **THEN** the clustering at 0.01 stays above 0.8 of its value at 0, while the mean path length falls below 0.3 of its
  value at 0

#### Scenario: Star centralities
- **WHEN** a user computes the betweenness centrality of a star of 11 nodes
- **THEN** the hub has normalized betweenness 1 and every leaf 0

#### Scenario: Transitivity against average clustering
- **WHEN** a user summarises a triangle with 10 leaves attached to one of its corners
- **THEN** the average clustering is 0.155 and the transitivity 0.044, both within 1e-3

#### Scenario: Katz centrality
- **WHEN** a user computes the Katz centrality of a star of 5 nodes with `alpha=0.1`
- **THEN** it equals the NetworkX Katz centrality of the same graph and `alpha` within 1e-12, and the hub ranks first

#### Scenario: Katz attenuation by default
- **WHEN** a user computes the Katz centrality without `alpha`
- **THEN** the attenuation is 0.9 over the spectral radius of the frame, and an `alpha` at or above the inverse
  spectral radius raises `AnalysisError`

#### Scenario: Edge betweenness of a bridge
- **WHEN** a user computes the edge betweenness of two complete graphs of 5 nodes joined by one edge
- **THEN** the joining edge has the largest betweenness

### Requirement: Degree mixing is measured
The analysis layer SHALL compute degree assortativity and assortativity by any node attribute, and the mean neighbour
degree against degree.

#### Scenario: Attribute mixing on two domains
- **WHEN** a user computes the assortativity by state of the contact network of a Schelling system after relaxation and
  of a stochastic block system with two blocks and in-block probability ten times the out-block probability
- **THEN** both exceed 0.5, while a random relabelling of the same states gives within 0.05 of 0

### Requirement: Communities are detected
The analysis layer SHALL detect communities by Louvain modularity maximization, by label propagation, by Girvan-Newman
removal of the edges of highest betweenness, and by the leading eigenvector of the modularity matrix, and return the
partition, its modularity, and its agreement with a reference partition by normalized mutual information.

#### Scenario: Planted partition
- **WHEN** a user detects communities of a stochastic block system of 4 blocks of 100 nodes with mean in-block degree 16
  and out-block degree 2
- **THEN** modularity maximization agrees with the planted blocks above 0.9 with a modularity above 0.5, and label
  propagation agrees above 0.5

#### Scenario: Girvan-Newman on a bridge
- **WHEN** a user detects two communities of two complete graphs of 5 nodes joined by one edge with
  `method='girvan_newman'` and `n_communities=2`
- **THEN** the communities are the two complete graphs

#### Scenario: Girvan-Newman of highest modularity
- **WHEN** a user detects communities of the karate club with `method='girvan_newman'`, `weight=None`, and no
  `n_communities`
- **THEN** the partition is the split of the Girvan-Newman hierarchy of highest modularity, 5 communities of modularity
  0.401 within 1e-3

#### Scenario: Leading eigenvector
- **WHEN** a user detects communities of the karate club with `method='leading_eigenvector'` and `weight=None`
- **THEN** the modularity exceeds 0.39, and the same call gives the same partition

#### Scenario: Setting only some methods read
- **WHEN** a user passes `n_communities` with `method='louvain'`
- **THEN** an `AnalysisError` names the setting and the methods that read it

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

### Requirement: Edge weights are chosen explicitly
`network_measures` SHALL read the edge attribute named by `weight` as the weight of the communities, the modularity,
and the spectrum, and SHALL treat the graph as unweighted when `weight` is `None`. The default `'weight'` SHALL keep
the weighted result of Louvain.

#### Scenario: Weighted and unweighted Louvain
- **WHEN** a user detects communities of the karate club by Louvain with `random_state=0`, once by default and once with
  `weight=None`
- **THEN** the modularities are 0.4439 and 0.4198 within 1e-4

### Requirement: The modularity of a given partition is computed
`network_measures(kind='modularity')` SHALL return the modularity of the partition in `partition`, given as the
community label of every node in node order or as the name of a node attribute, with its labels.

#### Scenario: Factions of the karate club
- **WHEN** a user computes the modularity of the karate club with `partition='club'` and `weight=None`
- **THEN** the modularity is 0.3582 within 1e-4

#### Scenario: Missing partition
- **WHEN** a user asks for the kind `'modularity'` without a partition
- **THEN** an `AnalysisError` asks for `partition`

### Requirement: The summary computes chosen measures
`network_measures(kind='summary')` SHALL compute only the measures named in `measures`, among `'size'`,
`'clustering'`, `'transitivity'`, `'paths'`, `'components'`, and `'reciprocity'`, and all of them by default, leaving
the fields of the others `None`.

#### Scenario: Clustering only
- **WHEN** a user summarises a graph with `measures=['clustering', 'transitivity']`
- **THEN** the clustering and the transitivity are set, and the path length, the diameter, and the components are
  `None`

#### Scenario: Unknown measure
- **WHEN** a user summarises a graph with `measures=['radius']`
- **THEN** an `AnalysisError` lists the measures of the summary

### Requirement: Path lengths are sampled on request
With `samples` set, the summary SHALL estimate the mean shortest path length of the largest component from that many
source nodes drawn with `random_state`, and SHALL give a lower bound of the diameter by the two sweep method.

#### Scenario: Sampled paths of a large graph
- **WHEN** a user summarises a seeded graph of 1000 nodes of `ConfigurationModelSystem(exponent=2.5)` with
  `samples=200`
- **THEN** the mean path length lies within 2% of the exact one, and the diameter is at most the exact one

#### Scenario: Same seed
- **WHEN** a user summarises one graph twice with `samples=50` and the same `random_state`
- **THEN** both results hold the same path length and diameter

### Requirement: Directed degrees and reciprocity are measured
On a directed frame, the degrees SHALL also give the in degree and the out degree of every node with their moments,
and the summary SHALL give the reciprocity, the share of edges whose reverse edge is present. On an undirected frame
the in and out degrees SHALL be `None` and the reciprocity `nan`.

#### Scenario: Small directed graph
- **WHEN** a user measures the directed graph with edges 0 to 1, 1 to 0, 1 to 2, and 2 to 0
- **THEN** the in degrees are `[2, 1, 1]`, the out degrees `[1, 2, 1]`, and the reciprocity 0.5

#### Scenario: Mean in and out degree
- **WHEN** a user measures the degrees of any directed graph
- **THEN** the first moments of the in and out degrees are equal, the number of edges over the number of nodes

### Requirement: Graph spectra are computed
`network_measures(kind='spectrum')` SHALL give for every frame the adjacency eigenvalues, the Laplacian and normalised
Laplacian eigenvalues, the algebraic connectivity with its Fiedler vector, the spectral radius, and the modularity
matrix. The Laplacian measures SHALL read a directed frame as undirected.

#### Scenario: Laplacian of a cycle
- **WHEN** a user computes the spectrum of a cycle of 4 nodes
- **THEN** the Laplacian eigenvalues are 0, 2, 2, and 4 in ascending order within 1e-12

#### Scenario: Algebraic connectivity of a path
- **WHEN** a user computes the spectrum of a path of 4 nodes
- **THEN** the algebraic connectivity is `2 - 2 cos(π / 4)` within 1e-12, and the Fiedler vector changes sign once
  along the path

#### Scenario: Spectral radius and epidemic threshold
- **WHEN** a user computes the spectrum of a complete graph of 5 nodes
- **THEN** the spectral radius is 4 within 1e-12, so the epidemic threshold `1 / λ1` is 0.25

#### Scenario: Disconnected graph
- **WHEN** a user computes the spectrum of two disjoint edges
- **THEN** the algebraic connectivity is 0 and the Fiedler vector holds `nan` for every node

#### Scenario: Modularity matrix rows
- **WHEN** a user computes the spectrum of any undirected graph
- **THEN** every row of the modularity matrix sums to 0 within 1e-12

### Requirement: Graph components list their members
`clusters` of a graph SHALL give, besides the sizes, the label of every node in node order, from 1 for the largest
component, and the nodes of every component, largest first, for the kinds `'sizes'` and `'percolation'`.

#### Scenario: Two components
- **WHEN** a user computes `clusters(graph, kind='sizes')` of a path of 3 nodes and a separate edge
- **THEN** the sizes are `[3, 2]`, the labels `[1, 1, 1, 2, 2]`, and the members the nodes of the path, then those of
  the edge
