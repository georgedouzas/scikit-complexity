## ADDED Requirements

### Requirement: Core-periphery structure is detected
The analysis layer SHALL split the nodes of every frame into a core and a periphery by the discrete model of Borgatti
and Everett (1999), whose ideal pattern links every pair that holds a core node, or by the innermost k-core. It SHALL
return the core, the coreness of every node, and the quality of the split, the correlation of the adjacency with the
ideal pattern.

#### Scenario: Star
- **WHEN** a user detects the core of a star of 11 nodes
- **THEN** the hub is the only core node and the quality is 1 within 1e-12

#### Scenario: Planted core
- **WHEN** a user detects the core of a stochastic block graph of a core of 20 nodes and a periphery of 80 nodes with
  link probabilities 0.8 within the core, 0.3 between core and periphery, and 0.02 within the periphery
- **THEN** both methods agree with the planted core on at least 90% of the nodes, and the quality exceeds 0.5

#### Scenario: Same seed
- **WHEN** a user detects the core of one graph twice with the same seed
- **THEN** both results hold the same core

### Requirement: Directed and weighted graphs are symmetrised for the core
Core-periphery detection SHALL read a directed graph as undirected with the sum of reciprocal edge weights, and SHALL
read the edge attribute `weight` when it is present and 1 otherwise.

#### Scenario: Transfer counts
- **WHEN** a user detects the core of a directed graph whose edges carry transfer counts as `weight`
- **THEN** the result equals that of the undirected graph whose weights are the sums of the counts in both directions

### Requirement: The method default follows the kind
`network_measures` SHALL take `method=None` by default and SHALL then use `'louvain'` for communities and
`'borgatti-everett'` for the core. A method that the kind does not offer SHALL raise `AnalysisError` that lists the
methods of the kind.

#### Scenario: Communities by default
- **WHEN** a user detects communities without a method
- **THEN** the communities are those of the Louvain method with the same seed

#### Scenario: Wrong method for the kind
- **WHEN** a user asks for the core with `method='louvain'`
- **THEN** an `AnalysisError` lists `'borgatti-everett'` and `'k-core'`

### Requirement: Causal networks are inferred from time series
`causal_network(source, method=...)` SHALL return the lagged causal links between the series of a source, by PCMCI,
convergent cross mapping, Granger tests of a vector autoregression, or transfer entropy, with the strength and p-value
of every link and lag, the significant links at a level, and a directed graph that `network_measures` reads.

#### Scenario: Known linear links
- **WHEN** 100 seeded realisations of 500 steps of the three-variable linear model `x1(t) = 0.7 x1(t-1) + e`, `x2(t) =
  0.8 x2(t-1) + 0.5 x1(t-1) + e`, `x3(t) = 0.5 x3(t-1) + 0.6 x2(t-2) + e` are analysed with `'pcmci'` at level 0.05
- **THEN** the links `x1 -> x2` at lag 1 and `x2 -> x3` at lag 2 are found in at least 95% of the realisations, and the
  share of realisations with the indirect link `x1 -> x3` is at most 0.1, close to the level as in Runge et al.
  (1702.07007)

#### Scenario: Granger on the same model
- **WHEN** one realisation of the model is analysed with `'granger'` and `max_lag=2`
- **THEN** the significant links are `x1 -> x2` and `x2 -> x3`, besides the links of every variable to itself

#### Scenario: Cross mapping of Lorenz-63
- **WHEN** the `x` and `y` series of a Lorenz-63 run sampled every 0.1 are analysed with `'ccm'` and library sizes from
  100 to 5000, after Datseris and Parlitz (Fig. 7.3 and 7.4)
- **THEN** the cross-map skill in both directions rises with the library size, its value at 5000 exceeds its value at
  100, and both links are significant

#### Scenario: Independent systems
- **WHEN** the `x` series of two Lorenz-63 runs from different initial states are analysed with `'ccm'`
- **THEN** no link is significant at level 0.05 with the seed fixed

#### Scenario: Graph for network measures
- **WHEN** the `graph` of a result is passed to `network_measures`
- **THEN** it is read as a directed weighted graph with one node per variable
