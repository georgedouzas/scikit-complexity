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
