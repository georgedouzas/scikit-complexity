## ADDED Requirements

### Requirement: Mean first-passage times between every pair of states
`passage(source, kind='mean-first-passage-matrix')` SHALL return the mean first-passage time from every state to every
other state of an irreducible transition matrix, with the mean return times on the diagonal. A NetworkX graph as the
source SHALL stand for the transition matrix `P = D^-1 A` of the random walk on it.

#### Scenario: Ring of five nodes
- **WHEN** the matrix of `nx.cycle_graph(5)` is computed
- **THEN** the time between neighbours is 4, the time between nodes two apart is 6, and every diagonal entry is 5

#### Scenario: Reducible chain
- **WHEN** the source is the graph of two disjoint edges
- **THEN** an `AnalysisError` says that the chain is reducible

### Requirement: Mean return times of every state
`passage(source, kind='mean-return-times')` SHALL return the mean return time `1 / pi_i` of every state of an
irreducible transition matrix or of the random walk on a graph, which is `2 L / k_i` on a connected graph of `L` edges.

#### Scenario: Karate club
- **WHEN** the mean return times of Zachary's karate club are computed
- **THEN** every node has `2 L / k_i` within `1e-9`, with `L = 78`
