# Proposal

## Why

`network_measures` and `clusters` answer the first questions about a graph but stop short of what a Networks course and
ordinary practice need, as issues #65, #66, #67, #69, and #72 report. The summary always computes all-pairs path
lengths, which takes over 100 s on `ConfigurationModelSystem(n_nodes=5000, exponent=2.5)`, and gives the average
clustering but not the transitivity. A directed graph gives total degrees only. There is no spectrum, so the epidemic
threshold `1 / λ1` and the Laplacian spectrum of the Networks chapter cannot be computed. Communities cannot be scored
for a given partition, Girvan-Newman and the leading eigenvector method are missing, and Louvain silently reads edge
weights (on the karate club the weighted modularity is 0.4439 and the unweighted 0.4198). `clusters(graph,
kind='sizes')` gives component sizes but not which nodes they hold, and Katz centrality is missing.

## What Changes

- `network_measures(kind='summary')` gains `measures`, the measures to compute among `'size'`, `'clustering'`,
  `'transitivity'`, `'paths'`, `'components'`, and `'reciprocity'`, all of them by default, so a large graph can skip
  the paths.
- `network_measures(kind='summary')` gains `samples`, the number of seeded source nodes from which the mean path length
  is estimated, with the diameter then a lower bound by the two sweep method, and reads `random_state` for the draw.
- The summary gains `transitivity`, the global clustering coefficient, and `reciprocity`, the share of directed edges
  whose reverse is present, `nan` for an undirected frame.
- `network_measures(kind='degrees')` on a directed graph gains `in_degrees` and `out_degrees` of every node, with their
  moments `in_moments` and `out_moments`.
- New kind `'spectrum'` of `network_measures`: the adjacency eigenvalues, the Laplacian and the normalised Laplacian
  eigenvalues, the algebraic connectivity with its Fiedler vector, the spectral radius, and the modularity matrix of
  every frame.
- New kind `'modularity'` of `network_measures`: the modularity of a partition the user gives in `partition`, as
  community labels in node order or the name of a node attribute.
- `network_measures(kind='communities')` gains the methods `'girvan_newman'` and `'leading_eigenvector'`, and the
  setting `n_communities`, the number of communities they stop at, or the split of highest modularity by default.
- `network_measures` gains `weight`, the edge attribute read as the weight by the communities, the modularity, and the
  spectrum, or `None` for an unweighted graph. Its default `'weight'` keeps the present Louvain result.
- `network_measures(kind='centrality')` gains the measures `'katz'`, with the setting `alpha`, and `'edge_betweenness'`,
  the betweenness of every edge.
- `clusters` of a graph with the kinds `'sizes'` and `'percolation'` gains `labels`, the component of every node in
  node order, and `members`, the nodes of every component, both largest first.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `network-analysis`: centralities gain Katz and edge betweenness, the summary gains a choice of measures, sampled
  paths, transitivity, and reciprocity, directed graphs give in and out degrees, graph spectra are computed,
  communities gain a given partition, Girvan-Newman, the leading eigenvector method, and a weight switch, and graph
  components list their members.

## Impact

- Code: `src/skcomplex/core/analysis/_networks.py` (new kinds, settings, methods, measures, and
  `NetworkMeasuresResult` fields) and `src/skcomplex/core/analysis/_clusters.py` (`ClustersResult.members` and the
  labels of graph components).
- Tests: `tests/core/analysis/test_networks.py` and `tests/core/analysis/test_clusters.py`.
- Docs: the network section of `docs/overview/user_guide/analyse.md`.
- Dependencies: none new. NetworkX 3.7 (2026-09-21) gives every measure but the leading eigenvector method, which
  igraph 1.0.0 (2025-10-23), already a runtime dependency for motifs, gives.
- The pending change `extend-network-and-emergence-measures` (#108) turns `METHODS` into a mapping from kind to
  methods with `method=None`. Whichever change lands second lists the new community methods under `'communities'`.
