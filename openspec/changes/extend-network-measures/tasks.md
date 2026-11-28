# Tasks

## 1. Summary

- [x] 1.1 Add `measures` and `SUMMARY_MEASURES` to `network_measures` in `src/skcomplex/core/analysis/_networks.py`,
  compute only the chosen measures in `_summary`, reject unknown names with `AnalysisError`, and verify the
  clustering-only and unknown-measure scenarios in `tests/core/analysis/test_networks.py`
- [x] 1.2 Add the `transitivity` and `reciprocity` fields to `NetworkMeasuresResult` and the summary, `nan` reciprocity
  on undirected frames, and verify the transitivity-against-average-clustering scenario and the reciprocity of the small
  directed graph
- [x] 1.3 Add `samples` and let the summary read `random_state`, with the sampled mean path length over seeded sources
  and the two sweep diameter, and verify the sampled-paths and same-seed scenarios

## 2. Directed degrees

- [x] 2.1 Add `in_degrees`, `out_degrees`, `in_moments`, and `out_moments` to `NetworkMeasuresResult` and fill them for
  directed frames of `'degrees'`, and verify the small-directed-graph and mean-in-and-out-degree scenarios

## 3. Spectrum

- [x] 3.1 Add the kind `'spectrum'` with its fields `adjacency_spectrum`, `laplacian_spectrum`,
  `normalized_laplacian_spectrum`, `algebraic_connectivity`, `fiedler_vector`, `spectral_radius`, and
  `modularity_matrix`, reading the undirected graph for the Laplacian measures and `nan` Fiedler vectors for
  disconnected frames, and verify the cycle, path, complete graph, disconnected graph, and modularity-matrix scenarios

## 4. Communities and weights

- [x] 4.1 Add `weight` to `network_measures`, read by `'communities'`, `'modularity'`, and `'spectrum'`, pass it to
  Louvain, the modularity, and the spectral functions, and verify the weighted-and-unweighted-Louvain scenario on the
  karate club
- [x] 4.2 Add the kind `'modularity'` with `partition`, the recoded labels, and the error without a partition, and
  verify the factions-of-the-karate-club and missing-partition scenarios
- [x] 4.3 Add `'girvan_newman'` and `'leading_eigenvector'` to `METHODS` and the setting `n_communities` with its method
  check, the leading eigenvector through igraph as `_motifs.py` loads it, and verify the bridge, highest-modularity,
  leading-eigenvector, and setting-only-some-methods-read scenarios. If #108 has landed, put the methods in its
  `'communities'` entry of `METHODS` instead

## 5. Centralities

- [x] 5.1 Add `'katz'` with the setting `alpha`, its default of 0.9 over the spectral radius, and the errors for a
  divergent `alpha` and for `alpha` with another measure, and verify the Katz-centrality and Katz-attenuation scenarios
- [x] 5.2 Add `'edge_betweenness'`, keyed by edges, and verify the edge-betweenness-of-a-bridge scenario

## 6. Components

- [x] 6.1 Add `members` to `ClustersResult` in `src/skcomplex/core/analysis/_clusters.py` and set `labels` and
  `members` of graph components for `'sizes'` and `'percolation'`, and verify the two-components scenario in
  `tests/core/analysis/test_clusters.py`

## 7. Documentation

- [x] 7.1 Update the docstrings of `network_measures`, `NetworkMeasuresResult`, `clusters`, and `ClustersResult` for
  every new kind, setting, method, measure, and field, with a doctest of the spectrum of a cycle and of the modularity
  of a given partition, and verify the doctests run
- [x] 7.2 Extend the network section of `docs/overview/user_guide/analyse.md` with the spectrum, the given partition,
  the new community methods, the `weight` switch, sampled paths, and directed degrees, and verify its Python blocks run
  under `pytest-markdown-docs`

## 8. Checks

- [x] 8.1 Run `black`, `ruff check`, `mypy`, `pydoclint`, `interrogate`, `lint-imports`, and
  `python openspec/scripts/check_conventions.py` on the touched files, and fix every finding without suppressions
- [x] 8.2 Run `tests/core/analysis/test_networks.py` and `tests/core/analysis/test_clusters.py` with the doctests of
  `_networks.py` and `_clusters.py`, with the thread counts of NumPy, Numba, and BLAS set to 1, and verify they pass
