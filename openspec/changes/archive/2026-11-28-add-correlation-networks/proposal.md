# Proposal

## Why

A network built from many time series, one node per series and one link per significant correlation, is how stock
markets, brain regions, climate grid points, and co-expressed genes are turned into graphs, and the Networks chapter,
§7, needs it (issue #68). The library measures graphs and correlates single series, but it cannot build this network:
there is no correlation matrix with p-values, no correction for the many tests of all pairs, and no backbone such as
the maximum spanning tree of Mantegna or the disparity filter of Serrano, Boguñá, and Vespignani.

## What Changes

- New analysis `correlation_network(source, *, method='pearson', variables=None, level=0.05,
  correction='benjamini-hochberg', backbone=None)` in `skcomplex.core.analysis`, which reads the series of a
  trajectory or the columns of an array and returns a frozen `CorrelationNetworkResult`.
- `method` picks the correlation, `'pearson'`, `'spearman'`, or `'kendall'`, each with the two sided p-value of every
  pair.
- `correction` picks the threshold of the p-values over all pairs at `level`: `'bonferroni'`, `'benjamini-hochberg'`,
  the default, or `None` for the level itself.
- `backbone` keeps instead a backbone of the complete weighted network: `'spanning-tree'`, the maximum spanning tree of
  the correlations, or `'disparity'`, the edges that the disparity filter finds significant at `level` for one of
  their ends.
- The result holds the labels of the series, the correlation matrix as `strengths`, the `p_values`, the p-value
  `threshold` of the correction, the kept `links`, and `graph`, a NetworkX graph of the links that `network_measures`
  reads, with the same field names that the planned `causal_network` of `extend-network-and-emergence-measures` (#108)
  gives.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `network-analysis`: networks are built from the correlations of many time series, with corrected thresholds and
  backbones.

## Impact

- Code: a new module `src/skcomplex/core/analysis/_correlation_network.py`, exported from
  `src/skcomplex/core/analysis/__init__.py` and listed in `ANALYSES` of `src/skcomplex/core/analysis/_discovery.py`.
- Tests: `tests/core/analysis/test_correlation_network.py`.
- Docs: the network section of `docs/overview/user_guide/analyse.md`.
- Dependencies: none new. NumPy and SciPy 1.18.1 (2026-08-21) give the correlations, their p-values, and the
  Benjamini-Hochberg adjustment, and NetworkX 3.7 (2026-09-21) gives the graph and the maximum spanning tree.
- `causal_network` (#108) and `correlation_network` share their argument order and result fields, so a user swaps one
  for the other. #108 is not edited.
