# Tasks

## 1. Correlations and the graph

- [x] 1.1 Add `correlation_network`, `CorrelationNetworkResult`, `METHODS`, and `TAGS` in
  `src/skcomplex/core/analysis/_correlation_network.py`, reading series with `_read_series` and `_read_real`, dropping
  rows with a missing value, with the Pearson, Spearman, and Kendall correlations and p-values and the graph of the
  links, and verify the two-groups, Pearson-p-values, labels-of-a-trajectory, graph-for-network-measures, and
  unknown-method scenarios in `tests/core/analysis/test_correlation_network.py`
- [x] 1.2 Export `correlation_network` and `CorrelationNetworkResult` from `src/skcomplex/core/analysis/__init__.py`
  and list `correlation_network` in `ANALYSES` of `src/skcomplex/core/analysis/_discovery.py`, and verify that
  `applicable_analyses` of an array of series lists it in `tests/core/analysis/test_discovery.py`

## 2. Corrections

- [x] 2.1 Add `correction` with `'bonferroni'`, `'benjamini-hochberg'` through `scipy.stats.false_discovery_control`,
  and `None`, and the `threshold` of each, and verify the Bonferroni-threshold, Benjamini-Hochberg-links,
  independent-series, and correction-order scenarios

## 3. Backbones

- [x] 3.1 Add `backbone` with `'spanning-tree'` through `nx.maximum_spanning_tree` and `'disparity'` as a private
  function over the matrix, with `SETTINGS` and `_check_settings` on `backbone`, and verify the spanning-tree,
  disparity-filter, disparity-significance, and ignored-setting scenarios

## 4. Documentation

- [x] 4.1 Write the docstrings of `correlation_network` and `CorrelationNetworkResult` with plain examples from
  finance, neuroscience, and climate, and a doctest of two correlated series and one independent series, and verify
  the doctest runs
- [x] 4.2 Add a paragraph and a runnable example of `correlation_network` passed to `network_measures` in the network
  section of `docs/overview/user_guide/analyse.md`, and verify its Python blocks run under `pytest-markdown-docs`

## 5. Checks

- [x] 5.1 Run `black`, `ruff check`, `mypy`, `pydoclint`, `interrogate`, `lint-imports`, and
  `python openspec/scripts/check_conventions.py` on the touched files, and fix every finding without suppressions
- [x] 5.2 Run `tests/core/analysis/test_correlation_network.py` and `tests/core/analysis/test_discovery.py` with the
  doctests of `_correlation_network.py`, with the thread counts of NumPy, Numba, and BLAS set to 1, and verify they
  pass
