# Design

## Context

The analyses of `skcomplex.core.analysis` are functions that take a `source` and keyword settings, check string
options with `_check_choice`, reject a changed setting the option does not read with `_check_settings`, declare the
data they read as the module constant `TAGS`, and return a frozen `*Result` dataclass. `_read_series` of `_base.py`
gives the named series of a trajectory, or the columns of an array, as a private `_Series` of times, values, and
labels. `correlation` in `_correlation.py` correlates one series with itself or a field with itself, and builds no
network. `network_measures` reads any NetworkX graph.

The pending change `extend-network-and-emergence-measures` (#108) plans `causal_network(source, *, method='pcmci',
variables=None, max_lag=5, level=0.05, test='parcorr', embedding=None, libraries=None, random_state=None)` in
`_causal_network.py`, with `AnalysisTags(DataKind.SERIES)` and a frozen `CausalNetworkResult` of `labels`, `strengths`
and `p_values` of shape `(variables, variables, max_lag + 1)`, `links` as `(source, target, lag)` tuples, and `graph`,
a NetworkX `DiGraph` with the strongest significant lag and its strength on every edge.

## Goals / Non-Goals

**Goals:**

- One analysis that turns many series into a graph that every kind of `network_measures` reads.
- The same shape of call and result as `causal_network`, so the two read as one family.

**Non-Goals:**

- Lagged cross correlations, partial correlations, and mutual information networks. `causal_network` covers directed
  and lagged links.
- Pooling several runs into one network, which `causal_network` plans for observed runs. A user stacks the runs first.
- Random matrix filtering of the correlation matrix and planar maximally filtered graphs.

## Decisions

### Name and signature

```python
correlation_network(
    source,
    *,
    method='pearson',
    variables=None,
    level=0.05,
    correction='benjamini-hochberg',
    backbone=None,
)
```

The name follows `causal_network`, what the links mean followed by `_network`. The first three settings after the
source, `method`, `variables`, and `level`, have the meaning and the order they have in `causal_network`: `method` is
the estimator of the links, `variables` the series of a trajectory, and `level` the significance level. The settings
that only one of the two reads, `correction` and `backbone` here and `max_lag` and `test` there, come after. A single
function with a `method` that spans correlations and causal discovery was rejected, because the two read different
settings and give results of different shapes, undirected and lag free here, directed and lagged there.

### Result

`CorrelationNetworkResult` is a frozen dataclass with the field names of `CausalNetworkResult` where the quantities
match:

- `labels`, the name of every series in column order,
- `strengths`, the correlation matrix of shape `(variables, variables)`, with ones on the diagonal,
- `p_values`, the two sided p-value of every pair, of the same shape, `nan` on the diagonal,
- `threshold`, the p-value below which a pair is a link, or `nan` for a backbone,
- `links`, the kept pairs as `(source, target)` tuples of labels, the source before the target in label order,
- `graph`, a NetworkX `Graph` whose nodes are the labels in order and whose edges are the links, each with `weight`,
  the absolute correlation, `strength`, the signed correlation, and `p_value`.

`links` holds labels, not indices, so a link reads by itself, and `graph` uses `weight` because NetworkX and the
`weight` setting of `network_measures` read that attribute. The edge attribute `strength` matches the field
`strengths`. The two results stay separate classes with no shared base, since a base with two children of different
shapes would add a class and save nothing. When #108 lands, its `links` tuples and edge attributes follow the same
labels and names, which its task for the graph can note, and #108 is not edited here.

### Correlation methods

`METHODS = ('pearson', 'spearman', 'kendall')`, the names of `pandas.DataFrame.corr` and of SciPy's functions.

- `'pearson'` computes `np.corrcoef` of the columns and the two sided p-value from Student's t distribution with
  `n - 2` degrees of freedom by `scipy.stats.t.sf`, the exact test that `scipy.stats.pearsonr` runs, without the
  `(n, variables, variables)` array that broadcasting `pearsonr` over all pairs would build.
- `'spearman'` calls `scipy.stats.spearmanr` on the array, which gives both matrices at once.
- `'kendall'` calls `scipy.stats.kendalltau` on every pair, the only form SciPy offers.

Rows with a missing value in any column are dropped before correlating, so every pair uses the same times, and a
constant series has `nan` correlations and p-values and no links. `_read_real` rejects complex series.

### Corrections

`CORRECTIONS = ('bonferroni', 'benjamini-hochberg')` and `None`. The `m = variables (variables - 1) / 2` p-values of
the pairs above the diagonal are corrected together.

- `'bonferroni'` keeps the pairs with `p < level / m`, and the threshold is `level / m`.
- `'benjamini-hochberg'` keeps the pairs whose adjusted p-value from `scipy.stats.false_discovery_control(p,
  method='bh')` is below `level`, and the threshold is `k level / m` for the `k` kept pairs, the step up cut of
  Benjamini and Hochberg (1995). It is the default because it controls the false discovery rate, the rate a network
  of many links cares about, while Bonferroni controls the chance of any false link and loses most true ones in large
  networks.
- `None` keeps the pairs with `p < level`, and the threshold is `level`.

The task suggested statsmodels for the multiple testing. SciPy 1.18.1 (2026-08-21), already a runtime dependency, has
`false_discovery_control` since 1.11, and Bonferroni is one division, so no dependency is added. statsmodels 0.15.0
(2026-08-27) is planned by #108 for Granger tests and is not needed here.

### Backbones

`BACKBONES = ('spanning-tree', 'disparity')` and `None`. A backbone is drawn from the complete weighted network of all
pairs, not from the corrected links, as its authors define it.

- `'spanning-tree'` keeps `nx.maximum_spanning_tree` of the complete graph weighted by the signed correlation, which
  is the minimum spanning tree of Mantegna (1999) in the distance `sqrt(2 (1 - r))`. It reads neither `level` nor
  `correction`, so `_check_settings` rejects both when changed.
- `'disparity'` keeps a pair when `(1 - w / s) ** (k - 1)` is below `level` at either end, with `w` the absolute
  correlation of the pair, `s` the strength of the end, and `k` its degree, after Serrano, Boguñá, and Vespignani
  (2009). It reads `level` but not `correction`.

No maintained library gives the disparity filter. NetworkX 3.7 has none, and the backbone packages on PyPI, such as
`netbone` 0.2.3 (2024-06-19) and `backbone-network` 1.1.3 (2022-08-16), have had no release within the 24 months the
constitution asks. It is a few lines of NumPy over the matrix, a private function of the module.

### Settings check

`_check_settings(correlation_network, 'backbone', SETTINGS[backbone], ...)` with `SETTINGS = {None: {'level',
'correction'}, 'spanning-tree': set(), 'disparity': {'level'}}`, besides the settings every backbone reads, `method` and
`variables`. `_check_choice` checks `method`, `correction`, and `backbone`.

### Placement and tags

Step 2 of the placement rule: an analysis of the core that reads series by their tags, `TAGS =
AnalysisTags(DataKind.SERIES)`. It is cross-domain: asset returns in finance (Mantegna), functional connectivity of
brain regions in neuroscience, climate networks of grid points, co-expression networks of genes, and synchronised
populations in ecology. The module `_correlation_network.py` is new inside the existing package
`skcomplex.core.analysis`, the way #108 adds `_causal_network.py`, so no package is created or moved. The function is
exported from the package and listed in `ANALYSES` of `_discovery.py`, so `applicable_analyses` offers it for series.

## Risks / Trade-offs

- [Kendall's tau loops over all pairs in Python] → The docstring says it suits tens of series, and Pearson and
  Spearman are vectorised.
- [The correlation matrix is dense, `variables²` floats] → A few thousand series fit in memory, which covers the uses
  above. The docstring states the size.
- [Dropping rows with a missing value loses data] → It keeps every pair on the same times, which the p-value
  correction assumes, and the docstring says so.
- [The disparity filter keeps few links of a dense network with even weights] → That is its definition. The docstring
  says so and the scenario uses a level suited to 10 series.
