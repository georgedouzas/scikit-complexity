## ADDED Requirements

### Requirement: Correlation networks are built from time series
`correlation_network(source, method=...)` SHALL return the Pearson, Spearman, or Kendall correlation of every pair of
series of a source, the two sided p-value of every pair, the significant links, and an undirected graph with one node
per series and one edge per link that `network_measures` reads.

#### Scenario: Two groups of series
- **WHEN** a user builds the network of 10 seeded series of 500 steps, two groups of 5 that each share one common
  Gaussian factor added to independent Gaussian noise
- **THEN** the links are the 20 pairs within a group and no pair across groups

#### Scenario: Pearson p-values
- **WHEN** a user builds the network of any array of series with `method='pearson'`
- **THEN** every correlation and p-value equals that of `scipy.stats.pearsonr` on the pair within 1e-12

#### Scenario: Labels of a trajectory
- **WHEN** a user builds the network of a trajectory with `variables` naming three of its series
- **THEN** the labels and the nodes of the graph are those three names in that order

#### Scenario: Graph for network measures
- **WHEN** the `graph` of a result is passed to `network_measures`
- **THEN** it is read as an undirected graph whose edge weight is the absolute correlation, with the signed correlation
  and the p-value on every edge

#### Scenario: Unknown method
- **WHEN** a user builds the network with `method='distance'`
- **THEN** an `AnalysisError` lists `'pearson'`, `'spearman'`, and `'kendall'`

### Requirement: Correlation thresholds correct for many tests
The links of a correlation network SHALL be the pairs whose p-value passes `level` after the correction in
`correction`, `'bonferroni'`, `'benjamini-hochberg'` by default, or `None` for no correction, and the result SHALL give
the p-value threshold that the correction sets.

#### Scenario: Bonferroni threshold
- **WHEN** a user builds the network of 20 series with `correction='bonferroni'` and `level=0.05`
- **THEN** the threshold is 0.05 over 190, and every link has a p-value below it

#### Scenario: Benjamini-Hochberg links
- **WHEN** a user builds the network of any array of series with the default correction
- **THEN** the links are the pairs whose `scipy.stats.false_discovery_control` adjusted p-value is below the level

#### Scenario: Independent series
- **WHEN** a user builds the networks of 20 independent white noise series of 1000 steps for 200 seeds with
  `correction='bonferroni'` and `level=0.05`
- **THEN** at most 10% of the networks hold any link

#### Scenario: Correction order
- **WHEN** a user builds the network of one array with no correction, with Benjamini-Hochberg, and with Bonferroni at
  the same level
- **THEN** every Bonferroni link is a Benjamini-Hochberg link, and every Benjamini-Hochberg link is an uncorrected link

### Requirement: Correlation networks have backbones
With `backbone='spanning-tree'`, the links SHALL be the maximum spanning tree of the signed correlations. With
`backbone='disparity'`, the links SHALL be the pairs that the disparity filter of the absolute correlations finds
significant at `level` for at least one of their ends. A setting that the backbone does not read SHALL raise
`AnalysisError`.

#### Scenario: Spanning tree of two groups
- **WHEN** a user builds the spanning tree of the two groups of series
- **THEN** it holds 9 links, of which exactly one joins the two groups

#### Scenario: Disparity filter of two groups
- **WHEN** a user builds the disparity backbone of the two groups of series with `level=0.2`
- **THEN** it holds the 20 pairs within a group and no pair across groups

#### Scenario: Disparity significance
- **WHEN** a user builds the disparity backbone of any array of series
- **THEN** a pair is a link exactly when `(1 - w / s) ** (k - 1)` is below the level at one of its ends, with `w` the
  absolute correlation of the pair, `s` the sum of the absolute correlations of the end, and `k` its number of pairs

#### Scenario: Ignored setting
- **WHEN** a user builds the spanning tree with `level=0.01`
- **THEN** an `AnalysisError` says that the spanning tree does not read the level
