# Design

## Context

`network_measures` in `src/skcomplex/core/analysis/_networks.py` is one function that dispatches on `kind` over
`KINDS`, checks `measure` against `CENTRALITIES` and `method` against `METHODS` with `_check_choice`, and rejects a
changed setting the kind does not read through `SETTINGS` and `_check_settings`. It reads one graph per frame with
`_read_networks` and returns a frozen `NetworkMeasuresResult` whose fields are `None` unless the kind fills them, one
entry per frame. `_summary` always runs `nx.average_shortest_path_length` and `nx.diameter` on the largest component,
`_degrees` reads `graph.degree()` only, `_centrality` maps a name to a NetworkX function, and `_communities` calls
`louvain_communities` and `modularity` with the NetworkX default `weight='weight'`.

`clusters` in `_clusters.py` reads a graph in `_of_graph`, which sorts the component sizes and drops the components.
`ClustersResult` already has `labels`, documented for grids and points, and no field for members.

The pending change `extend-network-and-emergence-measures` (#108) adds the kind `'core-periphery'`, makes `METHODS` a
mapping from kind to methods, and makes `method` default to `None`, which then picks `'louvain'` for communities.

## Goals / Non-Goals

**Goals:**

- Every new measure is a kind, a measure, a method, or a setting of the existing functions, read through the same
  `SETTINGS` check, with one field per quantity on the existing results.
- Every computation wraps NetworkX or igraph, both runtime dependencies already.

**Non-Goals:**

- Directed random graph models (the second half of #66), the walker start node, the master equation, and first
  passage times of #72. They belong to other changes.
- Weighted centralities and weighted path lengths. `weight` reaches only the communities, the modularity, and the
  spectrum.
- Sparse or partial eigensolvers for very large graphs. The spectrum is dense.

## Decisions

### Choice of summary measures

`measures: Sequence[str] | None = None`, read only by `'summary'`, names the measures to compute among
`SUMMARY_MEASURES = ('size', 'clustering', 'transitivity', 'paths', 'components', 'reciprocity')`. `None` computes all,
so the present call keeps its output and gains the new fields. `'size'` sets `nodes`, `edges`, and `mean_degree`,
`'paths'` sets `path_length` and `diameter`, and every other name sets its own field. A field of a measure not asked
for stays `None`, as the fields of other kinds do. An unknown name raises `AnalysisError` through `_check_choice` on
each item.

The issue suggested `measures=['clustering', 'degrees']`. The name `'degrees'` is taken by a kind, so the count
measures are `'size'`. The plural `measures` sits next to the singular `measure` of the centralities. The two are kept
apart by reading, `measure` picks one centrality and `measures` lists the summary measures, the way scikit-learn's
`scoring` takes one name or several. Renaming `measure` would break callers for no gain.

### Sampled path lengths

`samples: int | None = None`, read only by `'summary'`, together with `random_state`, which the summary now reads too.
With `samples` set, the mean path length is the mean over `samples` source nodes of the largest component, drawn
without replacement by `np.random.default_rng(random_state)`, of the shortest path lengths to every other node, from
`nx.single_source_shortest_path_length`. The diameter is `nx.approximation.diameter(largest, seed=random_state)`, the
two sweep lower bound of NetworkX. A `samples` at or above the component size computes the exact values. The sampled
mean is unbiased over sources, and the docstring says the diameter is then a lower bound.

### Transitivity and reciprocity

`transitivity` wraps `nx.transitivity` on the undirected graph, as `clustering` reads it. `reciprocity` wraps
`nx.overall_reciprocity` on a directed frame and is `nan` on an undirected one, so the array keeps one entry per frame
when an evolving network mixes neither.

### Directed degrees

On a `DiGraph` frame, `'degrees'` also sets `in_degrees` and `out_degrees`, from `graph.in_degree()` and
`graph.out_degree()` in node order, and `in_moments` and `out_moments`, the moments of the orders `orders`, as `moments`
does for the total degree. On an undirected frame the entry is `None`, and the fields are `None` when no frame is
directed. `degrees` and `moments` keep the total degree, so nothing breaks.

### Spectrum kind

`'spectrum'` reads `weight` and sets, per frame:

- `adjacency_spectrum`, from `nx.adjacency_spectrum`, real and ascending for an undirected frame, complex and ordered by
  descending modulus for a directed one,
- `laplacian_spectrum` and `normalized_laplacian_spectrum`, from `nx.laplacian_spectrum` and
  `nx.normalized_laplacian_spectrum` of the undirected graph, real and ascending,
- `algebraic_connectivity` and `fiedler_vector`, from `nx.algebraic_connectivity` and `nx.fiedler_vector` with
  `seed=0` for reproducibility, in node order,
- `spectral_radius`, the largest modulus of the adjacency eigenvalues,
- `modularity_matrix`, from `nx.modularity_matrix`, or `nx.directed_modularity_matrix` for a directed frame.

The Laplacian of a directed graph has several conventions, so the Laplacian measures read the undirected graph, as the
summary does, and the docstring says so. `nx.fiedler_vector` raises on a disconnected graph. The algebraic connectivity
of a disconnected graph is 0 and its Fiedler vector is not unique, so the vector is all `nan` there, rather than an
error that would stop an evolving network at its first split. A frame of fewer than two nodes gives an algebraic
connectivity of 0 and an empty vector.

### Modularity of a given partition

A kind `'modularity'` reads `partition` and `weight`. `partition: Sequence[object] | str | None = None` is the
community label of every node in node order or the name of a node attribute, as `reference` is. The kind sets `labels`,
the labels recoded to 0 onwards in order of first appearance, and `modularity`, from `nx.community.modularity`. A kind
is chosen over a `partition` setting of `'communities'`, because detection and scoring read different settings, and a
kind keeps the `SETTINGS` check exact. `'modularity'` without `partition` raises `AnalysisError`.

### New community methods

`METHODS` becomes `('louvain', 'label_propagation', 'girvan_newman', 'leading_eigenvector')`, the underscore style of
`'label_propagation'`. A new setting `n_communities: int | None = None` is read by `'communities'`.

- `'girvan_newman'` walks `nx.community.girvan_newman` and stops at the first split with `n_communities` communities.
  With `n_communities=None` it keeps the split of highest modularity, Newman and Girvan's own choice, walking until the
  graph has no edges. The edges are removed by unweighted betweenness, as in the paper, and `weight` enters only the
  modularity that picks the split.
- `'leading_eigenvector'` calls `community_leading_eigenvector(clusters=n_communities, weights=...)` of igraph on
  `igraph.Graph.from_networkx` of the undirected graph, through `load_module('igraph')` as `_motifs.py` does. NetworkX
  has no such method, and igraph is already a runtime dependency.
- `n_communities` with `'louvain'` or `'label_propagation'` raises `AnalysisError` naming the methods that read it, a
  check written beside the `SETTINGS` check because it depends on the method, not the kind.

Girvan-Newman costs `O(m² n)`, so the docstring says it suits graphs of a few hundred nodes.

### Weight switch

`weight: str | None = 'weight'` is read by `'communities'`, `'modularity'`, and `'spectrum'`. It goes to
`louvain_communities`, to `modularity`, to the leading eigenvector weights, and to the spectral functions. The default
keeps the present Louvain result and the NetworkX convention, so no call changes, and `weight=None` gives the
unweighted result of #69. Label propagation and the Girvan-Newman removals are topological, which the docstring states.
A default of `None` was rejected because it would change the result of existing calls silently.

### Katz and edge betweenness

`CENTRALITIES` gains `'katz'` and `'edge_betweenness'`. `'katz'` wraps `nx.katz_centrality_numpy(graph, alpha=...,
beta=1.0)`, which solves the linear system and so never fails to converge. A new setting `alpha: float | None = None`,
read by `'centrality'`, is the attenuation. `None` takes 0.9 over the spectral radius of the frame, a value that always
gives a finite centrality and adapts to every frame of an evolving network. An `alpha` at or above the inverse spectral
radius raises `AnalysisError`, since the Katz series diverges there. `alpha` with another measure raises
`AnalysisError`. `'edge_betweenness'` wraps `nx.edge_betweenness_centrality`, and its mapping is keyed by edges, as
the docstring of `centrality` then says.

### Component members

`_of_graph` keeps the components, sorted by size, largest first, ties broken by the position of their first node in
node order. It sets `labels`, the 1 based index of the component of every node in node order, as grids label clusters
from 1, and `members`, a new field of `ClustersResult` of type `list[list[object]] | None`, the nodes of every
component in node order. Both are set for `'sizes'` and `'percolation'` of a graph. Points and grids leave `members`
`None`.

### Placement

Step 2 of the placement rule: these are kinds and settings of analyses of the core. They are cross-domain. Spectra
serve epidemics (the threshold `1 / λ1`), synchronisation (the algebraic connectivity of coupled oscillators), and
diffusion. Communities and modularity serve social, metabolic, and neural networks. Reciprocity and in and out degrees
serve food webs, citation networks, and the World Wide Web. Katz centrality serves social influence and economics.

### Coordination with #108

#108 changes `METHODS` into `{'communities': (...), 'core-periphery': (...)}` with `method=None`. This change keeps the
tuple and the default `'louvain'`. Whichever change lands second moves the methods of this change into the
`'communities'` entry and the method check of `n_communities` into the same mapping. The task that touches `METHODS`
says so, and #108 is not edited.

## Risks / Trade-offs

- [The spectrum is dense, `O(n³)` per frame] → The docstring says so, and the summary, not the spectrum, is the kind to
  use on large graphs. A sparse partial solver can follow as a setting.
- [Girvan-Newman is slow] → The docstring gives the size it suits, and `n_communities` stops the walk early.
- [The igraph leading eigenvector method can split differently across igraph versions] → The tests check the
  modularity bound and repeatability, not one partition.
- [`NetworkMeasuresResult` grows many fields] → They follow the existing one-field-per-quantity layout and stay `None`
  for other kinds, which the result already does for every kind.
- [A sampled diameter is a lower bound] → The docstring of `diameter` says so whenever `samples` is set.
