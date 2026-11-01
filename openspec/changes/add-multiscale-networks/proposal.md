# Proposal

## Why

Gabrielli, Garlaschelli, Patil, and Serrano (2024, arXiv 2412.12988) review network renormalization: coarse-graining a
network across scales by its geometry or its Laplacian, so self-similarity and scale-dependent properties become
measurable, with popularity-similarity growth as the model that makes it exact. Ribeiro and Netto (2024, arXiv
2404.02642) review urban scaling laws, where quantities grow as powers of city size across a population of cities, a
cross-sectional fit the toolkit does not have.

The current code already provides most of the ground. `NetworkSpace` takes any NetworkX graph, so a generator needs no
new space. `TopologyInteraction(action='add-node')` without a target grows a network by one node per step from outside
it, joins `edges` existing nodes drawn in proportion to `toward`, and draws the values of the new node.
`NetworkAnalysis` measures degree exponents and clustering, `ScalingAnalysis` fits exponents over samples by size, and
`skcomplex.experiments` sweeps parameters with `GridDesign`. So this change no longer adds network space generators in
the core or the functions `renormalize_network` and `scaling_law`, since the analysis surface holds classes only. It
adds a renormalization analysis, a cross-sectional scaling kind, the reading of the new node in the attachment weight,
and two models.

## What Changes

- Let the attachment weight `toward` of `TopologyInteraction(action='add-node')` read the drawn values of the node being
  added, as `namespace.source`, beside the candidate node, so new nodes attach by their distance to existing ones.
- Add `RenormalizationAnalysis` to `skcomplex.analysis`, with the kinds `'laplacian'` and `'geometric'`, which returns
  the network at every scale with its degree distribution, degree exponent, and clustering.
- Add the kind `'allometric'` to `ScalingAnalysis`: the exponent of a quantity against system size across runs, with a
  bootstrap confidence interval.
- Add `popularity_similarity_network` to `mathematics/networks` and `urban_scaling_interactions` to
  `social/organizations`, both grown by `TopologyInteraction`, with curated gallery examples.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `adaptive-networks`: attachment weights that read the new node.
- `network-analysis`: renormalization across scales.
- `trajectory-analysis`: cross-sectional scaling across sizes.
- `model-library`: two models of multiscale networks.

## Impact

- `skcomplex.core.interactions.TopologyInteraction` and the topology kernels of `skcomplex.engine` pass the new node to
  the weight.
- `skcomplex.analysis` gains `RenormalizationAnalysis`, and `ScalingAnalysis` gains the kind `'allometric'`.
- The model library, its tests, the gallery, and the inventory.
