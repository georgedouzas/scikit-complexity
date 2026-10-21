# Proposal

## Why

Gabrielli, Garlaschelli, Patil, and Serrano (2024, arXiv 2412.12988) review network renormalization: coarse-graining a
network across scales by its geometry or its Laplacian, so self-similarity and scale-dependent properties become
measurable, with hyperbolic and popularity-similarity generators as the models that make it exact. Ribeiro and Netto
(2024, arXiv 2404.02642) review urban scaling laws, where quantities grow as powers of city size across a population of
cities, a cross-sectional fit the toolkit does not have. The network analysis measures one network at one scale, and
`fit_power_law` fits distributions, not quantities against size.

## What Changes

- Add network renormalization: Laplacian renormalization and geometric renormalization of embedded networks, returning
  the coarse-grained networks of each scale and their measures.
- Add hyperbolic and popularity-similarity network generators, seeded from the simulation.
- Add cross-sectional scaling fits of a quantity against system size over a sweep of runs, with exponents and their
  uncertainties.
- Add the probe models `popularity_similarity_network` and `urban_scaling_interactions`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `network-analysis`: renormalization across scales and cross-sectional scaling.
- `model-library`: two probe models of multiscale networks.

## Impact

- `skcomplex.core.analysis` gains `renormalize_network` and `scaling_law`; `skcomplex.core.model` gains the hyperbolic
  and popularity-similarity generators.
