# Tasks

## 1. Growth by distance

- [ ] 1.1 Pass the drawn values of the node being added to `toward`, the condition, and the probability of
  `TopologyInteraction(action='add-node')` as `namespace.source`, on the NumPy, Numba, and Mesa paths, and verify the
  attachment by distance scenario and that the existing topology tests still pass

## 2. Analyses

- [ ] 2.1 Add `RenormalizationAnalysis` with the kinds `'laplacian'` and `'geometric'` to `skcomplex.analysis`, with a
  doctest, and verify the self-similar network and Laplacian blocks scenarios
- [ ] 2.2 Add the kind `'allometric'` with `confidence`, `resamples`, and `interval_` to `ScalingAnalysis`, and verify
  the known exponent scenario

## 3. Library

- [ ] 3.1 Add `popularity_similarity_network` to `mathematics/networks`, and verify the hyperbolic degrees scenario
- [ ] 3.2 Add `urban_scaling_interactions` to `social/organizations`, and verify the superlinear interactions scenario
  over a `GridDesign` sweep
- [ ] 3.3 Add a curated gallery example per model with a review row, set both inventory entries to implemented with the
  tags of the systems they build, and verify the inventory check passes

## 4. Documentation and gate

- [ ] 4.1 Document growth by distance, renormalization, and allometric scaling in the user guide, and verify the
  documentation test passes
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
