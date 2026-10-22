# Tasks

Each task writes its tests first, checks that they fail on the current code, and then implements. Every test follows
the naming and public-API rules of the constitution.

## 1. Coarse-graining and effective information

- [x] 1.1 Survey maintained libraries for effective information and causal emergence, record the result in the design,
  and wrap one if it qualifies
- [x] 1.2 Add `partition_coarse_grain` for trajectories and transition matrices with uniform and stationary weighting,
  and verify the lumpable chain and trajectory scenarios
- [x] 1.3 Add `effective_information` with determinism and degeneracy and the named warning for unobserved states, and
  verify the permutation and fully noisy scenarios
- [x] 1.4 Add `causal_emergence` with the greedy and the small-exhaustive partition searches, and verify the noisy micro
  states and permutation scenarios

## 2. Ergodicity breaking

- [x] 2.1 Add `ergodicity_breaking` and `fluctuation_metric`, and the total-variation distance of trait distributions,
  reading trajectories through the adapters, and verify the ergodic diffusion and frozen disorder scenarios

## 3. Library

- [ ] 3.1 Add `evolutionary_sk_game` to `skcomplex.models.physics.criticality` on the core's group, draw, update, and
  extremal primitives under discrete time, with its softmax choice in the model package, and verify the
  self-organisation scenario and the conformance suite
- [ ] 3.2 Add its curated gallery example with a review row, and mark its inventory entry implemented

## 4. Documentation and gate

- [ ] 4.1 Document coarse-graining, effective information, causal emergence, and ergodicity breaking in the analysis
  section of the user guide, and verify the documentation test passes
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
