# Proposal

## Why

`sparse_identification` and `mode_decomposition` learn a runnable system from trajectories, but only in their plainest
form. SINDy differentiates noisy series by finite differences and fits one library with one regression, so it fails on
the noisy and stochastic data of 2607.15077 and 2605.30432. Mode decomposition gives modes and eigenvalues but no
Koopman eigenfunctions, no check of which of them are real, and no way to choose a dictionary from the data
(2306.12962). No function learns the interaction law of a group from particle trajectories, although the core runs such
groups and 2611.08314 shows that a body-ordered, equivariant expansion recovers the law, body order by body order, from
few runs.

## What Changes

- `sparse_identification` gains options, all on PySINDy:
  - a weak form, which integrates the library against test functions instead of differentiating the series
  - an ensemble, which bags samples or library terms and returns the inclusion probability of every term
  - a parametric library, whose coefficients depend on named parameters that differ between runs
  - control inputs, named variables that enter the library and become bounded adaptive parameters of the system
  - experiment results of `ensemble` and `sweep` as sources, one run per design point, from the mean of its replicates
    or from every replicate
- `mode_decomposition` gains Koopman eigenfunctions, their error on held-out pairs, and a readout of the original
  variables from the observables, so the lifted system records them as observables. It gains time-delay observables
  with PyDMD and the choice of a dictionary among candidates by validation error, with scikit-learn transformers.
- New analysis `cluster_expansion` learns one shared acceleration law of the members of a group from particle
  trajectories. The law is permutation invariant and rotation equivariant, split by body order, and fitted by Bayesian
  ridge regression with scikit-learn. Its result holds the contribution of every body order and a runnable group system
  with the learned law.
- No new system. The methods are tested and shown on catalog systems: `lorenz`, `sir` from several initial states,
  the stochastic `maki_thompson` through the means of its replicates, `duffing`, and `bird_swarm`. The many-body probe
  `many_body_alignment_swarm` comes with `add-many-body-neighbour-roles`, which needs it for its own core primitive.
- **BREAKING:** `SparseIdentificationResult` gains fields and its `features` now name the terms of the chosen library,
  which may not be monomials.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `data-driven-dynamics`: weak, ensemble, parametric, control, and multi-trajectory options of sparse identification,
  Koopman eigenfunctions and dictionary choice in mode decomposition, and many-body interaction laws by cluster
  expansion.

## Impact

- Code: `src/skcomplex/core/analysis/_sparse_identification.py`, `_mode_decomposition.py`, and a new
  `_cluster_expansion.py`. No module of `skcomplex.systems` changes.
- Dependencies: none new. PySINDy, PyDMD, and scikit-learn are already runtime dependencies.
- Depends on `add-many-body-neighbour-roles` (change 9): the system of a `cluster_expansion` above body order 2 holds
  its k-body neighbour interactions, and its probe is `many_body_alignment_swarm`. The 2-body part and the pairwise
  probe `bird_swarm` do not depend on it.
- Docs: the learned surrogates section of the user guide, gallery examples of the probe systems, and the primitive
  `learned-interaction-laws` of the inventory.
