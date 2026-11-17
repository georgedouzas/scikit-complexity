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
  - a parametric library, whose coefficients depend on named conditions that differ between the observed runs
  - control inputs, named variables or the inputs of observed runs, that enter the library and become bounded adaptive
    parameters of the system
  - experiment results of `ensemble` and `sweep` as sources, read as observed runs through `observe` of
    `add-learning-foundation`, one run per design point or per replicate
- `sparse_identification` gains `system` and `learned`: it evaluates the trained learned functions of a system at the
  states of the source and fits a sparse library to their outputs, so the missing term of a universal differential
  equation is recovered as an equation and the result's system holds the symbolic law in its place. This is the second
  step of the workflow of Rackauckas et al. (2001.04385), after `identify` of `add-learning-foundation` trains the
  network.
- `embedding` gains the kinds `'tica'` and `'vamp'`, the slow coordinates of time-lagged independent component analysis
  and of the variational approach to Markov processes, with their implied timescales, on deeptime.
- `mode_decomposition` gains Koopman eigenfunctions, their error on held-out pairs, and a readout of the original
  variables from the observables, so the lifted system records them as observables. It gains time-delay observables
  with PyDMD and the choice of a dictionary among candidates by the error on validation runs held out with
  `split_runs`, with scikit-learn transformers.
- New analysis `cluster_expansion` learns one shared acceleration law of the members of a group from particle
  trajectories. The law is permutation invariant and rotation equivariant, split by body order, and fitted by Bayesian
  ridge regression with scikit-learn. Its result holds the contribution of every body order and a runnable group system
  with the learned law.
- New system `online_offline_activity_network` in `social/spreading`, a stochastic process of online engagement and
  offline participation on a network with its mean-field limit, the probe of the multi-trajectory weak form.
- **BREAKING:** `SparseIdentificationResult` gains fields and its `features` now name the terms of the chosen library,
  which may not be monomials.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `data-driven-dynamics`: weak, ensemble, parametric, control, multi-trajectory, and learned-term options of sparse
  identification, slow coordinates by TICA and VAMP,
  Koopman eigenfunctions and dictionary choice in mode decomposition, and many-body interaction laws by cluster
  expansion.
- `model-library`: the system `online_offline_activity_network`.

## Impact

- Code: `src/skcomplex/core/analysis/_sparse_identification.py`, `_embedding.py`, `_mode_decomposition.py`, a new
  `_cluster_expansion.py`, and `src/skcomplex/systems/social/spreading/` with a new module.
- Dependencies: `deeptime>=0.5.0` (0.5.0, released 2026-11-07, LGPL-3.0, wheels for Python 3.10 to 3.14) becomes a
  runtime dependency for TICA and VAMP. PySINDy, PyDMD, and scikit-learn are already runtime dependencies.
- Depends on `add-learning-foundation`: every source is read as an `ObservedRunsSpec`, run conditions are the
  parameters of the parametric library, run inputs are the control inputs, `observe` turns experiment results into
  runs, `split_runs` holds out validation runs, and every result follows the field convention of
  `IdentificationResult` (`system`, `training_error`, `validation_error`), and `identify` and `learned_function` train
  the networks whose terms the learned-term option recovers. The private branch of `_read_samples` for
  experiment results that this change once planned is not needed.
- Depends on `add-many-body-neighbour-roles` (change 9): the system of a `cluster_expansion` above body order 2 holds
  its k-body neighbour interactions, and its probe is `many_body_alignment_swarm`. The 2-body part and the pairwise
  probe `bird_swarm` do not depend on it.
- Docs: the learned surrogates section of the user guide, gallery examples, and the inventory.
