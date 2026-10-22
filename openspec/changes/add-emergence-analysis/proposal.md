# Proposal

## Why

The analysis toolkit measures entropies, information transfer, criticality, and stochastic laws, but it cannot answer
the question the newer complex adaptive systems literature asks first: does a macro description of a system carry more
causal power than its micro description, and does a population of adaptive agents stay ergodic or freeze into one of
many states? Two papers in `references/papers` pose these questions and give measurable definitions:

- Russo (2026, arXiv 2604.19827) adapts Hoel's causal emergence framework: micro state variables, a coarse-graining
  function to macro states, and the effective information of the transitions at each scale, so emergence is the gain in
  effective information from micro to macro.
- Lesmana et al. (2026, arXiv 2604.15669) build EvoSK, reinforcement-learning agents on a Sherrington-Kirkpatrick
  landscape with Bak-Sneppen replacement of the least fit, and show that it self-organises to the edge of ergodicity
  breaking: a total-variation diagnostic of the evolved trait distribution, scale-free avalanches with exponent 3/2, and
  a cascade branching ratio equal to the spectral radius of the learning dynamics' Jacobian.

Both measures are general. Effective information applies to any Markov chain the library estimates, from lattice
dynamics to agent models, and ergodicity breaking applies to any ensemble of trajectories. EvoSK is a probe of the core:
it combines dense random couplings, per-agent learning with softmax choice, and extremal replacement, primitives the
core has, in a combination no implemented model exercises.

**Rebased on `redesign-system-model`.** The redesign does not cover any of this change's measures, so its scope is
unchanged. Only the vocabulary of the probe model changes: EvoSK is a fixed group of elements under
`DiscreteTime()`, with update and extremal laws, and its softmax choice is a realization the model package defines,
since learning heuristics left the core.

## What Changes

- Add partition coarse-graining: a map from micro states to macro states, given as a label array or a function, applied
  to discrete trajectories and to transition matrices, with the induced macro transition matrix.
- Add effective information of a transition matrix, the mutual information between a maximum-entropy intervention on
  the current state and the next state, with its determinism and degeneracy parts.
- Add causal emergence: the effective information gain of a coarse-graining, and a search over partitions that merges
  states greedily while the gain grows.
- Add ergodicity-breaking diagnostics: the ergodicity-breaking parameter of time-averaged observables over an ensemble,
  the Thirumalai-Mountain fluctuation metric, and the total-variation distance between time and ensemble distributions.
- Add the library model `evolutionary_sk_game` in `physics/criticality`, after Lesmana et al., with a gallery example
  that shows the self-organisation to the edge of ergodicity breaking. It is built on the core's `Group`, draws,
  `Update`, and `Extremal` laws under `DiscreteTime()`, with its softmax choice defined in the model package.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `complexity-measures`: partition coarse-graining, effective information, and causal emergence.
- `stochastic-analysis`: ergodicity-breaking diagnostics of trajectory ensembles.
- `model-library`: the evolutionary Sherrington-Kirkpatrick game.

## Impact

- `skcomplex.core.analysis` gains `partition_coarse_grain`, `effective_information`, `causal_emergence`,
  `ergodicity_breaking`, and `fluctuation_metric`.
- `skcomplex.models.physics.criticality` gains `evolutionary_sk_game`, with an inventory entry citing arXiv 2604.15669
  and tagged with the redesign's vocabulary.
- No new core primitive, solver, or backend, so no new declaration.
- Dependencies follow the constitution: the design surveys maintained libraries for effective information and causal
  emergence first, and a library the core uses is a runtime dependency.
