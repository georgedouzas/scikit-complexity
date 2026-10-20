# Tasks

## 1. Reductions in the modelling layer

- [x] 1.1 Add `total`, `mean`, `maximum`, `minimum`, `variance`, and `count` as Sage functions with `over`, `where`,
  and `weights`, add `count` to `REDUCTIONS`, re-export them from `skcomplex.core.model`, and verify their doctests and
  that unevaluated calls survive `simplify`
- [x] 1.2 Let a role bind to a set of single elements of one type, and verify the element set and mixed types scenarios

## 2. Lowering and ordering

- [x] 2.1 Collect and deduplicate reduction calls into reduction specs with `Source.REDUCED`, check the one-group rule
  and the weight shapes, and verify the two groups and wrong shape scenarios
- [x] 2.2 Evaluate reductions on NumPy in topological order before the kernels, with masks and dense or sparse weights,
  and verify the total, mean, maximum, count, no selected members, payoff matrix, and mean fitness scenarios
- [x] 2.3 Evaluate reductions at the state of each law kind, with held values across partitions and recomputation
  after jump events, and verify the synchronous map and propensity scenarios
- [x] 2.4 Fuse reductions into the Numba kernels with `prange` reductions and incremental updates after jump events,
  and verify the backends agree and linear cost scenarios
- [x] 2.5 Lower symbolic observables read in their own partition into the reduction pass, keep held reads across
  partitions, reject callable observables read in their own partition, and verify the Kuramoto through observables,
  callable observable, and observable read in its own partition scenarios, and that the reduction pass and
  `ObservableEvaluator` agree on every library model with observables
- [x] 2.6 Document reductions, element sets, and observable reads in the user guide with runnable examples, and verify
  the documentation test passes

## 3. Jacobians and analysis

- [x] 3.1 Derive the chain-rule Jacobian through reductions, state the pattern of `Source.REDUCED` with the size limit,
  and verify the Jacobian against finite differences scenario
- [x] 3.2 Expand groups of at most 64 members with reductions in `dynamics_of`, and verify the hawks and doves and
  group above the limit scenarios and the existing analysis tests
- [x] 3.3 Document the Jacobian structure, the size limit, and the advice on explicit steppers in the user guide, and
  verify the documentation test passes

## 4. Library

- [x] 4.1 Rebuild `kuramoto`, `kuramoto_sakaguchi_chimera`, and `winfree` on group means without positions or pair
  interactions, and verify the Kuramoto, transition, chimera, and Winfree scenarios
- [x] 4.2 Rebuild `theta_neurons`, `legion`, and `winners_take_all` on reductions, and verify the theta neuron, LEGION,
  and winners take all scenarios
- [x] 4.3 Rebuild `replicator_dynamics`, `hypercycles`, and `heterogeneous_mean_field_sis` with members in groups and
  weighted reductions, and verify the replicator, hawks and doves, hypercycle, and degree class scenarios
- [x] 4.4 Rebuild `bouchaud_mezard_wealth` on the group mean for the default complete graph, keeping the field for a
  given graph, and verify the wealth exchange scenario
- [x] 4.5 Rebuild `mirollo_strogatz` with the oscillators as members of a group and the pulse coupling read through a
  reduction instead of one vector state in `EuclideanSpace(n)`, and verify the Mirollo-Strogatz scenario
- [x] 4.6 Update the docstrings and examples of the rebuilt models, the inventory entries, and the changelog with the
  changed states, and verify the inventory check, the conformance suite, and the doctests pass

## 5. Gate

- [x] 5.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
