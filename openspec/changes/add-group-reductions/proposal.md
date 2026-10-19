# Proposal

## Why

Mean-field models couple every member to a reduction over a population: the mean pulse of all theta neurons
(arXiv:1902.05307 §2.3), the maximum activity that drives the LEGION inhibitor (Gros 2015 §9.5), the mean fitness of
replicator dynamics (Gros 2015 §8.7), and the degree-weighted infection density of heterogeneous mean-field SIS
(arXiv:1408.2701 §V.1). A law can only add a linear term to a group through a role today, so it cannot put a sum, a
mean, or a maximum inside a nonlinear expression. Laws also cannot read the system's observables in their own
partition. The library models work around both gaps:
- **Theta neurons:** a fast synapse element stands in for the instantaneous mean of the pulses.
- **LEGION:** the global inhibitor sums the active oscillators instead of taking their maximum.
- **Winners take all:** the mutual inhibition is `N (N - 1)` pair interactions, and the sigmoid of the summed
  inhibition cannot be written.
- **Replicator dynamics, hypercycles, and heterogeneous mean-field SIS:** one element holds a vector state in
  `EuclideanSpace(n)` only to get `n` components, so the strategies, molecules, and degree classes are not members.
- **Bouchaud-Mézard wealth:** a field on a complete graph with a Laplacian stands in for exchange with the mean wealth.
- **Kuramoto, Kuramoto-Sakaguchi chimera, and Winfree:** a pair interaction with a placeholder position and an
  all-covering cutoff stands in for the mean field, at a cost of `N²` pairs.

## What Changes

- Add symbolic reductions that laws use inside any expression: `total`, `mean`, `maximum`, `minimum`, `variance`, and
  `count`, over the members of a group or over a set of single elements, with an optional condition that selects the
  members and an optional constant weight matrix.
- Let a role bind to a set of single elements of one type, which laws then treat like a group.
- Let laws read the system's symbolic observables in their own partition, at the current state. Observables in other
  partitions stay held at synchronisation points, as today.
- Lower reductions in the compiler before the member kernels on NumPy and Numba, in dependency order, inside drift,
  noise, update, propensity, and boundary evaluations, and state their Jacobian structure.
- Expand small groups with reductions into per-member equations for symbolic analysis, so fixed points and stability of
  replicator dynamics stay derivable.
- Rebuild the library models listed above without their workarounds, under the same names.

Neighbourhood reductions, whose argument depends on both the reading member and its neighbour, are specified by
`add-adaptive-networks` as `neighbors(...)`, and stay there. `szabo_cells` and `active_passive_flocking` need them, so
they stay blocked on `add-adaptive-networks`. This change lands the reduction pass that `neighbors` extends with
per-reader segment reductions.

## Capabilities

### New Capabilities

- `group-reductions`: reductions over groups and element sets inside laws, observable reads in the same partition, and
  their lowering, ordering, and Jacobian structure on every backend.

### Modified Capabilities

- `system-modeling`: roles bind to element sets, and laws read observables in their own partition.
- `symbolic-analysis`: fixed points and stability of systems whose small groups read reductions.
- `model-library`: the mean-field models rebuilt on reductions.

## Impact

- `skcomplex.core.model` gains `total`, `mean`, `maximum`, `minimum`, `variance`, and `count`, which share the reduction
  table of `Observable`.
- `skcomplex.core.compile` gains a reduction pass and a reduced symbol source. The error that rejects observable reads
  inside one partition is removed for symbolic observables.
- `skcomplex.core.analysis` expands groups up to a member limit for `jacobian`, `fixed_points`, and `stability`.
- No new dependency. The lowering uses NumPy reductions and `np.add.at`, SciPy sparse matrix products, and Numba
  loops, which the project already depends on.
- Models in `physics/synchronization`, `biology/neuroscience`, `biology/evolution`, `biology/epidemiology`, and
  `economics/finance` change their states and interactions but keep their names and parameters. Their docstrings,
  examples, and the inventory follow.
- Relates to `add-adaptive-networks`, whose `neighbors` reuses the reduction pass, and to
  `adopt-scipy-sparse-jacobians`, whose pattern gains the reduced source.
