## Why

The k-body terms of collective motion act on the current neighbours of an agent, not on fixed hyperedges. The swarm of
Gutleb (2026, arXiv 2611.08314) has a 3-body flanking term over pairs of neighbours and a 4-body encirclement term over
triples of neighbours, each within a cutoff radius that moving agents enter and leave. Today `hyperedges` and
`CouplingInteraction(mode='hyperedges')` read only the fixed hyperedges of a network space, and `neighbors` reads one
neighbour at a time. Writing a 3-body term as pooled pair sums is awkward and wrong for terms that are not products of
pair terms, such as `1 - r_ij . r_ik`.

## What Changes

- The reduction `hyperedges` gains `over`, a neighbourhood of a continuous space. With it, the hyperedges around a
  member are the member together with every set of `size - 1` of its current neighbours, and the expression reads
  them through `namespace.others` and `namespace.members` as on a network.
- The sets are recomputed every step from the neighbour search the core already runs for `neighbors`, so members that
  enter or leave the radius join or leave the sets.
- `CouplingInteraction(mode='hyperedges')` takes `neighborhood` in the same sense, so a k-body coupling over moving
  neighbours is one interaction.
- The NumPy and Numba backends compile it. Other backends report an issue at compile time.
- New system `ManyBodyAlignmentSwarmSystem` in `biology/collective_behaviour`: self-propelled particles with a Morse
  potential, Cucker-Smale alignment, a 3-body flanking term, and a 4-body encirclement term, all under one smooth
  cutoff. It is also the probe of `extend-law-discovery`, which learns its interaction law from trajectories.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `higher-order-interactions`: hyperedge reductions and hyperedge couplings run over sets of current neighbours of a
  continuous space, recomputed every step.
- `model-library`: adds the many-body alignment swarm.

## Impact

- Code: `src/skcomplex/core/components/laws/utils/_reductions.py` (`hyperedges`, `hyperedges_in`),
  `src/skcomplex/core/specs/_hyperedge_reduction_spec.py`, `src/skcomplex/core/components/interactions/_coupling.py`,
  the reduction lowering in `src/skcomplex/core/engine/compile/passes/_reduction_lowering.py`,
  `src/skcomplex/core/engine/problems/_neighborhood_kernel.py`, and the Numba neighbour loops in
  `src/skcomplex/core/engine/backends/_numba_neighbours.py`. A new module in
  `src/skcomplex/systems/biology/collective_behaviour/`.
- No new dependencies.
- Docs: the higher-order section of the guide on networks and interactions, a gallery example, an inventory entry, and
  a new inventory primitive.
- Not breaking. Without `over`, `hyperedges` and the coupling behave as before.
