## Context

Two mechanisms exist and neither covers k-body terms over moving neighbours:

- `neighbors(expression, over=RadiusNeighborhood(r))` in `components/laws/utils/_reductions.py` reduces a pair
  expression of `namespace` and `namespace.other`. `NeighborhoodKernel` in `engine/problems/_neighborhood_kernel.py`
  finds the reader and other rows every evaluation through `_neighbor_search`, which on Numba runs the cell lists of
  `engine/backends/_numba_neighbours.py` (`_radius_neighbours`, `_cone_neighbours`, `_nearest_neighbours`), and
  reduces them with `_reduce_pairs`.
- `hyperedges(expression, size=k)` reduces an expression of `namespace.others` and `namespace.members` over the fixed
  hyperedges of a `NetworkSpace`. `HyperedgeTables` in `engine/compile/passes/_hyperedges.py` builds gather tables once
  per size and expands the reductions over `namespace.members`. `CouplingInteraction(mode='hyperedges', size=k)` wraps
  it.

The swarm of 2611.08314 needs the second form with the rows of the first.

The core feature is cross-domain. Sets of current neighbours carry the k-body terms of flocks and fish schools, the
three-body potentials of physics and chemistry such as Axilrod-Teller and Stillinger-Weber, and group interactions in
crowds and social models. So the core names only the mathematical objects, hyperedges and sets of neighbours, and
everything swarm-specific, such as the flanking and encirclement terms and the smooth cutoff, stays in the system
module.

## Goals / Non-Goals

**Goals:**
- One concept, the hyperedges around a member, with two sources: the fixed hyperedges of a network or the current
  neighbour sets of a continuous space.
- Exact agreement with a loop, on the NumPy and Numba backends.

**Non-Goals:**
- Neighbour hyperedges on lattices or networks. A network already has its hyperedges, and a lattice can declare them.
- The JAX backend. Its fixed shapes do not fit sets whose number changes every step. It reports an issue.
- Weighting a set by body-order purification, as in the dACE fit. That belongs to `extend-law-discovery`.

## Decisions

- **Extend `hyperedges`, not `neighbors`.** The expression of a k-body term reads several other members, which is the
  namespace of `hyperedges`. Giving `neighbors` a `size` would add a second way to write the same thing. The new
  argument `over` takes the neighbourhoods `neighbors` takes in a continuous space: `RadiusNeighborhood`,
  `NearestNeighborhood`, and `ConeNeighborhood`. `HyperedgeReductionSpec` gains the field `over`, and the call
  `HYPEREDGES` gains the neighbourhood symbol of `_neighborhood_symbol` as its last argument.
- **Sets are unordered.** The sets are the combinations of `size - 1` neighbours, in increasing member order, as the
  sums `j < k` and `j < k < l` of Eq. 2. A law that needs ordered tuples sums its expression over permutations itself.
- **Kernel.** A new private method of `NeighborhoodKernel` turns the reader and other rows of one evaluation into the
  rows of the sets with `itertools.combinations` per reader on NumPy, vectorised by neighbour count, and into a nested
  loop over the neighbour list on Numba. The expression is gathered per set and reduced per reader with the existing
  `_reduce_pairs`. The cost is `O(N d^(size - 1))` for `d` neighbours per member.
- **Lowering.** `_reduction_lowering.py` dispatches a hyperedge reduction with `over` to the neighbourhood kernel
  rather than to `HyperedgeTables`, and checks the space and the size there, with the issue code `law-space` used today
  for hyperedges outside a network.
- **Coupling.** `CouplingInteraction` already takes `neighborhood` in the mode `'edges'`. In the mode `'hyperedges'` it
  passes it to `hyperedges` as `over`. No new parameter.
- **Swarm system.** `ManyBodyAlignmentSwarmSystem` lives in a new module of `systems/biology/collective_behaviour/`.
  Particles are a `GroupElement` with position and velocity in a `ContinuousSpace` of two or three dimensions. The
  1-body term is a rate law, the Morse and alignment terms are pair couplings over `RadiusNeighborhood(R)`, and the
  3-body and 4-body terms are hyperedge couplings over the same neighbourhood. The smooth cutoff
  `f_c(r) = (1 - r^2 / R^2)^2` multiplies every pair, and `g(r) = f_c(r) exp(-r / l_3)` weights each neighbour of a set.
  The module defines its own interactions and does not import `_flocks`, whose `MorsePotentialInteraction` has a hard
  cutoff, so it stays self-contained.
- **Inventory primitive.** The primitive `neighbour-hyperedges` names the new core feature, with
  `change = "add-many-body-neighbour-roles"`.
- **Libraries surveyed.** `freud-analysis` 3.6.1 (2026-09-22) gives neighbour lists, which the core already has, but
  no k-body reductions. `jax-md` is excluded because it pulls flax and pymatgen. ACE packages fit interaction laws and
  do not simulate them. So the sets are built in the core on its own neighbour search.

## Risks / Trade-offs

- [Cost grows with the neighbour count to the power `size - 1`] → The guide states the cost, and the swarm keeps
  `R` at a few body lengths, as the paper does. Size 4 at 10 neighbours is 84 sets per member.
- [Numba memory for set rows] → The Numba loop reduces each set as it visits it and stores no set table. The nested
  loops are printed by `_numba_printer.py` over the neighbour rows that `_numba_neighbours.py` already returns sorted,
  so `_numba_neighbours.py` needs no new function.
- [The namespace needs its slots before the laws exist] → A group in a continuous space gets `others` slots for
  hyperedges of at most `MAX_NEIGHBOR_HYPEREDGE_SIZE = 5` members, and `hyperedges` rejects a larger size over a
  neighbourhood. Size 5 at 10 neighbours is already 210 sets per member.

## Migration Plan

None. The argument is new and optional.
