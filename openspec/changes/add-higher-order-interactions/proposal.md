# Proposal

## Why

Interactions in many complex systems are not pairwise. Majhi, Perc, and Ghosh (2022, arXiv 2203.06601) review dynamics
on hypergraphs and simplicial complexes: synchronization, contagion, consensus, and games whose rates depend on groups
of three or more agents at once. Millán, Sun, Torres, and Bianconi (2023, arXiv 2311.14877) show that triadic
regulation, a node that switches a link on or off, turns percolation into a dynamical process with period doubling and
chaos.

The core already holds hyperedges: `NetworkSpace(graph, hyperedges=...)` stores them, `GroupElement(on='hyperedges')`
has one member per hyperedge, `HyperedgeInteraction` is the base of interactions over them, and the compiler reports a
hyperedge interaction on a space without hyperedges with the issue code `binding`. `GroupingInteraction` draws pairs,
partitions, neighbourhoods, and groups by state every step, and acts on each group with an agent rule. So this change
adds no space, element, or interaction base. What is missing is the execution: the compiler runs the laws of a
`HyperedgeInteraction` once per member and ignores the hyperedges, no law reduces over the hyperedges around a member,
and no symbolic law runs once per group of a grouping. Coupling tensors and the reduction `contraction`, with the model
`higher_order_random_lotka_volterra` that needs them, moved to `add-learning-dynamics`, whose shaped parameters they
build on.

## What Changes

- Accept an XGI hypergraph or simplicial complex as the `hyperedges` of `NetworkSpace`, beside sequences of nodes, with
  the incidence matrix and the hyperedges around every node as derived state.
- Compile the laws of a `HyperedgeInteraction` once per hyperedge, grouped by size, as gathers and scatter-adds on the
  NumPy and Numba backends. A law reads the members of its hyperedge, the focal member it writes, and the member of a
  group on hyperedges or on edges that a second role binds, so a regulating node switches a link (triadic regulation).
- Add the reduction `hyperedges` to `skcomplex.core.laws.utils`, the reduction over the hyperedges around every member,
  and the mode `'hyperedges'` of `CouplingInteraction`, which couples a member to the other members of its hyperedges.
- Give `GroupingInteraction` the symbolic form, `form='symbolic'`, whose function of a group compiles on the same
  path, with the groups drawn every step as the hyperedges of that step.
- Let `TopologyInteraction(action='add-node')` attach every new node through a hyperedge of a given size.
- Add the probe models of the inventory, and rebuild `galam_majority` and `public_goods_game` on the symbolic grouping.

## Capabilities

### New Capabilities

- `higher-order-interactions`: XGI hyperedges, laws per hyperedge, hyperedge reductions and coupling, symbolic
  groupings, and growth by hyperedges.

### Modified Capabilities

- `model-library`: the higher-order probe models, and Galam and the spatial public goods game on symbolic groupings.

## Impact

- `skcomplex.core.spaces.NetworkSpace` accepts XGI objects and derives `incidence_` and `memberships_`.
- `skcomplex.core.interactions` changes `HyperedgeInteraction` (a `size` argument and per-hyperedge laws),
  `CouplingInteraction` (the mode `'hyperedges'`), `GroupingInteraction` (the symbolic form), and
  `TopologyInteraction` (the `hyperedge` argument).
- `skcomplex.core.laws.utils` gains `hyperedges`.
- `skcomplex.engine.compile.passes` gains the hyperedge pass, and the NumPy and Numba backends gain its gather kernels.
- XGI becomes a runtime dependency if the survey confirms it, since the core uses it.
- No dependency on `add-learning-dynamics` remains, since coupling tensors, `contraction`, and
  `higher_order_random_lotka_volterra` moved there.
