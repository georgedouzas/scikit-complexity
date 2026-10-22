# Proposal

## Why

Interactions in many complex systems are not pairwise. Majhi, Perc, and Ghosh (2022, arXiv 2203.06601) review dynamics
on hypergraphs and simplicial complexes: synchronization, contagion, and games whose rates depend on groups of three or
more agents at once. Millán, Sun, Torres, and Bianconi (2023, arXiv 2311.14877) show that triadic regulation, a node that
switches a link on or off, turns percolation into a dynamical process with period doubling and chaos. Sidhom and Galla
(2024, arXiv 2409.10990) solve random Lotka-Volterra communities with k-body interaction tensors. The core places
interactions on pairs through roles and networks, and the pending `add-link-fields` adds cell complexes, but no law
runs over a group of more than two members, no node gates a link, and no parameter holds a k-body coupling tensor.

**Rebased on `redesign-system-model`.** The redesign already gives networks hyperedges, `NetworkSpace(graph,
hyperedges=)`, groups with one member per hyperedge, `Group(on='hyperedges')`, and the interaction binding
`over='hyperedges'` with its validation, and it leaves to this change the laws run once per hyperedge. This change
therefore no longer adds a `HypergraphSpace` or a `HyperedgeInteraction`: hypergraphs are network spaces with
hyperedges, and k-body laws are laws of interactions over hyperedges. Triadic regulation no longer needs a
`Regulation` primitive either: it is an interaction over hyperedges of size three that updates an edge group of link
activities. This change still adds XGI construction and incidence matrices, the compiled execution of laws over
hyperedges grouped by size, reductions over the hyperedges of a node, the edge role of hyperedge interactions that
regulation needs, k-body coupling tensors, and the probe models.

## What Changes

- Build network spaces with hyperedges from an XGI hypergraph or simplicial complex, or from lists of node groups, with
  incidence matrices and the hyperedges around each node, in `core.space`.
- Run the laws of interactions with `over='hyperedges'` once per hyperedge of a declared size, with all members of the
  hyperedge readable, compiled to gathers and scatter-adds on NumPy and Numba.
- Add reductions over the hyperedges of a node in `core.laws`, so a law reads the sum over the triangles a node belongs
  to.
- Let an interaction over hyperedges bind an edge group of the same network, whose member is the edge between two
  declared members of the hyperedge, so a regulating node switches a link on or off with an `Update` law, for triadic
  percolation.
- Add k-body coupling tensors in `core.components`: random symmetric or general coupling tensors of order k over the
  members of a group, as quenched parameters.
- Declare the new primitives for validation, so `validate()` reports a hyperedge size the space lacks, a hyperedge
  reduction outside a network with hyperedges, and a coupling tensor on a component that is not a group.
- Add the probe models `hypergraph_kuramoto`, `simplicial_contagion`, `triadic_percolation`, and
  `higher_order_random_lotka_volterra`.

## Capabilities

### New Capabilities

- `higher-order-interactions`: laws over hyperedges, reductions over hyperedges, XGI construction, regulation of links
  by nodes, k-body coupling tensors, and their validation declarations.

### Modified Capabilities

- `model-library`: four probe models of higher-order interactions.

## Impact

- `skcomplex.core.space` gains the construction of network spaces with hyperedges from XGI and incidence matrices.
- `skcomplex.core.interactions` gains the hyperedge size and the edge role of interactions over hyperedges.
- `skcomplex.core.laws` gains reductions over the hyperedges of a node.
- `skcomplex.core.components` gains `CouplingTensor`.
- `skcomplex.core.compile` and the NumPy and Numba backends in `skcomplex.core.backends` gain the hyperedge gathers and
  declare their support for hyperedge groups.
- XGI becomes a runtime dependency if the design's survey confirms it, since the core uses it.
