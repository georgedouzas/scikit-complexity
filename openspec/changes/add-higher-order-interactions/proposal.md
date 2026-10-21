# Proposal

## Why

Interactions in many complex systems are not pairwise. Majhi, Perc, and Ghosh (2022, arXiv 2203.06601) review dynamics
on hypergraphs and simplicial complexes: synchronization, contagion, and games whose rates depend on groups of three or
more agents at once. Millán, Sun, Torres, and Bianconi (2023, arXiv 2311.14877) show that triadic regulation, a node that
switches a link on or off, turns percolation into a dynamical process with period doubling and chaos. Sidhom and Galla
(2024, arXiv 2409.10990) solve random Lotka-Volterra communities with k-body interaction tensors. The core places
interactions on pairs through roles and networks, and the pending `add-link-fields` adds simplicial complexes, but no
primitive declares a law over a group of more than two members or a node that gates a link.

## What Changes

- Add hypergraph spaces: hyperedges of any size, built or loaded through XGI, with incidence matrices and the
  hyperedges around each node.
- Add k-body interactions: an interaction role bound to every hyperedge of a size, with laws over its members, and
  reductions over the hyperedges of a node, so a law reads the sum over the triangles a node belongs to.
- Add regulatory hyperedges: a node that switches a link on or off as a function of its state, for triadic percolation.
- Add k-body tensor couplings: random symmetric or general coupling tensors of order k as quenched parameters.
- Add the probe models `hypergraph_kuramoto`, `simplicial_contagion`, `triadic_percolation`, and
  `higher_order_random_lotka_volterra`.

## Capabilities

### New Capabilities

- `higher-order-interactions`: hypergraph spaces, k-body interactions, reductions over hyperedges, regulatory
  hyperedges, and k-body coupling tensors.

### Modified Capabilities

- `model-library`: four probe models of higher-order interactions.

## Impact

- `skcomplex.core.model` gains `HypergraphSpace`, `HyperedgeInteraction`, `Regulation`, and `CouplingTensor`.
- XGI becomes a runtime dependency if the design's survey confirms it, since the core uses it.
