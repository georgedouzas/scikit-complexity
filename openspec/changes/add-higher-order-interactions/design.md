# Design

## Context

After `redesign-system-model`, roles bind interactions to components, groups vectorise laws over members, and an
interaction binds its laws `over` members, pairs, neighbours, or hyperedges. `NetworkSpace(graph, hyperedges=)` holds
hyperedges as sequences of nodes, `Group(on='hyperedges')` holds one member per hyperedge, and validation reports the
binding `'hyperedges'` outside a network with hyperedges. The laws of such an interaction do not run yet. The pending
`add-link-fields` adds cell complexes as network spaces, and `add-adaptive-networks` adds interactions over edges that
read edge groups.

## Goals / Non-Goals

**Goals:**
- Run laws over hyperedges of any size, with the same symbolic and compiled path as pairwise laws.
- Let a node gate a link, so structure and dynamics regulate each other.
- Add no space kind, component kind, or law kind: build on the redesign's hyperedges and bindings.

**Non-Goals:**
- Persistent homology of hypergraphs, which belongs to analysis libraries.

## Decisions

### Hypergraphs through XGI

XGI (CompleXGroupInteractions) is the maintained library for hypergraphs and simplicial complexes, with generators,
statistics, and incidence matrices. A network space with hyperedges is built from an XGI hypergraph or simplicial
complex, or from lists of node groups, and converts back to XGI, as `NetworkSpace` wraps NetworkX; HyperNetX is the
fallback the survey compares. The space computes its incidence matrix and the hyperedges around each node, cached as
derived state.

### k-body laws lower to gathers

An interaction with `over = 'hyperedges'` declares the hyperedge size it acts on, or every size. It compiles to an index
table of members per hyperedge, so a k-body law is a gather and a scatter-add, the pattern pair interactions already
use. Hyperedges are grouped by size, one index table per size. Reductions over the hyperedges of a node, in
`core.laws`, use the incidence matrix.

Declarations: the NumPy and Numba backends add `ComponentKind.HYPEREDGE_GROUP` to their supported components, and the
hyperedge reduction declares `Requires(space={SpaceKind.NETWORK})`. Validation reports, with the existing code
`binding`, an interaction whose declared size no hyperedge of the space has, and, with `law-space`, a hyperedge
reduction in a space without hyperedges.

### Triadic regulation

Regulation is a composition, not a primitive. The links carry a Boolean activity in a group on the edges of the network.
Each regulated link and its regulating node form a hyperedge of size three, whose first two members are the link's ends
and whose third is the regulator. An interaction over these hyperedges binds the edge group through an edge role, whose
member is the edge between the first two members, and its `Update` law sets the activity from the regulator's state.
Applied each step under discrete time, percolation alternates with the regulation as in Millán et al.

Declaration: the edge role requires an edge group on the same network. Validation reports, with `binding`, an edge role
whose two members are not joined by an edge of the bound group.

### Coupling tensors

`CouplingTensor(order, mean, variance, symmetric=True)` in `core.components` is a quenched parameter value of a group:
an array of shape `(N,) * order` over its `N` members, drawn once from the simulation's random stream. Laws read it
through contractions with the members' states.

Declaration: a coupling tensor requires a group. Validation reports a coupling tensor on a single element with the code
`parameter-shape`, and suggests a group.

## Risks / Trade-offs

- [Large hyperedges make gathers wide] → Hyperedges are grouped by size, one index table per size.
- [A coupling tensor of order k over N members holds N^k entries] → The documentation states the memory, and the
  probe model keeps N at 100 for order 3.
